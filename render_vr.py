"""
render_vr.py : MISTA pipeline wired with C++ VR Viewer, pose driven from dataset

All CUDA-IPC transport, the wire protocol and the streaming loop live in the 
`vr_viewer/` package; this file supplies only what is unique to MISTA:

  * building the (factorized) scene and decoding the canonical avatar,
  * the view-independent feature extraction from MISTA Gaussians,
  * producing the five per-Gaussian attribute tensors for each animation frame.

The generic Color-MLP TorchScript export lives in `vr_viewer.colormlp_export`. 

Wire v2 ("V42E"): the per-Gaussian tail is a precomputed 3D covariance
(cov3D[6], upper triangle [00,01,02,11,12,22]), the C++ rasterizer's
cov3D_precomp path, matching render.py (compute_cov3D_python=True).

Usage (same CLI as render_vr_v1.py):
  python render_vr_v1_modular.py mode=predict dataset.predict_seq=0 \\
      dataset=zju_377_mono wandb_disable=True \\
      load_ckpt="./exp/<run>/ckpt<iter>.pth" appearance_identity=0
"""

import os
import logging

import torch

from vr_viewer import VRSource, run_server
from vr_viewer.colormlp_export import export_color_mlp
from vr_viewer.colormlp_export import _view_indep_feat_dim as view_indep_feat_dim
from pipeline.mista_features import AXIS_FIX, deform_and_pack
from render import build_scene

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mista_vr_v1_modular")

_PORT  = 6012
_N_MAX = 50_000   # IPC buffer capacity; must be >= actual Gaussian count.

# ─────────────────────────────────────────────────────────────────────────────
# MISTA checkpoint loading now lives in render.build_scene(), shared verbatim
# with predict() and the other viewer adapters — it is imported below, not
# reimplemented here.
# ─────────────────────────────────────────────────────────────────────────────


def collect_smpl_frames(dataset) -> list:
    """Load every camera/pose entry from a dataset, dropping unneeded image data.

    @param dataset: indexable, sized dataset whose items are camera objects with
        a `.data` dict.
    @return: list of camera objects, in dataset order, with `original_image` and
        `original_mask` removed from `.data`.
    """
    cams = []
    for i in range(len(dataset)):
        cam = dataset[i]
        cam.data.pop('original_image', None)
        cam.data.pop('original_mask', None)
        cams.append(cam)
    log.info("Loaded %d animation frames.", len(cams))
    return cams


# ─────────────────────────────────────────────────────────────────────────────
# MISTA frame source — the single pipeline-specific seam.
# ─────────────────────────────────────────────────────────────────────────────

class MistaVRSource(VRSource):
    """VRSource implementation that drives a MISTA scene from a precomputed animation sequence."""

    def __init__(self, scene, smpl_cams, iteration, K, model_bytes, N_max):
        """Store the decoded scene and animation cameras for use in `produce_frame`.

        @param scene: MISTA scene object, already decoded (`scene.gaussians`
            holds the canonical avatar).
        @param smpl_cams: list of camera objects, one per animation frame.
        @param iteration: training iteration passed to `pose_correction`/`deformer`.
        @param K: view-independent feature dimension (unused directly; forwarded
            to the VRSource handshake by the caller).
        @param model_bytes: serialized Color-MLP TorchScript model (unused
            directly; forwarded to the VRSource handshake by the caller).
        @param N_max: IPC buffer capacity (unused directly; forwarded to the
            VRSource handshake by the caller).
        """
        self.scene       = scene
        self.converter   = scene.converter
        self.gaussians   = scene.gaussians          # canonical avatar (decoded from MIGS)
        self.color_mlp   = self.converter.texture
        self.cano        = bool(getattr(self.color_mlp, 'cano_view_dir', False))
        self.iteration   = iteration
        self.smpl_cams   = smpl_cams
        self.n_frames    = len(smpl_cams)

        self.K           = K
        self.model_bytes = model_bytes
        self.N_max       = N_max
        self.device      = next(self.color_mlp.parameters()).device

        self.axis_fix       = torch.tensor(AXIS_FIX, dtype=torch.float32, device=self.device)
        self.avatar_center0 = None                  # captured on frame 0 of each connection

    def on_connect(self):
        """Reset the cached avatar origin so it re-pins to world origin on the next frame."""
        self.avatar_center0 = None

    def produce_frame(self, frame_idx: int):
        """Deform, place, and pack the canonical avatar for one animation frame.

        @param frame_idx: frame counter; wrapped with `% self.n_frames` to index
            into the animation loop.
        @return: tuple `(xyz, feat, R_bwd_f, opacity, cov3D)` — the five
            per-Gaussian attribute tensors sent to the C++/SIBR viewer.
        @note: On `frame_idx == 0`, `avatar_center0` is captured from the current
            frame's mean xyz and subtracted from all subsequent frames, and
            placement/covariance diagnostics are logged.
        """
        cam = self.smpl_cams[frame_idx % self.n_frames]

        xyz, feat, R_bwd_f, opacity, cov3D, self.avatar_center0, deformed_pc = deform_and_pack(
            self.converter, self.gaussians, self.color_mlp, self.cano, self.axis_fix,
            cam, self.iteration, self.avatar_center0,
        )

        if frame_idx == 0:
            lo = xyz.amin(dim=0).tolist()
            hi = xyz.amax(dim=0).tolist()
            ctr = xyz.mean(dim=0).tolist()
            log.info(
                "[PLACEMENT] center=(%.3f, %.3f, %.3f)  "
                "bbox_min=(%.3f, %.3f, %.3f)  bbox_max=(%.3f, %.3f, %.3f)  "
                "extent=(%.3f, %.3f, %.3f)",
                ctr[0], ctr[1], ctr[2],
                lo[0], lo[1], lo[2], hi[0], hi[1], hi[2],
                hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2],
            )
            # ── cov3D diagnostic (round-splat debugging) ──────────────────────
            has_rp   = hasattr(deformed_pc, 'rotation_precomp')
            scl      = deformed_pc.get_scaling
            scl_ratio = (scl.amax(dim=1) / scl.amin(dim=1).clamp_min(1e-12))  # per-Gaussian anisotropy
            diag = cov3D[:, [0, 3, 5]]           # xx, yy, zz
            off  = cov3D[:, [1, 2, 4]]           # xy, xz, yz
            log.info(
                "[COV3D] rotation_precomp=%s  scaling[min=%.2e max=%.2e]  "
                "aniso_ratio[med=%.2f max=%.2f]  diag[mean=%.2e max=%.2e]  |off|[mean=%.2e max=%.2e]",
                has_rp, float(scl.min()), float(scl.max()),
                float(scl_ratio.median()), float(scl_ratio.max()),
                float(diag.mean()), float(diag.max()),
                float(off.abs().mean()), float(off.abs().max()),
            )

        return xyz, feat, R_bwd_f, opacity, cov3D


# ─────────────────────────────────────────────────────────────────────────────
# Hydra entry point (MISTA-specific setup).
# ─────────────────────────────────────────────────────────────────────────────

import hydra
from omegaconf import OmegaConf, DictConfig
import wandb


@hydra.main(version_base=None, config_path="configs", config_name="config_5d")
def main(config: DictConfig):
    """Build the MISTA scene, decode the canonical avatar, and run the VR streaming server.

    @param config: Hydra config (config_5d), mutated in place (struct mode
        disabled; `exp_dir`, `mode`, and `suffix` are filled in if absent).
    @note: Reads `config.gaussians_vr.n_max`/`.port` for IPC buffer capacity and
        TCP port, falling back to `_N_MAX`/`_PORT`. Blocks until the server
        (`run_server`) stops.
    """
    OmegaConf.set_struct(config, False)
    config.dataset.preload = False
    config.exp_dir = config.get("exp_dir") or os.path.join("./exp", config.exp_name)
    os.makedirs(config.exp_dir, exist_ok=True)

    if config.mode not in ("predict", "test"):
        config.mode = "predict"

    predict_dict = {0: "sameperson", 1: "TransferMotion", 2: "dance0",
                    3: "dance1", 4: "flipping", 5: "canonical"}
    config.suffix = "predict-" + predict_dict.get(config.dataset.get("predict_seq", 0), "dance0")

    wandb.init(
        mode="disabled" if config.get("wandb_disable", True) else None,
        name=f"{config.exp_name}-mista-vr-v1-modular",
        project="mista-gaussians-vr",
        config=OmegaConf.to_container(config, resolve=True),
        settings=wandb.Settings(start_method="spawn"),
    )

    from utils.general_utils import fix_random
    fix_random(config.seed)

    # ── Extra MISTA start steps: load factorized checkpoint & build scene ─────
    scene, migs_type, appearance_id = build_scene(config)
    iteration = config.opt.iterations

    # ── Decode the canonical Gaussians from the MIGS factorization ONCE ───────
    smpl_cams = collect_smpl_frames(scene.test_dataset)
    if appearance_id is not None:
        identity_id = int(appearance_id)
    else:
        identity_id = int(getattr(smpl_cams[0], "person_id", 0))
    log.info("Decoding canonical Gaussians for identity %d ...", identity_id)
    with torch.no_grad():
        scene.update_gaussians_from_migs(identity_id)
    log.info("Canonical avatar ready: %d Gaussians.", scene.gaussians.get_xyz.shape[0])

    # ── Color-MLP export + feature dim (generic helper from vr_viewer) ────────
    color_mlp = scene.converter.texture
    device    = next(color_mlp.parameters()).device

    K = view_indep_feat_dim(color_mlp)
    log.info("Feature dim K=%d. Tracing Color MLP...", K)
    model_bytes = export_color_mlp(color_mlp, K, device)
    log.info("TorchScript model: %d bytes.", len(model_bytes))

    N_max = int(config.get("gaussians_vr", {}).get("n_max", _N_MAX))
    port  = int(config.get("gaussians_vr", {}).get("port", _PORT))

    # ── Hand off to the reusable, pipeline-agnostic streaming loop ────────────
    source = MistaVRSource(scene, smpl_cams, iteration, K, model_bytes, N_max)
    run_server(source, port=port)


if __name__ == "__main__":
    main()