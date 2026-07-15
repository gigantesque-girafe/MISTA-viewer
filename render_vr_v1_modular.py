"""
render_vr_v1_modular.py  —  MISTA adapter for the reusable `vr_viewer` package.

This is the *thin, pipeline-specific* half of the VR viewer. All CUDA-IPC
transport, the wire protocol and the streaming loop live in the pipeline-agnostic
`vr_viewer/` package; this file supplies only what is unique to MISTA:

  * building the (factorized) scene and decoding the canonical avatar,
  * the view-independent feature extraction from MISTA Gaussians,
  * producing the five per-Gaussian attribute tensors for each animation frame.

The generic Color-MLP TorchScript export lives in `vr_viewer.colormlp_export`
(duck-typed, shared by the 3dgs-v2 / MISTA family). This file depends ONLY on
`vr_viewer` and the MISTA codebase.

To bring the viewer to another similar pipeline, copy this single file, swap the
MISTA imports/helpers for that pipeline's equivalents, and reimplement
`produce_frame`. The pipeline's own render.py / training code is never touched.

Wire v2 ("V42E"): the per-Gaussian tail is a precomputed 3D covariance
(cov3D[6], upper triangle [00,01,02,11,12,22]) — the C++ rasterizer's
cov3D_precomp path, matching render.py (compute_cov3D_python=True).

Usage (same CLI as render_vr_v1.py):
  python render_vr_v1_modular.py mode=predict dataset.predict_seq=0 \\
      dataset=zju_377_mono wandb_disable=True \\
      load_ckpt="./exp/<run>/ckpt<iter>.pth" appearance_identity=0
"""

import os
import logging

import torch
import torch.nn as nn

from utils.general_utils import build_rotation

from vr_viewer import VRSource, run_server
from vr_viewer.colormlp_export import export_color_mlp, view_indep_feat_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mista_vr_v1_modular")

_PORT  = 6012
_N_MAX = 50_000   # IPC buffer capacity; must be >= actual Gaussian count.

# Axis-convention correction for the predict-sequence source data (chest-to-back
# depth comes out along world Y instead of Z). See render_vr_v4_2.py for the full
# derivation. If the avatar renders upside down, negate row 1 (height/Y) entries.
_AXIS_FIX = [[-0.9601507782936096,  0.15131592750549316, -0.2349766492843628],
             [-0.27107205986976624, -0.29949113726615906,  0.914781391620636],
             [ 0.06804757565259933,  0.9420236349105835,   0.32857415080070496]]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — mirrors MISTA ColorMLP.compose_input (view-independent part).
# These read MISTA's Gaussian data model directly, so they stay in the adapter
# (not in the pipeline-agnostic vr_viewer package).
# ─────────────────────────────────────────────────────────────────────────────

def extract_view_indep_features(color_mlp, gaussians, camera) -> torch.Tensor:
    features = gaussians.get_features.squeeze(-1)
    if getattr(color_mlp, 'use_xyz', False):
        aabb     = color_mlp.metadata["aabb"]
        xyz_norm = aabb.normalize(gaussians.get_xyz, sym=True)
        features = torch.cat([features, xyz_norm], dim=1)
    if getattr(color_mlp, 'use_cov', False):
        features = torch.cat([features, gaussians.get_covariance()], dim=1)
    if getattr(color_mlp, 'use_normal', False):
        scale  = gaussians._scaling
        rot    = build_rotation(gaussians._rotation)
        normal = torch.gather(rot, dim=2,
            index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        features = torch.cat([features, normal], dim=1)
    if getattr(color_mlp, 'non_rigid_dim', 0) > 0:
        features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
    return features   # [N, K]


def extract_R_bwd(gaussians, cano_view_dir: bool) -> torch.Tensor:
    if cano_view_dir and hasattr(gaussians, 'fwd_transform'):
        T_fwd = gaussians.fwd_transform
        R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
        return R_bwd.reshape(-1, 9).contiguous()
    N   = gaussians.get_xyz.shape[0]
    dev = gaussians.get_xyz.device
    return torch.eye(3, dtype=torch.float32, device=dev)\
               .unsqueeze(0).expand(N, -1, -1).reshape(N, 9).contiguous()


def cov3D_in_vr_frame(cov6: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Rotate a packed covariance into the VR frame.

    `cov6` is [N, 6] in strip_symmetric order [00,01,02,11,12,22] (exactly what
    gaussians.get_covariance() returns via the rotation_precomp path, matching
    render.py). Points are mapped p -> A p by the axis fix, so the covariance
    must transform Σ -> A Σ Aᵀ. We rebuild the full 3×3, rotate, and re-strip.
    Doing it on the full matrix (not a quaternion) preserves the non-orthonormal
    scale/shear that LBS blending puts into rotation_precomp.
    """
    N = cov6.shape[0]
    Sig = cov6.new_zeros(N, 3, 3)
    Sig[:, 0, 0] = cov6[:, 0]
    Sig[:, 0, 1] = Sig[:, 1, 0] = cov6[:, 1]
    Sig[:, 0, 2] = Sig[:, 2, 0] = cov6[:, 2]
    Sig[:, 1, 1] = cov6[:, 3]
    Sig[:, 1, 2] = Sig[:, 2, 1] = cov6[:, 4]
    Sig[:, 2, 2] = cov6[:, 5]
    Sig = A.unsqueeze(0) @ Sig @ A.t().unsqueeze(0)
    out = cov6.new_empty(N, 6)
    out[:, 0] = Sig[:, 0, 0]; out[:, 1] = Sig[:, 0, 1]; out[:, 2] = Sig[:, 0, 2]
    out[:, 3] = Sig[:, 1, 1]; out[:, 4] = Sig[:, 1, 2]; out[:, 5] = Sig[:, 2, 2]
    return out.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# MISTA checkpoint loading — reproduces render.py predict()'s "extra start steps".
# Builds the Scene, allocates TT/CP cores from the checkpoint shapes, loads weights
# and the converter state. Returns (scene, migs_type, appearance_id).
# ─────────────────────────────────────────────────────────────────────────────

def build_scene_from_checkpoint(config):
    from scene import GaussianModel, Scene

    load_ckpt = config.get("load_ckpt", None)
    if load_ckpt is None:
        raise ValueError("Please provide load_ckpt.")

    log.info("Loading checkpoint: %s", load_ckpt)
    tmp = torch.load(load_ckpt, map_location="cpu")
    sd  = tmp["migs_module_state_dict"]

    migs_type = tmp.get("migs_type", config.migs.type)
    log.info("Detected migs_type = %s", migs_type)
    config.migs.type = migs_type

    appearance_id = getattr(config, "appearance_identity", None)

    # CASE 1: CP / Tucker — plain checkpoint loading.
    if migs_type in ("cp", "tucker"):
        config.migs.skip_init_from_tensor = False
        gaussians = GaussianModel(config.model.gaussian)
        scene = Scene(config, gaussians, config.exp_dir)
        scene.appearance_identity = appearance_id
        scene.eval()
        scene.load_checkpoint(load_ckpt)
        log.info("CP/Tucker checkpoint loaded")
        return scene, migs_type, appearance_id

    # CASE 2: TT (tt5d / tt / MARS-wrapped) — allocate cores from checkpoint shapes.
    has_mars_prefix = any(k.startswith("tensorized_model.tt.") for k in sd.keys())
    prefix = "tensorized_model.tt." if has_mars_prefix else ""
    if has_mars_prefix:
        log.info("Detected MARS-wrapped checkpoint")

    r1 = sd[f"{prefix}tt_tensor_gpu.0"].shape[-1]
    r2 = sd[f"{prefix}tt_tensor_gpu.1"].shape[-1]
    r3 = sd[f"{prefix}tt_tensor_gpu.2"].shape[-1]
    r4 = sd[f"{prefix}tt_tensor_gpu.3"].shape[-1]
    config.migs.rank = [1, r1, r2, r3, r4, 1]

    n_id = sd[f"{prefix}tt_tensor_gpu.0"].shape[1]
    n1   = sd[f"{prefix}tt_tensor_gpu.1"].shape[1]
    n2   = sd[f"{prefix}tt_tensor_gpu.2"].shape[1]
    n3   = sd[f"{prefix}tt_tensor_gpu.3"].shape[1]

    try:
        M_xyz = sd[f"{prefix}core4_xyz"].shape[1]
        M_scl = sd[f"{prefix}core4_scaling"].shape[1]
        M_rot = sd[f"{prefix}core4_rotation"].shape[1]
        M_dc  = sd[f"{prefix}core4_dc"].shape[1]
        M_rst = sd[f"{prefix}core4_rest"].shape[1]
        M_opa = sd[f"{prefix}core4_opacity"].shape[1]
        M = M_xyz + M_scl + M_rot + M_dc + M_rst + M_opa
    except KeyError:
        M_xyz, M_scl, M_rot, M_dc, M_rst, M_opa = 3, 3, 4, 1, 31, 1
        M = 43

    config.migs.tt_shape = [n_id, n1, n2, n3, M]
    config.migs.n_identities_ckpt = int(n_id)
    config.migs.use_mars = False
    config.migs.skip_init_from_tensor = True

    log.info("rank     = %s", config.migs.rank)
    log.info("tt_shape = %s", config.migs.tt_shape)

    if has_mars_prefix:
        sd = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

    gaussians = GaussianModel(config.model.gaussian)
    scene = Scene(config, gaussians, config.exp_dir)
    scene.appearance_identity = appearance_id
    scene.eval()

    tt_module = scene.migs_module
    tt_module.tt_shape = tuple(config.migs.tt_shape)
    tt_module.tt_rank  = config.migs.rank

    device = "cuda"
    tt_module.tt_tensor_gpu = nn.ParameterList([
        nn.Parameter(torch.zeros(1,  n_id, r1, device=device)),
        nn.Parameter(torch.zeros(r1, n1,   r2, device=device)),
        nn.Parameter(torch.zeros(r2, n2,   r3, device=device)),
        nn.Parameter(torch.zeros(r3, n3,   r4, device=device)),
    ])
    tt_module.core4_xyz      = nn.Parameter(torch.zeros(r4, M_xyz, 1, device=device))
    tt_module.core4_scaling  = nn.Parameter(torch.zeros(r4, M_scl, 1, device=device))
    tt_module.core4_rotation = nn.Parameter(torch.zeros(r4, M_rot, 1, device=device))
    tt_module.core4_dc       = nn.Parameter(torch.zeros(r4, M_dc,  1, device=device))
    tt_module.core4_rest     = nn.Parameter(torch.zeros(r4, M_rst, 1, device=device))
    tt_module.core4_opacity  = nn.Parameter(torch.zeros(r4, M_opa, 1, device=device))

    G = n1 * n2 * n3
    tt_module.register_buffer("perm",     torch.arange(G, dtype=torch.long, device=device))
    tt_module.register_buffer("inv_perm", torch.arange(G, dtype=torch.long, device=device))
    log.info("TT cores allocated (G=%d)", G)

    missing, unexpected = tt_module.load_state_dict(sd, strict=False)
    if missing:
        log.info("Missing keys   : %s%s", missing[:5], '...' if len(missing) > 5 else '')
    if unexpected:
        log.info("Unexpected keys: %s%s", unexpected[:5], '...' if len(unexpected) > 5 else '')
    log.info("TT weights loaded")

    scene.converter.load_state_dict(tmp["converter_state"])
    log.info("Converter loaded")

    return scene, migs_type, appearance_id


def collect_smpl_frames(dataset) -> list:
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
    def __init__(self, scene, smpl_cams, iteration, K, model_bytes, N_max):
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

        self.axis_fix       = torch.tensor(_AXIS_FIX, dtype=torch.float32, device=self.device)
        self.avatar_center0 = None                  # captured on frame 0 of each connection

    def on_connect(self):
        # Re-pin the avatar start to world origin for each fresh viewer connection.
        self.avatar_center0 = None

    def produce_frame(self, frame_idx: int):
        cam = self.smpl_cams[frame_idx % self.n_frames]

        with torch.no_grad():
            # ── Deformation ──────────────────────────────────────────────────
            cam_c, _ = self.converter.pose_correction(cam, self.iteration)
            deformed_pc, _ = self.converter.deformer(
                self.gaussians, cam_c, self.iteration, compute_loss=False
            )
            torch.cuda.synchronize()

            # ── View-independent feature extraction ──────────────────────────
            feat    = extract_view_indep_features(self.color_mlp, deformed_pc, cam_c)
            R_bwd_f = extract_R_bwd(deformed_pc, self.cano)

            # ── Placement (axis fix + origin pin) ────────────────────────────
            xyz = deformed_pc.get_xyz @ self.axis_fix.T
            if self.avatar_center0 is None:
                self.avatar_center0 = xyz.mean(dim=0, keepdim=True).clone()
            xyz     = xyz - self.avatar_center0
            opacity = deformed_pc.get_opacity.squeeze(-1)

            # ── Posed covariance (cov3D_precomp path), rotated into VR frame ──
            # get_covariance() uses rotation_precomp (posed, possibly non-orthonormal
            # from LBS), matching render.py's compute_cov3D_python=True. We then apply
            # the axis fix Σ -> A Σ Aᵀ so the splats sit in the same frame as xyz.
            cov6  = deformed_pc.get_covariance()
            cov3D = cov3D_in_vr_frame(cov6, self.axis_fix)

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
    scene, migs_type, appearance_id = build_scene_from_checkpoint(config)
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
