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

Live identity switching (multi-identity checkpoint; the viewer's buttons send
CTRL_SET_IDENTITY, which lands in source.pending_id):
  python render_vr_v1_modular.py mode=predict dataset=migs_multi_zju_5d_mars \\
      migs.type=tt5d migs.use_mars=false dataset.predict_seq=0 wandb_disable=True \\
      +drive_identity=0 +start_identity=0 load_ckpt="<8-identity ckpt>.pth"

  drive_identity  which subject's motion drives the animation (default 0)
  start_identity  initial avatar; the GUI overrides it live
  Do NOT pass appearance_identity here — it filters the dataset to one subject and
  the other identities' reference poses never load.

Switching identity means swapping BOTH the decoded TT core and the reference pose
handed to the pose-conditioned non-rigid field. The latter is what actually changes
the visible appearance — see collect_identity_poses().
"""

import os
import logging

import torch
import torch.nn as nn

from vr_viewer import VRSource, run_server, ColorMLPModule

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
# Covariance placement — rebuild/rotate the packed covariance into the VR frame.
# The ColorMLP feature-extraction seam now lives in vr_viewer.ColorMLPModule;
# only this axis-fix-specific helper stays in the adapter.
# ─────────────────────────────────────────────────────────────────────────────

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


def collect_identity_poses(dataset) -> dict:
    """Capture one reference pose (`rots`/`Jtrs`) per identity.

    The rendered appearance is selected by the pose condition `rots`, not by the
    TT identity core (non_rigid.HashGridwithMLP conditions its field on
    pose_encoder(rots, Jtrs), and that field feeds the ColorMLP). Each subject was
    trained on a disjoint pose distribution, so the network reads identity off
    `rots`. Feeding identity N's rots while keeping the driving frame's
    bone_transforms gives N's appearance on the shared motion.

    dataset index i == TT identity i.
    """
    subs = getattr(dataset, "datasets", None)
    if not subs:
        return {}

    poses = {}
    names = getattr(dataset, "identities", None)
    for i, sub in enumerate(subs):
        # A subject whose images fail to decode (see the CoreView_315 note in
        # zjumocap.py) yields no reference pose and falls back to the driving
        # subject's appearance, so warn rather than abort the other seven.
        for f in range(min(len(sub), 4)):
            try:
                cam = sub[f]
            except Exception:
                continue
            poses[i] = {'rots': cam.rots.detach().clone(),
                        'Jtrs': cam.Jtrs.detach().clone()}
            break
        else:
            log.warning("No loadable frame for identity %d (%s); it will render with "
                        "the driving subject's appearance.",
                        i, names[i] if names else "?")
    log.info("Captured reference poses for %d/%d identities.", len(poses), len(subs))
    return poses


def collect_smpl_frames(dataset, drive_idx: int = 0) -> list:
    """Collect the driving SMPL frames (poses + cameras) for the animation loop.

    Only ONE subject's motion is needed: every identity is driven by the same
    sequence and appearance is selected separately (see collect_identity_poses).
    Restricting to a single sub-dataset also avoids walking subjects whose frames
    are incomplete (e.g. CoreView_315 starts at 000001, not 000000) and avoids
    decoding thousands of GT images that are popped immediately below anyway.

    drive_idx indexes dataset.names — 0:386 1:387 2:377 3:392 4:315 5:394 6:393 7:390.
    """
    subs = getattr(dataset, "datasets", None)
    if subs:
        idx   = max(0, min(int(drive_idx), len(subs) - 1))
        names = getattr(dataset, "identities", None)
        log.info("Driving motion from sub-dataset %d (%s) of %d available.",
                 idx, names[idx] if names else "?", len(subs))
        dataset = subs[idx]
    else:
        idx = 0

    cams = []
    for i in range(len(dataset)):
        cam = dataset[i]
        cam.data.pop('original_image', None)
        cam.data.pop('original_mask', None)
        # Indexing the sub-dataset directly bypasses the multi-person wrapper that
        # normally stamps person_id, so set it here.
        if hasattr(cam, 'data'):
            cam.data['person_id'] = idx
            setattr(cam, 'person_id', idx)
        cams.append(cam)
    log.info("Loaded %d animation frames.", len(cams))
    return cams


# ─────────────────────────────────────────────────────────────────────────────
# MISTA frame source — the single pipeline-specific seam.
# ─────────────────────────────────────────────────────────────────────────────

class MistaVRSource(VRSource):
    def __init__(self, scene, smpl_cams, iteration, colormlp, model_bytes, N_max,
                 identity_id: int = 0, identity_poses=None):
        self.scene       = scene
        self.converter   = scene.converter
        self.gaussians   = scene.gaussians          # canonical avatar (decoded from MIGS)
        self.colormlp    = colormlp                 # vr_viewer.ColorMLPModule
        self.iteration   = iteration
        self.smpl_cams   = smpl_cams
        self.n_frames    = len(smpl_cams)

        self.K           = colormlp.K
        self.model_bytes = model_bytes
        self.N_max       = N_max
        self.device      = next(colormlp.color_mlp.parameters()).device

        self.axis_fix       = torch.tensor(_AXIS_FIX, dtype=torch.float32, device=self.device)
        self.avatar_center0 = None                  # captured on frame 0 of each connection

        # Current appearance identity + a pending request set by the control channel
        # (see run_server). Applied at the top of produce_frame, between frames, so it
        # never races the streaming loop's use of self.gaussians.
        self.identity_id   = int(identity_id)   # whichever main already decoded
        self.pending_id    = None

        # What actually selects the rendered appearance (measured, not assumed):
        # the pose condition `rots`. non_rigid.HashGridwithMLP computes
        #     pose_feat = pose_encoder(rots, Jtrs)
        # and conditions its field on it; that field's output is the non_rigid half
        # of the ColorMLP input. Each ZJU subject trains on a disjoint pose
        # distribution, so the model learned to read identity off `rots` rather than
        # off the TT identity core — swapping the core alone leaves the avatar
        # wearing the driving subject's clothes. Overriding `rots` per identity is
        # therefore what makes the button work, and it leaves the rigid deformer
        # (which consumes bone_transforms) free to follow the driving motion.
        self.identity_poses = identity_poses or {}
        self._pose_override = self.identity_poses.get(self.identity_id)
        if self.identity_poses:
            log.info("Captured reference poses for identities: %s",
                     sorted(self.identity_poses))

    @property
    def num_identities(self):
        migs = getattr(self.scene, "migs_module", None)
        return int(getattr(migs, "num_identities", 1)) if migs is not None else 1

    def switch_identity(self, new_id: int):
        """Hot-swap the canonical avatar to a different MIGS identity.

        Cheap because identity lives entirely in the per-Gaussian tensors that
        update_gaussians_from_migs reassigns in place: G and N_max are fixed, and
        the Color MLP is identity-agnostic so model_bytes never change (no re-export,
        no client re-handshake).
        """
        new_id = max(0, min(int(new_id), self.num_identities - 1))
        if new_id == self.identity_id:
            return
        log.info("Switching identity %d -> %d", self.identity_id, new_id)

        with torch.no_grad():
            self.scene.update_gaussians_from_migs(new_id)

        # The canonical Gaussians carry only part of the identity — the visible
        # appearance rides on the pose condition (see __init__). Swap both.
        self._pose_override = self.identity_poses.get(new_id)
        if self._pose_override is None and self.identity_poses:
            log.warning("No reference pose for identity %d; appearance will follow "
                        "the driving subject.", new_id)

        self.gaussians      = self.scene.gaussians   # same object, rebind for clarity
        self.identity_id    = new_id
        self.avatar_center0 = None                   # re-pin new body to world origin

    def on_connect(self):
        # Re-pin the avatar start to world origin for each fresh viewer connection.
        self.avatar_center0 = None

    def produce_frame(self, frame_idx: int):
        # Apply any pending identity switch here, between frames — safe w.r.t. the
        # streaming loop which only calls produce_frame serially.
        if self.pending_id is not None:
            self.switch_identity(self.pending_id)
            self.pending_id = None

        cam = self.smpl_cams[frame_idx % self.n_frames]

        with torch.no_grad():
            # ── Deformation ──────────────────────────────────────────────────
            # Hand the pose-conditioned non-rigid field (and hence the ColorMLP)
            # this identity's `rots`, which is what actually selects the rendered
            # appearance. bone_transforms is left alone, so the rigid deformer
            # still follows the driving motion and the body animates normally.
            # smpl_cams are reused every loop, so restore afterwards.
            saved = None
            if self._pose_override is not None:
                saved = {k: cam.data[k] for k in self._pose_override}
                cam.data.update(self._pose_override)
            try:
                cam_c, _ = self.converter.pose_correction(cam, self.iteration)
                deformed_pc, _ = self.converter.deformer(
                    self.gaussians, cam_c, self.iteration, compute_loss=False
                )
            finally:
                if saved is not None:
                    cam.data.update(saved)
            torch.cuda.synchronize()

            # ── View-independent feature extraction (ColorMLP seam) ──────────
            feat    = self.colormlp.features(deformed_pc)
            R_bwd_f = self.colormlp.R_bwd(deformed_pc)

            # ── Axis-fix correction for the SH view direction ─────────────────
            # The traced Color-MLP computes  dir = R_bwd @ (xyz - cam_center)  in C++.
            # But the xyz we send is axis-fixed (x_vr = A @ x_orig - center0) while
            # R_bwd is in the ORIGINAL frame, so those two disagree — render.py has no
            # axis fix at all, which is why only the VR path is affected. A is a
            # rotation, so A^T maps VR -> original; folding it into R_bwd makes the
            # wrapper's own maths evaluate in the original frame:
            #     (R_bwd @ A^T) @ dir_vr == R_bwd @ dir_orig
            # (the center0 offset cancels in the subtraction).
            if self.colormlp.cano:
                R = R_bwd_f.view(-1, 3, 3)
                R_bwd_f = torch.matmul(R, self.axis_fix.t()).reshape(-1, 9).contiguous()

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

    # ── Driving motion: one subject's sequence; appearance is switched live ───
    # NOTE: do NOT set appearance_identity to pick the avatar — it filters the
    # multi-person dataset down to that subject, so the other identities' poses
    # (which is what selects appearance) are never loaded. The full list must
    # load. Use drive_identity for the motion source and start_identity for the
    # initial avatar instead.
    drive_idx = int(config.get("drive_identity", 0))
    smpl_cams = collect_smpl_frames(scene.test_dataset, drive_idx)
    identity_poses = collect_identity_poses(scene.test_dataset)

    if config.get("start_identity", None) is not None:
        identity_id = int(config.start_identity)
    elif appearance_id is not None:
        identity_id = int(appearance_id)
    else:
        identity_id = int(getattr(smpl_cams[0], "person_id", 0))
    log.info("Decoding canonical Gaussians for identity %d ...", identity_id)
    with torch.no_grad():
        scene.update_gaussians_from_migs(identity_id)
    log.info("Canonical avatar ready: %d Gaussians.", scene.gaussians.get_xyz.shape[0])

    # ── Color-MLP module (feature dim + export live in vr_viewer) ─────────────
    color_mlp = scene.converter.texture
    device    = next(color_mlp.parameters()).device

    colormlp = ColorMLPModule(color_mlp)
    log.info("Feature dim K=%d. Tracing Color MLP...", colormlp.K)
    model_bytes = colormlp.export(device)
    log.info("TorchScript model: %d bytes.", len(model_bytes))

    N_max = int(config.get("gaussians_vr", {}).get("n_max", _N_MAX))
    port  = int(config.get("gaussians_vr", {}).get("port", _PORT))

    # ── Hand off to the reusable, pipeline-agnostic streaming loop ────────────
    source = MistaVRSource(scene, smpl_cams, iteration, colormlp, model_bytes, N_max,
                           identity_id=identity_id,
                           identity_poses=identity_poses)
    run_server(source, port=port)


if __name__ == "__main__":
    main()
