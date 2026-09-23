"""MISTA-specific deform/feature-extraction helpers, shared by the VR (dataset-driven)
and webcam (live-driven) render entry points. 
"""

import torch

from utils.general_utils import build_rotation

# Axis-convention correction to align avatar in VR world. 
# If the avatar renders upside down,
# negate row 1 (height/Y) entries.
AXIS_FIX = [[-0.9601507782936096,  0.15131592750549316, -0.2349766492843628],
            [-0.27107205986976624, -0.29949113726615906,  0.914781391620636],
            [ 0.06804757565259933,  0.9420236349105835,   0.32857415080070496]]


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction — mirrors MISTA ColorMLP.compose_input (view-independent part).
# ─────────────────────────────────────────────────────────────────────────────

def extract_view_indep_features(color_mlp, gaussians, camera) -> torch.Tensor:
    """Build the view-independent feature tensor consumed by the Color MLP.

    Mirrors MISTA's `ColorMLP.compose_input`, concatenating only the
    view-independent parts (SH base features plus any of xyz/covariance/normal/
    non-rigid features enabled on `color_mlp`).

    @param color_mlp: texture module; its `use_xyz`, `use_cov`, `use_normal` and
        `non_rigid_dim` attributes select which optional features are appended.
    @param gaussians: Gaussian container exposing `get_features`, `get_xyz`,
        `get_covariance()`, `_scaling`, `_rotation`, and (if non_rigid_dim > 0)
        `non_rigid_feature`.
    @param camera: unused; kept for interface parity with the caller.
    @return: torch.Tensor of shape [N, K], K depending on which optional
        features are enabled on `color_mlp`.
    """
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


def extract_R_bwd(gaussians, cano_view_dir: bool, axis_fix: torch.Tensor) -> torch.Tensor:
    """Compute the per-Gaussian rotation matrix the client applies to the SH view direction.

    @param gaussians: Gaussian container; if `cano_view_dir` is set and it has a
        `fwd_transform` attribute, the world->canonical rotation is taken from its
        upper-left 3x3 block (transposed).
    @param cano_view_dir: if False, or if `gaussians` has no `fwd_transform`, the
        rotation is identity (world-space view direction used as-is).
    @param axis_fix: [3, 3] world-to-VR-frame rotation matrix A.
    @return: torch.Tensor of shape [N, 9] — each row is the flattened 3x3 matrix
        `R_bwd @ Aᵀ` (canonical rotation composed with the inverse axis fix), N =
        `gaussians.get_xyz.shape[0]`.
    @note: The client reconstructs `dir_pp = xyz - cam_center` from VR-frame xyz
        (p_vr = A p_world), but the texture was trained with dir_pp in world
        space (models/texture/texture.py:107). Shipping `R_bwd` alone would
        evaluate the SH basis at a direction rotated by A itself, shifting hue on
        view-dependent surfaces; `Aᵀ` must be composed in to undo the axis fix
        before the canonical rotation is applied.
    """
    N   = gaussians.get_xyz.shape[0]
    dev = gaussians.get_xyz.device
    if cano_view_dir and hasattr(gaussians, 'fwd_transform'):
        T_fwd = gaussians.fwd_transform
        R_bwd = T_fwd[:, :3, :3].transpose(1, 2)          # world -> canonical
    else:
        R_bwd = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0).expand(N, -1, -1)
    R_bwd = torch.matmul(R_bwd, axis_fix.t())             # VR -> world -> canonical
    return R_bwd.reshape(N, 9).contiguous()


def cov3D_in_vr_frame(cov6: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Rotate packed per-Gaussian covariances by the axis-fix matrix (Σ -> A Σ Aᵀ).

    @param cov6: torch.Tensor of shape [N, 6], strip_symmetric order
        [00,01,02,11,12,22] (as returned by `gaussians.get_covariance()` via the
        rotation_precomp path).
    @param A: [3, 3] world-to-VR-frame rotation matrix.
    @return: torch.Tensor of shape [N, 6], same strip_symmetric order, rotated
        into the VR frame.
    @note: The full 3x3 matrix is rebuilt and rotated rather than working from a
        quaternion, to preserve the non-orthonormal scale/shear that LBS
        blending puts into rotation_precomp.
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


def deform_and_pack(converter, gaussians, color_mlp, cano, axis_fix, cam, iteration, avatar_center0):
    """Deform the canonical Gaussians by a posed camera and pack the per-Gaussian attributes.

    Shared by the dataset-driven VR source (`render_vr.MistaVRSource.produce_frame`)
    and the live webcam renderer (`pipeline.renderer.MistaRenderer.render`) — both
    just supply their own `converter`/`gaussians`/`color_mlp`/`cano`/`axis_fix` state.

    @param avatar_center0: cached avatar origin from a previous call, or None to
        capture it fresh from this frame's mean xyz.
    @return: tuple `(xyz, feat, R_bwd, opacity, cov3D, avatar_center0, deformed_pc)`
        — the five per-Gaussian attribute tensors sent to the C++/SIBR viewer,
        the (possibly newly captured) avatar origin for the caller to cache, and
        the deformed Gaussian container itself (for callers that want extra
        diagnostics, e.g. `deformed_pc.get_scaling`).
    """
    with torch.no_grad():
        cam_c, _ = converter.pose_correction(cam, iteration)
        deformed_pc, _ = converter.deformer(gaussians, cam_c, iteration, compute_loss=False)
        torch.cuda.synchronize()

        feat    = extract_view_indep_features(color_mlp, deformed_pc, cam_c)
        R_bwd_f = extract_R_bwd(deformed_pc, cano, axis_fix)

        xyz = deformed_pc.get_xyz @ axis_fix.T
        if avatar_center0 is None:
            avatar_center0 = xyz.mean(dim=0, keepdim=True).clone()
        xyz     = xyz - avatar_center0
        opacity = deformed_pc.get_opacity.squeeze(-1)

        cov6  = deformed_pc.get_covariance()
        cov3D = cov3D_in_vr_frame(cov6, axis_fix)

    return xyz, feat, R_bwd_f, opacity, cov3D, avatar_center0, deformed_pc
