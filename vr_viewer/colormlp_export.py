"""
vr_viewer.colormlp_export: helper for pipelines whose texture is a
ColorMLP with SH (3DGS-v2 / MISTA family).

This module has no pipeline-specific imports. It only reads these attributes
from the `color_mlp` object:
    .mlp
    .color_activation
    .sh_degree
    .cano_view_dir
    .cfg.feature_dim
    .use_xyz
    .use_cov
    .use_normal
    .non_rigid_dim

It serializes the ColorMLP into the TorchScript blob sent during the handshake
and determines the size K of the view-independent feature vector.
"""

import io

import torch
import torch.nn as nn
import torch.nn.functional as F

# Public surface: the module object, plus the raw export for callers that
# already hold a K and just want the TorchScript blob. Everything else
# (SH basis, the traced wrapper, dim/feature helpers) is a private detail.
__all__ = ["ColorMLPModule", "export_color_mlp"]

def _sh_bases(deg: int, dirs: torch.Tensor) -> torch.Tensor:
    """Evaluate real spherical-harmonic basis functions up to degree `deg` at `dirs`.

    @param deg: SH degree, 0-3.
    @param dirs: torch.Tensor of shape [N, 3], unit view directions.
    @return: torch.Tensor of shape [N, (deg+1)^2], SH basis values (column 0 is
        the constant term).
    """
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    cols = [torch.full((dirs.shape[0],), C0, dtype=dirs.dtype, device=dirs.device)]
    if deg >= 1:
        x = dirs[:, 0]; y = dirs[:, 1]; z = dirs[:, 2]
        cols.extend([-C1 * y, C1 * z, -C1 * x])
        if deg >= 2:
            C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                  -1.0925484305920792, 0.5462742152960396]
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            cols.extend([C2[0]*xy, C2[1]*yz, C2[2]*(2*zz-xx-yy), C2[3]*xz, C2[4]*(xx-yy)])
            if deg >= 3:
                C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                       0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                      -0.5900435899266435]
                cols.extend([
                    C3[0]*y*(3*xx-yy), C3[1]*xy*z, C3[2]*y*(4*zz-xx-yy),
                    C3[3]*z*(2*zz-3*xx-3*yy), C3[4]*x*(4*zz-xx-yy),
                    C3[5]*z*(xx-yy), C3[6]*x*(xx-3*yy),
                ])
    return torch.stack(cols, dim=1)

class _ColorMLPV4Wrapper(nn.Module):
    """TorchScript-traceable wrapper injecting the SH view-direction embedding
    into a ColorMLP's feature vector before its MLP forward.

    @note: Mirrors `ColorMLP.compose_input`'s feature ordering: base_features |
        xyz_norm? | cov? | normal? | dir_embed(SH) | non_rigid?.
    """

    def __init__(self, color_mlp, pre_dim: int):
        """@brief Capture the MLP, activation, and feature-layout info needed to trace forward.
        @param color_mlp: source ColorMLP-with-SH texture module.
        @param pre_dim: column count before the SH dir_embed insertion point
            (see `_pre_sh_feat_dim`).
        """
        super().__init__()
        self._mlp        = color_mlp.mlp
        self._activation = color_mlp.color_activation
        self._sh_degree  = int(color_mlp.sh_degree)
        self._cano       = bool(getattr(color_mlp, 'cano_view_dir', False))
        self._pre_dim    = int(pre_dim)

    def forward(self, feat_no_view, xyz, cam_center, R_bwd):
        """Insert the SH view-direction embedding and run the ColorMLP's MLP.

        @param feat_no_view: [N, K] view-independent features, ordered per the
            class docstring, with the SH slot not yet inserted.
        @param xyz: [N, 3] Gaussian centers.
        @param cam_center: [3] camera center.
        @param R_bwd: [N, 3, 3] per-Gaussian canonical->posed inverse rotation,
            applied to the view direction when `cano_view_dir` is set.
        @return: torch.Tensor of shape [N, C], the MLP's activated color output.
        """
        if self._sh_degree > 0:
            dir_pp = xyz - cam_center.unsqueeze(0)
            if self._cano:
                dir_pp = torch.bmm(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            dir_pp_norm = F.normalize(dir_pp, dim=-1)
            basis       = _sh_bases(self._sh_degree, dir_pp_norm)
            dir_embed   = basis[:, 1:]
            feat_pre    = feat_no_view[:, :self._pre_dim]
            feat_post   = feat_no_view[:, self._pre_dim:]
            features    = torch.cat([feat_pre, dir_embed, feat_post], dim=1)
        else:
            features = feat_no_view
        return self._activation(self._mlp(features))


def _pre_sh_feat_dim(color_mlp) -> int:
    """@brief Return the column count before the SH dir_embed insertion point (base + xyz? + cov? + normal?).
    @param color_mlp: ColorMLP-with-SH texture module.
    @return: int, feature width before the SH slot.
    """
    d = int(color_mlp.cfg.feature_dim)
    if getattr(color_mlp, 'use_xyz',    False): d += 3
    if getattr(color_mlp, 'use_cov',    False): d += 6
    if getattr(color_mlp, 'use_normal', False): d += 3
    return d


def _view_indep_feat_dim(color_mlp) -> int:
    """Compute the full view-independent feature width K.

    @param color_mlp: ColorMLP-with-SH texture module.
    @return: int K = base + xyz? + cov? + normal? + non_rigid?.
    @throws TypeError: if `color_mlp` lacks `cfg.feature_dim` (not a
        ColorMLP-style texture).
    """
    if not hasattr(color_mlp, 'cfg') or not hasattr(color_mlp.cfg, 'feature_dim'):
        raise TypeError(
            "colormlp_export requires a ColorMLP-style texture "
            "(needs cfg.feature_dim, .mlp, .sh_degree, .color_activation)."
        )
    K = int(color_mlp.cfg.feature_dim)
    if getattr(color_mlp, 'use_xyz',    False): K += 3
    if getattr(color_mlp, 'use_cov',    False): K += 6
    if getattr(color_mlp, 'use_normal', False): K += 3
    if getattr(color_mlp, 'non_rigid_dim', 0) > 0: K += color_mlp.non_rigid_dim
    return K


def export_color_mlp(color_mlp, K: int, device: torch.device) -> bytes:
    """Trace the ColorMLP (with SH dir-embed injection) to TorchScript.

    @param color_mlp: ColorMLP-with-SH texture module.
    @param K: view-independent feature width (from `_view_indep_feat_dim`),
        used to size the trace example inputs.
    @param device: torch device to trace on.
    @return: bytes, a serialized TorchScript module.
    """
    pre_dim = _pre_sh_feat_dim(color_mlp)
    wrapper = _ColorMLPV4Wrapper(color_mlp, pre_dim).to(device).eval()
    N_ex   = 64
    opts   = dict(dtype=torch.float32, device=device)
    f_ex   = torch.zeros(N_ex, K, **opts)
    xyz_ex = torch.zeros(N_ex, 3, **opts)
    cam_ex = torch.zeros(3,       **opts)
    r_ex   = torch.eye(3, **opts).unsqueeze(0).expand(N_ex, -1, -1).contiguous()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (f_ex, xyz_ex, cam_ex, r_ex),
                                 strict=False, check_trace=False)
    buf = io.BytesIO()
    torch.jit.save(traced, buf)
    return buf.getvalue()

def _build_rotation(r: torch.Tensor) -> torch.Tensor:
    """@brief Convert quaternions to rotation matrices.
    @param r: torch.Tensor of shape [N, 4], (w, x, y, z) quaternions (not
        required to be pre-normalized).
    @return: torch.Tensor of shape [N, 3, 3], rotation matrices.
    """
    norm = torch.sqrt((r * r).sum(dim=1))
    q = r / norm[:, None]
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = r.new_zeros((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - w*z)
    R[:, 0, 2] = 2 * (x*z + w*y)
    R[:, 1, 0] = 2 * (x*y + w*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - w*x)
    R[:, 2, 0] = 2 * (x*z - w*y)
    R[:, 2, 1] = 2 * (y*z + w*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def _extract_view_indep_features(color_mlp, gaussians) -> torch.Tensor:
    """Build the view-independent feature tensor for a duck-typed Gaussian point cloud.

    @param color_mlp: texture module; `use_xyz`/`use_cov`/`use_normal`/
        `non_rigid_dim` select which optional features are appended.
    @param gaussians: Gaussian container exposing `get_features`, `get_xyz`,
        `get_covariance()`, `_scaling`, `_rotation`, and (if non_rigid_dim > 0)
        `non_rigid_feature`.
    @return: torch.Tensor of shape [N, K], ordered base_features | xyz_norm? |
        cov? | normal? | non_rigid? (the SH dir_embed is injected later, inside
        the traced `_ColorMLPV4Wrapper`, between normal? and non_rigid?).
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
        rot    = _build_rotation(gaussians._rotation)
        normal = torch.gather(rot, dim=2,
            index=scale.argmin(1).reshape(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1)
        features = torch.cat([features, normal], dim=1)
    if getattr(color_mlp, 'non_rigid_dim', 0) > 0:
        features = torch.cat([features, gaussians.non_rigid_feature], dim=1)
    return features   # [N, K]


def _extract_R_bwd(gaussians, cano_view_dir: bool) -> torch.Tensor:
    """Compute the per-Gaussian rotation applied to the SH view direction.

    @param gaussians: Gaussian container; if `cano_view_dir` is set and it has
        a `fwd_transform` attribute, the rotation is taken from its
        upper-left 3x3 block (transposed).
    @param cano_view_dir: if False, or if `gaussians` has no `fwd_transform`,
        the rotation is identity.
    @return: torch.Tensor of shape [N, 9], flattened 3x3 world->canonical
        rotation per Gaussian.
    """
    if cano_view_dir and hasattr(gaussians, 'fwd_transform'):
        T_fwd = gaussians.fwd_transform
        R_bwd = T_fwd[:, :3, :3].transpose(1, 2)
        return R_bwd.reshape(-1, 9).contiguous()
    N   = gaussians.get_xyz.shape[0]
    dev = gaussians.get_xyz.device
    return torch.eye(3, dtype=torch.float32, device=dev)\
               .unsqueeze(0).expand(N, -1, -1).reshape(N, 9).contiguous()

class ColorMLPModule:
    """Bundles the full ColorMLP seam for a ColorMLP-with-SH texture: feature
    width K, TorchScript export, and per-frame view-independent feature/R_bwd
    extraction."""

    def __init__(self, color_mlp):
        """@brief Wrap a ColorMLP-with-SH texture and compute its feature width `K`.
        @param color_mlp: source ColorMLP-with-SH texture module.
        """
        self.color_mlp = color_mlp
        self.cano      = bool(getattr(color_mlp, 'cano_view_dir', False))
        self.K         = _view_indep_feat_dim(color_mlp)

    def export(self, device: torch.device) -> bytes:
        """@brief Trace the wrapped ColorMLP to TorchScript on `device` (see `export_color_mlp`)."""
        return export_color_mlp(self.color_mlp, self.K, device)

    def features(self, gaussians) -> torch.Tensor:
        """@brief Return the [N, K] view-independent features for `gaussians` (see `_extract_view_indep_features`)."""
        return _extract_view_indep_features(self.color_mlp, gaussians)

    def R_bwd(self, gaussians) -> torch.Tensor:
        """@brief Return the [N, 9] per-Gaussian view-direction rotation for `gaussians` (see `_extract_R_bwd`)."""
        return _extract_R_bwd(gaussians, self.cano)
