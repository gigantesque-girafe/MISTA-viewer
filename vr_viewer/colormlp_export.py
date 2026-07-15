"""
vr_viewer.colormlp_export  —  OPTIONAL helper for pipelines whose texture is a
ColorMLP-with-SH (the 3dgs-v2 / MISTA family).

This module is *duck-typed and import-clean*: it never imports anything from a
specific pipeline (no `models`, `scene`, `utils`, `hydra`). It only reads
attributes off the `color_mlp` object you pass in:
    .mlp, .color_activation, .sh_degree, .cano_view_dir,
    .cfg.feature_dim, .use_xyz, .use_cov, .use_normal, .non_rigid_dim

so it is safe to live in the reusable package. It knows how to *serialize* a
ColorMLP into the TorchScript blob sent at handshake and how to size the
view-independent feature vector K. It does NOT know how to read your Gaussians —
extracting per-frame features from your own point cloud stays in your adapter,
because that touches your pipeline's data model.

If a pipeline's texture is not a ColorMLP-with-SH, don't use this module; write
your own export in the adapter instead.
"""

import io

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# SH basis (trace-friendly)
# ─────────────────────────────────────────────────────────────────────────────

def sh_bases(deg: int, dirs: torch.Tensor) -> torch.Tensor:
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


# ─────────────────────────────────────────────────────────────────────────────
# TorchScript wrapper — mirrors ColorMLP.compose_input ordering.
#   base_features | xyz_norm? | cov? | normal? | dir_embed(SH) | non_rigid?
# The SH dir_embed is inserted after the geometric features and BEFORE the
# non-rigid block, so pre_dim = feature_dim + xyz? + cov? + normal?.
# ─────────────────────────────────────────────────────────────────────────────

class ColorMLPV4Wrapper(nn.Module):
    def __init__(self, color_mlp, pre_dim: int):
        super().__init__()
        self._mlp        = color_mlp.mlp
        self._activation = color_mlp.color_activation
        self._sh_degree  = int(color_mlp.sh_degree)
        self._cano       = bool(getattr(color_mlp, 'cano_view_dir', False))
        self._pre_dim    = int(pre_dim)

    def forward(self, feat_no_view, xyz, cam_center, R_bwd):
        if self._sh_degree > 0:
            dir_pp = xyz - cam_center.unsqueeze(0)
            if self._cano:
                dir_pp = torch.bmm(R_bwd, dir_pp.unsqueeze(-1)).squeeze(-1)
            dir_pp_norm = F.normalize(dir_pp, dim=-1)
            basis       = sh_bases(self._sh_degree, dir_pp_norm)
            dir_embed   = basis[:, 1:]
            feat_pre    = feat_no_view[:, :self._pre_dim]
            feat_post   = feat_no_view[:, self._pre_dim:]
            features    = torch.cat([feat_pre, dir_embed, feat_post], dim=1)
        else:
            features = feat_no_view
        return self._activation(self._mlp(features))


def pre_sh_feat_dim(color_mlp) -> int:
    """Column count before the SH dir_embed insertion point (base + xyz? + cov? + normal?)."""
    d = int(color_mlp.cfg.feature_dim)
    if getattr(color_mlp, 'use_xyz',    False): d += 3
    if getattr(color_mlp, 'use_cov',    False): d += 6
    if getattr(color_mlp, 'use_normal', False): d += 3
    return d


def view_indep_feat_dim(color_mlp) -> int:
    """Full view-independent feature width K (base + xyz? + cov? + normal? + non_rigid?)."""
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
    """Trace the ColorMLP (with SH dir-embed injection) and return a TorchScript blob."""
    pre_dim = pre_sh_feat_dim(color_mlp)
    wrapper = ColorMLPV4Wrapper(color_mlp, pre_dim).to(device).eval()
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
