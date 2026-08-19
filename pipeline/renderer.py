"""MistaRenderer: deformer camera -> 5 per-Gaussian attribute tensors for C++ IPC."""

import torch

# The MISTA VR adapter: we compose its source and reuse its feature helpers verbatim.
from render_vr_v1_modular import (
    MistaVRSource,
    extract_view_indep_features,
    extract_R_bwd,
    cov3D_in_vr_frame,
)


# Sentinel deform/texture iteration: past every training/delay/export threshold so no
# iteration-keyed disk writes fire (the VR deform path already avoids render()'s export
# gate, but we stay consistent with Phase 1).
RENDER_ITER = 10 ** 9


class MistaRenderer:
    """Deform + pack: camera -> 5 per-Gaussian attribute tensors for the C++ IPC.

    Composes (rather than subclasses) `MistaVRSource` so it reuses that adapter's
    deform/feature/axis-fix machinery verbatim while staying a plain collaborator.
    Also owns the C++ handshake attributes (device/N_max/K/model_bytes), the
    per-connection center reset, and live identity switching.
    """

    def __init__(self, scene, template_cam, K, model_bytes, N_max):
        # n_frames=1 placeholder: frame_idx is ignored (poses are live-sourced).
        self._vr = MistaVRSource(scene, [template_cam], RENDER_ITER, K, model_bytes, N_max)
        self.scene = scene
        # Handshake attributes passed straight through to run_server via VRSource.
        self.device = self._vr.device
        self.N_max = self._vr.N_max
        self.K = self._vr.K
        self.model_bytes = self._vr.model_bytes

    def on_connect(self):
        # Re-capture the avatar center on the first frame of each connection.
        self._vr.on_connect()

    def switch_identity(self, identity):
        with torch.no_grad():
            self.scene.update_gaussians_from_migs(int(identity))
        self._vr.gaussians = self.scene.gaussians

    def render(self, cam):
        """Deform the canonical Gaussians by `cam` -> (xyz, feat, R_bwd, opacity, cov3D)."""
        vr = self._vr
        with torch.no_grad():
            cam_c, _ = vr.converter.pose_correction(cam, vr.iteration)
            deformed_pc, _ = vr.converter.deformer(
                vr.gaussians, cam_c, vr.iteration, compute_loss=False
            )
            torch.cuda.synchronize()

            feat = extract_view_indep_features(vr.color_mlp, deformed_pc, cam_c)
            R_bwd_f = extract_R_bwd(deformed_pc, vr.cano, vr.axis_fix)

            xyz = deformed_pc.get_xyz @ vr.axis_fix.T
            if vr.avatar_center0 is None:
                vr.avatar_center0 = xyz.mean(dim=0, keepdim=True).clone()
            xyz = xyz - vr.avatar_center0
            opacity = deformed_pc.get_opacity.squeeze(-1)
            cov6 = deformed_pc.get_covariance()
            cov3D = cov3D_in_vr_frame(cov6, vr.axis_fix)
        return xyz, feat, R_bwd_f, opacity, cov3D
