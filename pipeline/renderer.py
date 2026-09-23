"""MistaRenderer: deformer camera -> 5 per-Gaussian attribute tensors for C++ IPC."""

import torch

from pipeline.mista_features import AXIS_FIX, deform_and_pack


# Sentinel deform/texture iteration: past every training/delay/export threshold so no
# iteration-keyed disk writes fire (the VR deform path already avoids render()'s export
# gate, but we stay consistent with Phase 1).
RENDER_ITER = 10 ** 9


class MistaRenderer:
    """Deform + pack: camera -> 5 per-Gaussian attribute tensors for the C++ IPC.

    Derives its MISTA scene state (converter/gaussians/color_mlp/axis_fix) the
    same way `render_vr.MistaVRSource` does, and shares its deform/pack logic via
    `pipeline.mista_features.deform_and_pack` — both are plain consumers of that
    module, with no dependency on each other.
    Also owns the C++ handshake attributes (device/N_max/K/model_bytes), the
    per-connection center reset, and live identity switching.
    """

    def __init__(self, scene, template_cam, K, model_bytes, N_max):
        """Derive MISTA scene state from the decoded scene and store the C++ handshake attributes.

        @param scene: MISTA scene object, already decoded (`scene.gaussians`
            holds the canonical avatar).
        @param template_cam: base camera reused for every posed frame (unused
            directly; kept for interface parity with the VR entry point).
        @param K: view-independent feature dimension, forwarded to the
            VRSource handshake (`self.K`).
        @param model_bytes: serialized Color-MLP TorchScript model, forwarded
            to the VRSource handshake (`self.model_bytes`).
        @param N_max: IPC buffer capacity, forwarded to the VRSource handshake
            (`self.N_max`).
        """
        self.scene       = scene
        self.converter   = scene.converter
        self.gaussians   = scene.gaussians          # canonical avatar (decoded from MIGS)
        self.color_mlp   = self.converter.texture
        self.cano        = bool(getattr(self.color_mlp, 'cano_view_dir', False))

        self.K           = K
        self.model_bytes = model_bytes
        self.N_max       = N_max
        self.device      = next(self.color_mlp.parameters()).device

        self.axis_fix       = torch.tensor(AXIS_FIX, dtype=torch.float32, device=self.device)
        self.avatar_center0 = None                  # captured on frame 0 of each connection

    def on_connect(self):
        """@brief Re-pin the avatar center on the next frame."""
        self.avatar_center0 = None

    def switch_identity(self, identity):
        """Decode a different identity's canonical Gaussians and use them for subsequent renders.

        @param identity: identity index to decode from the MIGS factorization.
        """
        with torch.no_grad():
            self.scene.update_gaussians_from_migs(int(identity))
        self.gaussians = self.scene.gaussians

    def render(self, cam):
        """Deform the canonical Gaussians by a posed camera and pack the per-Gaussian attributes.

        @param cam: posed camera (e.g. from `MistaPoseAdapter.to_camera`/`canonical`).
        @return: tuple `(xyz, feat, R_bwd, opacity, cov3D)` — the five
            per-Gaussian attribute tensors sent to the C++/SIBR viewer.
        @note: On the first call after `on_connect()`, captures the current
            frame's mean xyz as the avatar origin and subtracts it from all
            subsequent frames.
        """
        xyz, feat, R_bwd_f, opacity, cov3D, self.avatar_center0, _ = deform_and_pack(
            self.converter, self.gaussians, self.color_mlp, self.cano, self.axis_fix,
            cam, RENDER_ITER, self.avatar_center0,
        )
        return xyz, feat, R_bwd_f, opacity, cov3D
