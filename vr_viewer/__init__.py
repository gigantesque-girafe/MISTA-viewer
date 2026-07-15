"""
vr_viewer  —  Reusable CUDA-IPC VR streaming transport for the SIBR remote
OpenXR viewer.

Pipeline-agnostic. To wire the viewer onto a new pipeline, implement a
`VRSource` subclass that produces the six per-Gaussian attribute tensors for a
frame, then call `run_server(source, port)`. The pipeline's own render.py /
training code is never touched.

    from vr_viewer import VRSource, run_server

    class MySource(VRSource):
        def __init__(self, ...):
            self.device, self.N_max, self.K = ...
            self.model_bytes, self.n_frames = ...
        def produce_frame(self, i):
            return xyz, feat, R_bwd, opacity, scale, rot

    run_server(MySource(...), port=6012)
"""

from .server import VRSource, run_server
from .protocol import HANDSHAKE, DEFAULT_PORT
from .ipc import GaussianAttrBuffer, GaussianIPCManager

__all__ = [
    "VRSource", "run_server",
    "HANDSHAKE", "DEFAULT_PORT",
    "GaussianAttrBuffer", "GaussianIPCManager",
]
