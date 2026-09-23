"""
vr_viewer: CUDA-IPC transport for the SIBR remote viewer.

A `VRSource` provides the six per-Gaussian attributes for each frame.
`run_server(source, port)` handles streaming to the C++ viewer.
"""

from .server import VRSource, run_server
from .protocol import HANDSHAKE, DEFAULT_PORT
from .ipc import GaussianAttrBuffer, GaussianIPCManager
from .colormlp_export import ColorMLPModule

__all__ = [
    "VRSource", "run_server",
    "HANDSHAKE", "DEFAULT_PORT",
    "GaussianAttrBuffer", "GaussianIPCManager",
    "ColorMLPModule",
]
