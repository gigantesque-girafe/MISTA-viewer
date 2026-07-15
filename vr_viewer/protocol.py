"""
vr_viewer.protocol  —  Pipeline-agnostic wire protocol for the SIBR remote
OpenXR application v4.2 (SIBR_remoteGaussianOpenXRv4_2_app.exe).

Both the startup handshake (port 6012, magic b"V42E") and the per-frame packet
are defined here so every adapter speaks exactly the same bytes. Wire v2 ("V42E")
carries cov3D[6] in the buffer tail instead of the old scale[3]+rot[4].
"""

import socket
import struct
import logging

log = logging.getLogger("vr_viewer.protocol")

HANDSHAKE = b"V42E"    # matches SIBR_remoteGaussianOpenXRv4_2_app
                       # v2 wire: buffer tail is cov3D[6] instead of scale[3]+rot[4]
                       # (must match the magic sent by the C++ viewer's handshake).
DEFAULT_PORT = 6012


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("viewer disconnected")
        buf.extend(chunk)
    return bytes(buf)


def do_handshake(conn, ipc_mgrs, attr_bufs, K, model_bytes):
    """Startup handshake (identical wire format to render_vr_v4_2 / render_vr_v1)."""
    magic = recv_exact(conn, 4)
    if magic != HANDSHAKE:
        raise ValueError(f"Unexpected handshake magic: {magic!r} (expected {HANDSHAKE!r})")

    log.info("Handshake OK — sending N_max=%d  K=%d  model=%d bytes (double-buffered).",
             attr_bufs[0].N_max, K, len(model_bytes))

    conn.sendall(struct.pack("<III", attr_bufs[0].N_max, K, len(model_bytes)))
    conn.sendall(model_bytes)
    conn.sendall(struct.pack("<i", attr_bufs[0].device_idx))  # int32

    for buf, mgr in zip(attr_bufs, ipc_mgrs):
        conn.sendall(struct.pack("<Q", buf.ipc_offset))            # uint64
        conn.sendall(bytes(mgr.mem_handle.raw))                    # 64 bytes
        conn.sendall(bytes(mgr.data_ready_evt_handle.raw))         # 64 bytes
        conn.sendall(bytes(mgr.read_complete_evt_handle.raw))      # 64 bytes

    log.info("IPC handles sent for both buffers. device=%d", attr_bufs[0].device_idx)
