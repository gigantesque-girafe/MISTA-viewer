"""
vr_viewer.server  —  Pipeline-agnostic double-buffered streaming loop.

This is the reusable heart of the VR viewer. It owns:
  * IPC buffer + handle allocation (double-buffered),
  * the TCP accept / reconnect loop,
  * the startup handshake,
  * per-frame double-buffering, event signalling, frame packet and pacing.

Everything that is *pipeline-specific* is hidden behind the `VRSource`
interface below. To bring the viewer to a new pipeline you implement ONE
subclass of `VRSource` (see render_vr_v1_modular.py for the MISTA example) and
call `run_server(source, port)`. render.py / your training code stays intact.
"""

import os
import socket
import struct
import time
import logging

import torch

from .ipc import GaussianAttrBuffer, GaussianIPCManager
from .protocol import do_handshake, DEFAULT_PORT

log = logging.getLogger("vr_viewer.server")

_LOG_INTERVAL = 60


class VRSource:
    """
    Contract between a pipeline and the generic streaming loop.

    A subclass must set these attributes (typically in __init__ / setup):
        device       : torch.device   — CUDA device the attribute tensors live on
        N_max        : int            — IPC buffer capacity (>= max Gaussian count)
        K            : int            — per-Gaussian view-independent feature dim
        model_bytes  : bytes          — TorchScript Color-MLP blob sent at handshake
        n_frames     : int            — number of animation frames to cycle over

    and implement:
        produce_frame(frame_idx) -> (xyz, feat, R_bwd, opacity, cov3D)
            All five are CUDA tensors ready to be copied into the IPC buffer.
            cov3D is [N, 6] (posed covariance upper triangle) — the cov3D_precomp
            path, matching render.py (compute_cov3D_python=True). `frame_idx` is
            the raw monotonically-increasing counter; the source is responsible
            for wrapping it (e.g. frame_idx % n_frames).
    """

    device: torch.device
    N_max: int
    K: int
    model_bytes: bytes
    n_frames: int

    def produce_frame(self, frame_idx: int):
        raise NotImplementedError

    # Optional hook called once when a viewer connects, before the frame loop.
    def on_connect(self):
        pass


def run_server(source: VRSource, port: int = DEFAULT_PORT, target_fps: int = 30):
    """Allocate IPC resources for `source` and serve the streaming loop forever."""
    target_frame_seconds = 1.0 / target_fps

    # Diagnostics / fix knobs (env):
    #   VR_PRODUCE_HZ   : hard-cap the producer to this rate. The default cap does
    #                     nothing when produce > 33 ms; this forces a real sleep so
    #                     the GPU is freed for the render process. This is the fix
    #                     for deform<->render GPU contention.
    #   VR_FREEZE_AFTER : after this many frames, stop deforming and just
    #                     re-announce the last buffer. The TCP connection stays up
    #                     and C++ keeps rendering, but Python does ZERO GPU work.
    #                     If C++'s mlp/raster times then drop, the slowdown was GPU
    #                     contention with the deformer (not thermal).
    produce_hz = float(os.environ.get("VR_PRODUCE_HZ", "0"))
    if produce_hz > 0:
        target_frame_seconds = 1.0 / produce_hz
        log.info("[RATE] producer hard-capped to %.1f Hz (%.1f ms/frame)",
                 produce_hz, target_frame_seconds * 1000.0)
    freeze_after = int(os.environ.get("VR_FREEZE_AFTER", "0"))
    if freeze_after:
        log.info("[FREEZE] will stop deforming after %d frames (connection stays up).",
                 freeze_after)

    log.info("Allocating double-buffered IPC buffers: N_max=%d  K=%d", source.N_max, source.K)
    attr_bufs = [GaussianAttrBuffer(source.N_max, source.K, source.device) for _ in range(2)]

    log.info("Creating CUDA IPC handles...")
    ipc_mgrs = [GaussianIPCManager(buf) for buf in attr_bufs]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(1)
    log.info("VR CUDA-IPC Gaussian server on 0.0.0.0:%d", port)
    log.info("Launch: SIBR_remoteGaussianOpenXRv4_2_app.exe --port %d", port)

    try:
        while True:
            conn, addr = srv.accept()
            log.info("C++ viewer connected from %s:%d", *addr)
            try:
                do_handshake(conn, ipc_mgrs, attr_bufs, source.K, source.model_bytes)
                source.on_connect()

                pipeline_frame_id = 0
                buf_idx = 0
                anim_frame = 0
                frozen = False
                last_buf, last_N = 0, source.N_max

                while True:
                    t0 = time.perf_counter()

                    # ── FREEZE diagnostic: keep the socket + C++ rendering alive,
                    # but do ZERO GPU work — just re-announce the last good buffer.
                    if freeze_after and anim_frame >= freeze_after:
                        if not frozen:
                            log.info("[FREEZE] producer stopped at frame %d — re-announcing "
                                     "buf %d (N=%d). Python GPU now idle; watch C++ mlp/raster.",
                                     anim_frame, last_buf, last_N)
                            frozen = True
                        conn.sendall(struct.pack("<QII", pipeline_frame_id, last_buf, last_N))
                        pipeline_frame_id += 1
                        time.sleep(target_frame_seconds)
                        continue

                    # Don't overwrite this buffer until C++ is done reading it.
                    tw = time.perf_counter()
                    ipc_mgrs[buf_idx].wait_read_complete()
                    wait_ms = (time.perf_counter() - tw) * 1000.0

                    # ── Pipeline-specific frame production ────────────────────
                    t1 = time.perf_counter()
                    xyz, feat, R_bwd, opacity, cov3D = source.produce_frame(anim_frame)
                    produce_ms = (time.perf_counter() - t1) * 1000.0
                    N = xyz.shape[0]

                    # ── Pack into IPC buffer (GPU-to-GPU, non-blocking) ───────
                    t3 = time.perf_counter()
                    attr_bufs[buf_idx].write(xyz, feat, R_bwd, opacity, cov3D)
                    pack_ms = (time.perf_counter() - t3) * 1000.0

                    # ── Sync + record data-ready event, announce frame ────────
                    t4 = time.perf_counter()
                    ipc_mgrs[buf_idx].record_data_ready()
                    conn.sendall(struct.pack("<QII", pipeline_frame_id, buf_idx, N))
                    signal_ms = (time.perf_counter() - t4) * 1000.0
                    last_buf, last_N = buf_idx, N   # newest valid buffer (for FREEZE)

                    total_ms = (time.perf_counter() - t0) * 1000.0

                    remaining = target_frame_seconds - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

                    anim_frame += 1
                    pipeline_frame_id += 1
                    buf_idx = 1 - buf_idx

                    if anim_frame % _LOG_INTERVAL == 0 or anim_frame <= 3:
                        log.info(
                            "frame %4d  N=%d  K=%d  total=%.1fms"
                            "  wait=%.1f  produce=%.1f  pack=%.1f  signal=%.1f",
                            anim_frame, N, source.K, total_ms,
                            wait_ms, produce_ms, pack_ms, signal_ms,
                        )

            except (ConnectionError, BrokenPipeError, OSError) as e:
                log.info("Disconnected (%s) — waiting for reconnect.", e)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        log.info("Server stopped.")
    finally:
        srv.close()
