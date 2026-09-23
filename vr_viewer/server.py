"""
vr_viewer.server: Double-buffered streaming loop.

It owns:
  * IPC buffer + handle allocation (double-buffered),
  * the TCP accept / reconnect loop,
  * the startup handshake,
  * per-frame double-buffering, event signalling, frame packet and pacing.

Everything that is *pipeline-specific* is hidden behind the `VRSource`
interface below. To bring the viewer to a new pipeline you implement ONE
subclass of `VRSource` (see render_vr.py for the MISTA example) and
call `run_server(source, port)`.
"""

import os
import socket
import struct
import time
import logging

import torch

from .ipc import GaussianAttrBuffer, GaussianIPCManager
from .protocol import (
    do_handshake, poll_control,
    CTRL_SET_IDENTITY, CTRL_SET_PAUSE, CTRL_STEP_FRAME,
    DEFAULT_PORT,
)

log = logging.getLogger("vr_viewer.server")

_LOG_INTERVAL = 60


class VRSource:
    """Contract between a pipeline and the generic streaming loop.

    A subclass must set these attributes (typically in `__init__`/setup):
        device       : torch.device    CUDA device the attribute tensors live on
        N_max        : int             IPC buffer capacity (>= max Gaussian count)
        K            : int             per-Gaussian view-independent feature dim
        model_bytes  : bytes           TorchScript Color-MLP blob sent at handshake
        n_frames     : int             number of animation frames to cycle over
    """

    device: torch.device
    N_max: int
    K: int
    model_bytes: bytes
    n_frames: int

    # Set by the server's control channel (poll_control); a subclass that supports
    # live identity switching consumes this at the top of produce_frame. Sources
    # that ignore it are unaffected.
    pending_id = None

    def produce_frame(self, frame_idx: int):
        """Produce one frame's per-Gaussian attribute tensors.

        @param frame_idx: raw monotonically-increasing frame counter; the
            subclass is responsible for wrapping it (e.g. `frame_idx % n_frames`).
        @return: tuple `(xyz, feat, R_bwd, opacity, cov3D)`, all CUDA tensors
            ready to be copied into the IPC buffer. `cov3D` is [N, 6] (posed
            covariance upper triangle) the cov3D_precomp path, matching
            render.py (compute_cov3D_python=True).
        """
        raise NotImplementedError

    def on_connect(self):
        """@brief Optional hook called once when a viewer connects, before the frame loop. No-op by default."""
        pass


def run_server(source: VRSource, port: int = DEFAULT_PORT, target_fps: int = 30):
    """Allocate double-buffered IPC resources for `source` and serve the streaming loop forever.

    @param source: VRSource implementation supplying per-frame Gaussian attributes.
    @param port: TCP port to listen on for the C++/SIBR viewer.
    @param target_fps: target frame pacing; overridden by the `VR_PRODUCE_HZ`
        env var if set (> 0).
    @note: Runs until KeyboardInterrupt. Accepts and re-accepts connections in
        a loop (a dropped connection does not stop the server
    """
    target_frame_seconds = 1.0 / target_fps
    produce_hz = float(os.environ.get("VR_PRODUCE_HZ", "0"))
    if produce_hz > 0:
        target_frame_seconds = 1.0 / produce_hz
        log.info("[RATE] producer hard-capped to %.1f Hz (%.1f ms/frame)",
                 produce_hz, target_frame_seconds * 1000.0)
    freeze_after = int(os.environ.get("VR_FREEZE_AFTER", "0"))
    if freeze_after:
        log.info("[PAUSE] will auto-pause after %d frames (connection stays up).",
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
                # Animation pause state. A fresh viewer connection always starts
                # running; 
                paused = False
                announced_pause = False
                step_delta = 0
                last_frame = 0
                freeze_armed = bool(freeze_after)   
                last_buf, last_N = 0, source.N_max

                while True:
                    t0 = time.perf_counter()

                    # Client -> server control messages (e.g. GUI identity switch).
                    # Non-blocking; sets a pending request the source applies at the
                    # top of its next produce_frame (between frames, no race).
                    for magic, payload in poll_control(conn):
                        if magic == CTRL_SET_IDENTITY:
                            log.info("[CTRL] set identity -> %d", payload)
                            source.pending_id = payload
                        elif magic == CTRL_SET_PAUSE:
                            paused = bool(payload)
                            log.info("[CTRL] %s at frame %d",
                                     "pause" if paused else "resume", last_frame)
                            if not paused:
                                announced_pause = False
                                step_delta = 0
                        elif magic == CTRL_STEP_FRAME:
                            # Only meaningful while paused; ignore otherwise rather
                            # than nudging a running animation.
                            if paused:
                                step_delta += payload
                                log.info("[CTRL] step %+d (pending %+d)", payload, step_delta)
                            else:
                                log.info("[CTRL] step %+d ignored — not paused", payload)
                        else:
                            log.warning("[CTRL] unknown control magic %r", magic)

                    # Auto-pause diagnostic: sets the same state the Pause button does.
                    # Disarms after firing so the viewer's Resume button still works
                    # (it would otherwise re-pause on the very next iteration).
                    if freeze_armed and anim_frame >= freeze_after:
                        log.info("[PAUSE] auto-pausing at frame %d (VR_FREEZE_AFTER=%d).",
                                 anim_frame, freeze_after)
                        paused = True
                        freeze_armed = False

                    # Paused: keep the socket + C++ rendering alive, but do ZERO
                    # GPU work just re-announce the last good buffer. A pending
                    # step falls through to produce exactly one new frame.
                    if paused and step_delta == 0:
                        if not announced_pause:
                            log.info("[PAUSE] producer stopped on frame %d re-announcing "
                                     "buf %d (N=%d). Python GPU now idle; watch C++ mlp/raster.",
                                     last_frame, last_buf, last_N)
                            announced_pause = True
                        conn.sendall(struct.pack("<QII", pipeline_frame_id, last_buf, last_N))
                        pipeline_frame_id += 1
                        time.sleep(target_frame_seconds)
                        continue

                    if paused:
                        # Step relative to the pose actually on screen (last_frame),
                        # not anim_frame — that already points at the NEXT frame.
                        anim_frame = (last_frame + step_delta) % source.n_frames
                        log.info("[PAUSE] step %+d: frame %d -> %d",
                                 step_delta, last_frame, anim_frame)
                        step_delta = 0
                        announced_pause = False   # re-log once the stepped frame is held

                    # Don't overwrite this buffer until C++ is done reading it.
                    tw = time.perf_counter()
                    ipc_mgrs[buf_idx].wait_read_complete()
                    wait_ms = (time.perf_counter() - tw) * 1000.0

                    # Pipeline-specific frame production
                    t1 = time.perf_counter()
                    xyz, feat, R_bwd, opacity, cov3D = source.produce_frame(anim_frame)
                    produce_ms = (time.perf_counter() - t1) * 1000.0
                    N = xyz.shape[0]

                    # Pack into IPC buffer (GPU-to-GPU, non-blocking)
                    t3 = time.perf_counter()
                    attr_bufs[buf_idx].write(xyz, feat, R_bwd, opacity, cov3D)
                    pack_ms = (time.perf_counter() - t3) * 1000.0

                    # Sync + record data-ready event, announce frame
                    t4 = time.perf_counter()
                    ipc_mgrs[buf_idx].record_data_ready()
                    conn.sendall(struct.pack("<QII", pipeline_frame_id, buf_idx, N))
                    signal_ms = (time.perf_counter() - t4) * 1000.0
                    last_buf, last_N = buf_idx, N   # newest valid buffer (re-announced while paused)
                    last_frame = anim_frame % source.n_frames   # pose now on screen; steps are relative to it

                    total_ms = (time.perf_counter() - t0) * 1000.0

                    remaining = target_frame_seconds - (time.perf_counter() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

                    # anim_frame always means "the frame to produce next" — a paused
                    # loop never reaches here (it re-announces and continues), so
                    # this only runs when free-running or after a step, and in both
                    # cases the next frame is the following one.
                    anim_frame += 1
                    pipeline_frame_id += 1
                    buf_idx = 1 - buf_idx

                    if anim_frame % _LOG_INTERVAL == 0 or anim_frame <= 3:
                        dropped = getattr(
                            getattr(source, "frame_source", None), "dropped", 0)
                        log.info(
                            "frame %4d  N=%d  K=%d  total=%.1fms"
                            "  wait=%.1f  produce=%.1f  pack=%.1f  signal=%.1f"
                            "  dropped=%d",
                            anim_frame, N, source.K, total_ms,
                            wait_ms, produce_ms, pack_ms, signal_ms, dropped,
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
