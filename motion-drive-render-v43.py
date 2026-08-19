"""
motion-drive-render-v43.py  —  ROMP -> MISTA driving the C++/SIBR viewer (Phase 2)

Phase 1 (`motion-driven-render.py`) rendered MISTA in-process and showed a Python
side-by-side. Phase 2 keeps the SAME working ROMP -> One-Euro filter pipeline but,
instead of rendering in Python, streams the deformed Gaussian attributes to the
EXISTING C++/SIBR viewer over the EXISTING CUDA-IPC path (so the C++ side keeps doing
the rasterization and its ColorMLP), and opens a SECOND Python/OpenCV window that shows
only the ORIGINAL source frame driving the currently-sent pose.

    Window 1 (C++/SIBR process) : rendered MISTA identity   (unchanged viewer)
    Window 2 (this Python proc) : original webcam/video frame

Architecture (nothing on the C++ side changes):
  - The C++ viewer speaks a fixed wire protocol (handshake "V42E", per-frame packet
    <Q pipeline_frame_id, I buf_idx, I N>, control channel CTL0/1/2). It is agnostic to
    where poses come from, so this ROMP-driven server drives the ALREADY-BUILT
    SIBR_remoteGaussianDesktopV42_app (or the OpenXR VR viewer) with NO C++ changes.
  - We reuse `render_vr_v1_modular.MistaVRSource` (deformation + view-independent feature
    extraction + axis fix) and only swap its static predict-sequence `smpl_cams` list for
    a LIVE capture -> ROMP -> One-Euro -> pose_to_camera_fields step inside produce_frame.
  - `vr_viewer.run_server` (TCP / IPC / double-buffering / pacing) is reused UNCHANGED.
  - ColorMLP + Gaussian rasterization stay entirely on the C++ side. We only send the
    5 per-Gaussian attribute tensors, exactly like the original MISTA VR adapter.

Synchronization:
  The same captured frame's pose is deformed, packed into the IPC buffer, announced to
  C++, and cv2.imshow'n WITHIN ONE produce_frame() call, so source-frame N and pose N are
  coupled by call ordering. C++ renders the newest buffer asynchronously, so the avatar
  trails the source window by <= one C++ render frame. The protocol's `pipeline_frame_id`
  is a server counter (not the source-video index); there is no source-index handshake,
  so sync relies on same-call ordering + newest-buffer semantics, not an explicit ID.

Note: the OpenCV source window opens only AFTER a C++ viewer connects, because
produce_frame() runs inside the server's per-connection loop. Start this script first
(it listens), then launch the viewer.

Run:
  # 1) Python server (webcam)
  python motion-drive-render-v43.py --source webcam --camera-index 0 \
      --identity 3 --load-ckpt "<ckpt>.pth" --port 6012
  # 1) Python server (video)
  python motion-drive-render-v43.py --source video --video clip.mp4 \
      --identity 3 --load-ckpt "<ckpt>.pth" --port 6012
  # 2) Desktop viewer (reuse the existing V42 binary)
  submodules\\sibr-core\\install\\bin\\SIBR_remoteGaussianDesktopV42_app_rwdi.exe --ip 127.0.0.1 --port 6012
  # 2) VR viewer (unchanged)
  submodules\\sibr-core\\install\\bin\\SIBR_remoteGaussianOpenXRv4_2_app_rwdi.exe --ip 127.0.0.1 --port 6012

Smoothing / missing-detection behavior and flags are identical to Phase 1.
This file adds NO changes to vr_viewer/*, render_vr_v1_modular.py, motion-driven-render.py
or any C++/SIBR/CUDA-IPC code.
"""

import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch

# Make sure the MISTA package root (this file's directory) is importable, so that
# both the repo-root modules and the `pipeline` package resolve when running this
# script directly.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# MISTA / viewer building blocks (reused as-is).
from render import build_scene                              # checkpoint + scene loader
from vr_viewer import run_server, VRSource                  # streaming loop + pipeline contract (UNCHANGED)
from vr_viewer.colormlp_export import (
    export_color_mlp,
    _view_indep_feat_dim as view_indep_feat_dim,
)

# Pipeline stages (single-responsibility collaborators; see pipeline/), listed in
# the order data flows through them. One-time setup first, then the per-frame chain
# that LiveVRSource sequences: frame -> pose -> smooth -> camera -> deform -> preview.
from pipeline import (
    load_mista_config,  # [setup]  composes the MISTA Hydra config (test split, chosen identity, no PLY export)
    build_estimator,    # [setup]  factory -> a PoseEstimator backend (ROMP or PARE): frame -> raw (72,) pose or None
    FrameSource,        # [1] grabs the next BGR frame from the webcam/video; signals end-of-stream
    #     (the estimator built above runs here: frame -> raw (72,) SMPL pose or None)
    PoseProcessor,      # [2] One-Euro smoothing of the raw pose + missing-detection (reuse/reset) policy
    MistaPoseAdapter,   # [3] turns a (72,) SMPL pose into the MISTA deformer camera (rots + bone transforms)
    MistaRenderer,      # [4] deforms the canonical Gaussians by that camera -> 5 attribute tensors for C++ IPC
    SourceWindow,       # [5] Python OpenCV preview of the driving frame + status overlay ('q'/Esc quits)
)


# --------------------------------------------------------------------------- #
# Orchestrator: LiveVRSource wires the pipeline stages (pipeline/) into the
# vr_viewer server contract. All domain logic lives in the pipeline collaborators;
# this class only sequences them per produce_frame() call.
# --------------------------------------------------------------------------- #
class LiveVRSource(VRSource):
    """Thin orchestrator wiring the pipeline stages into the server contract.

    Owns no domain logic beyond per-frame sequencing and the --romp-every-n skip
    policy: each produce_frame() reads a frame, (every Nth) estimates + processes
    + adapts + renders a pose, freezes the last avatar on miss/skip, previews the
    frame, and hands the 5-tuple back to run_server for IPC to C++.
    """

    def __init__(self, frame_source, estimator, processor, adapter, renderer, args):
        self.frame_source = frame_source
        self.estimator = estimator          # PoseEstimator (ROMP/PARE adapter)
        self.processor = processor
        self.adapter = adapter
        self.renderer = renderer
        self.args = args
        self.identity = int(args.identity)

        # VRSource handshake contract, forwarded from the renderer.
        self.device = renderer.device
        self.N_max = renderer.N_max
        self.K = renderer.K
        self.model_bytes = renderer.model_bytes
        self.n_frames = 1                   # poses are live-sourced, not indexed

        self.last_tensors = None            # last produced 5-tuple (freeze on miss/skip)
        self._romp_ctr = 0                  # drives --romp-every-n skipping
        self.window = None if args.no_window else SourceWindow(estimator.name, args)

    def on_connect(self):
        self.renderer.on_connect()

    def produce_frame(self, frame_idx: int):
        t0 = time.perf_counter()

        # ── Live identity switch from the C++ GUI (CTL0) -> re-decode once ─────
        if self.pending_id is not None:
            self.renderer.switch_identity(int(self.pending_id))
            self.identity = int(self.pending_id)
            self.pending_id = None

        # ── 1) Live source frame ──────────────────────────────────────────────
        frame = self.frame_source.read()

        # ── 2) Run the estimator only every Nth frame (--romp-every-n); reuse the
        #      last avatar on the others. Estimation (~95 ms) dominates the frame,
        #      so on skip frames we do NO estimate and NO deform: the C++ viewer
        #      keeps rendering the last posed buffer while the source window still
        #      advances every frame. N=1 (default) = every frame (unchanged). ────
        run_estimator = (self._romp_ctr % self.args.romp_every_n == 0) or (self.last_tensors is None)
        self._romp_ctr += 1

        if run_estimator:
            # ── estimate -> One-Euro -> pose (missing handled) -> deform + pack ─
            raw = self.estimator.estimate(frame)
            pose, status = self.processor.process(raw, t0)
            if pose is not None:
                cam = self.adapter.to_camera(pose, self.identity)
                tensors = self.renderer.render(cam)
                self.last_tensors = tensors
            elif self.last_tensors is not None:
                tensors = self.last_tensors      # freeze avatar on lost target
            else:
                # No pose yet and nothing cached: send the canonical avatar once.
                tensors = self.renderer.render(self.adapter.canonical())
                self.last_tensors = tensors
        else:
            # ── Skip frame: reuse the last posed avatar (no estimate, no deform) ─
            tensors = self.last_tensors
            status = f"SKIP {self._romp_ctr % self.args.romp_every_n}/{self.args.romp_every_n}"

        produce_ms = (time.perf_counter() - t0) * 1000.0

        # ── 3) Show the source frame (same call => synced with the pose sent) ──
        if self.window is not None:
            self.window.show(frame, status, produce_ms,
                             self.identity, self.processor.missing_count)

        # ── 4) Hand the 5 attribute tensors to the server for IPC to C++ ──────
        return tensors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="ROMP -> MISTA -> C++/SIBR viewer (Phase 2), with a Python source window")
    p.add_argument("--source", choices=["webcam", "video"], default="webcam")
    p.add_argument("--camera-index", type=int, default=0)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--identity", type=int, default=0, help="Target MISTA identity index (0-7).")
    p.add_argument("--estimator", choices=["romp", "pare"], default="romp",
                   help="Pose estimation backend. 'romp' (default) = current "
                        "behavior; 'pare' uses the vendored PARE network.")

    # PARE backend options (only used when --estimator pare). Defaults point at
    # the weights downloaded under the vendored submodule (submodules/PARE).
    p.add_argument("--pare-ckpt", type=str,
                   default="submodules/PARE/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt",
                   help="PARE checkpoint (.ckpt). Only used with --estimator pare.")
    p.add_argument("--pare-cfg", type=str,
                   default="submodules/PARE/data/pare/checkpoints/pare_w_3dpw_config.yaml",
                   help="PARE model config (.yaml). Only used with --estimator pare.")
    p.add_argument("--pare-crop-size", type=int, default=224,
                   help="PARE input crop size (default 224).")
    p.add_argument("--pare-scale", type=float, default=1.0,
                   help="PARE bbox scale factor for the full-frame crop (default 1.0).")
    p.add_argument("--load-ckpt", type=str, required=True,
                   help="Path to the MISTA 8-identity checkpoint (.pth).")
    p.add_argument("--port", type=int, default=6012,
                   help="TCP port the C++/SIBR viewer connects to (default 6012).")
    p.add_argument("--no-window", action="store_true",
                   help="Disable the Python source-frame window (stream to C++ only).")
    p.add_argument("--romp-every-n", type=int, default=1,
                   help="Run ROMP (and re-deform) every N frames, reusing the last "
                        "pose on the others. 1 = every frame (default). N=2 ~doubles "
                        "FPS at the cost of coarser pose sampling.")

    # ROMP inference backend. ONNX+onnxruntime-gpu is ROMP's real-time path (~3x
    # faster than the plain PyTorch backbone) and is the default here; pass
    # --no-onnx to force the PyTorch path. Falls back to PyTorch automatically if
    # onnxruntime's CUDA provider isn't available (so we never silently run on CPU,
    # which would be slower than PyTorch).
    p.add_argument("--onnx", dest="onnx", action="store_true",
                   help="Use ROMP's ONNX GPU backend (default; big speedup).")
    p.add_argument("--no-onnx", dest="onnx", action="store_false",
                   help="Force ROMP's plain PyTorch backbone instead of ONNX.")
    p.set_defaults(onnx=True)

    # One-Euro temporal smoothing (enabled by default; same flags as Phase 1).
    p.add_argument("--smooth", dest="smooth", action="store_true",
                   help="Enable One-Euro smoothing of the ROMP pose (default).")
    p.add_argument("--no-smooth", dest="smooth", action="store_false",
                   help="Disable smoothing; feed the raw ROMP pose.")
    p.set_defaults(smooth=True)
    p.add_argument("--min-cutoff", type=float, default=0.8)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--derivative-cutoff", type=float, default=1.0)
    p.add_argument("--smooth-frequency", type=float, default=12.5)
    p.add_argument("--filter-space", choices=["axis-angle", "quat"], default="quat",
                   help="Rotation filter space. 'quat' (default) is flip-safe; "
                        "'axis-angle' is the naive per-element filter (can flip).")
    p.add_argument("--reuse-frames", type=int, default=12)
    p.add_argument("--reset-after", type=int, default=30)
    p.add_argument("--debug", action="store_true",
                   help="Print raw vs filtered frame-to-frame pose deltas.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.source == "video" and not args.video:
        raise SystemExit("--source video requires --video PATH")
    if not (0 <= args.identity <= 7):
        raise SystemExit("--identity must be in 0..7")
    if args.romp_every_n < 1:
        raise SystemExit("--romp-every-n must be >= 1")

    device = "cuda"

    # ---- MISTA scene (checkpoint + identity restriction) ----
    print(f"[INIT] Loading MISTA scene for identity {args.identity} ...")
    config = load_mista_config(args.load_ckpt, args.identity)
    with torch.no_grad():
        scene, migs_type, _ = build_scene(config)
    print(f"[INIT] Scene ready (migs_type={migs_type}).")

    # Per-identity geometry + a template camera (fixed R/T/FoV, identity Jtrs).
    meta = scene.test_dataset.metadata
    Jtr_target = np.asarray(meta["Jtr"], dtype=np.float64)
    b02v_inv = np.linalg.inv(np.asarray(meta["bone_transforms_02v"], dtype=np.float64))
    template_cam = scene.test_dataset[0]

    # Decode the canonical Gaussians ONCE for this identity (only pose changes/frame).
    with torch.no_grad():
        scene.update_gaussians_from_migs(int(args.identity))
    print(f"[INIT] Canonical avatar ready: {scene.gaussians.get_xyz.shape[0]} Gaussians.")

    # Color-MLP export + feature dim (shipped to C++ once at handshake; runs in C++).
    color_mlp = scene.converter.texture
    K = view_indep_feat_dim(color_mlp)
    model_bytes = export_color_mlp(color_mlp, K, next(color_mlp.parameters()).device)
    N_max = 50_000
    print(f"[INIT] Feature dim K={K}; ColorMLP TorchScript={len(model_bytes)} bytes.")

    # ---- Pose estimator (ROMP or PARE, selected by --estimator) ----
    # The ROMP notes still apply to its backend: we only consume the (72,)
    # axis-angle pose, so ROMP runs --show_largest (single person) + --calc_smpl
    # (disables the unused SMPL mesh forward) with ONNX-GPU when available.
    estimator = build_estimator(args, device)

    # ---- Pipeline stages (FrameSource -> PoseEstimator -> PoseProcessor ->
    #      MistaPoseAdapter -> MistaRenderer, orchestrated by LiveVRSource) ----
    frame_source = FrameSource(args)
    renderer = MistaRenderer(scene, template_cam, K, model_bytes, N_max)
    adapter = MistaPoseAdapter(template_cam, Jtr_target, b02v_inv, renderer.device)
    processor = PoseProcessor(args)
    source = LiveVRSource(frame_source, estimator, processor, adapter, renderer, args)

    print(f"[RUN] Streaming to C++/SIBR on port {args.port}. "
          f"Start the viewer:  ...\\SIBR_remoteGaussianDesktopV42_app_rwdi.exe "
          f"--ip 127.0.0.1 --port {args.port}")
    print(f"[RUN] smoothing={'ON' if args.smooth else 'OFF'}  "
          f"(min_cutoff={args.min_cutoff}, beta={args.beta}, "
          f"derivative_cutoff={args.derivative_cutoff}, freq={args.smooth_frequency}). "
          f"Focus the source window; press 'q'/Esc there to quit.")

    try:
        run_server(source, port=args.port)
    finally:
        frame_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
