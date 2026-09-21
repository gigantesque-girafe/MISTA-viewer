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

# ── Optional per-stage profiler (opt-in; zero cost unless MISTA_PROFILE=1) ──────
# Set MISTA_PROFILE=1 to print median per-stage timings every MISTA_PROFILE_EVERY
# frames. CUDA is async, so we torch.cuda.synchronize() around the GPU stage to
# measure real execution time rather than kernel-launch time.
_PROFILE = os.environ.get("MISTA_PROFILE", "0") == "1"
_PROFILE_EVERY = int(os.environ.get("MISTA_PROFILE_EVERY", "60"))
_PROF = {"cap": [], "est": [], "rtg": [], "rnd": [], "tot": []}


def _prof_report():
    import statistics as _st
    n = len(_PROF["tot"])
    if n == 0:
        return

    def med(k):
        v = _PROF[k]
        return _st.median(v) if v else 0.0

    print(f"[PROFILE] n={n:>4}  capture={med('cap'):6.1f}  estimator={med('est'):6.1f}  "
          f"retarget+cam={med('rtg'):6.1f}  deform/render={med('rnd'):6.1f}  ||  "
          f"produce_total={med('tot'):6.1f} ms  (~{1000.0/max(med('tot'),1e-6):.1f} FPS, medians)",
          flush=True)
    for k in _PROF:
        del _PROF[k][:-300]  # keep memory bounded

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
from pipeline.retarget import (add_retarget_args, build_retargeter,  # optional IK retarget (--retarget)
                               source_jtr_from_betas)


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

    def __init__(self, frame_source, estimator, processor, adapter, renderer, args,
                 head_pose=None):
        self.frame_source = frame_source
        self.estimator = estimator          # PoseEstimator (ROMP/PARE adapter)
        self.processor = processor
        self.adapter = adapter
        self.renderer = renderer
        self.args = args
        # Optional face-driven head pose; overrides SMPL neck+head on the raw pose
        # before smoothing. None => estimator's head joints used as-is.
        self.head_pose = head_pose
        # Zero-dep head-yaw amplification of the estimator's own neck/head joints.
        self.head_amplify = bool(getattr(args, "head_amplify", False))
        self.identity = int(args.identity)

        # VRSource handshake contract, forwarded from the renderer.
        self.device = renderer.device
        self.N_max = renderer.N_max
        self.K = renderer.K
        self.model_bytes = renderer.model_bytes
        self.n_frames = 1                   # poses are live-sourced, not indexed

        self.last_tensors = None            # last produced 5-tuple (freeze on miss/skip)
        self._romp_ctr = 0                  # drives --romp-every-n skipping
        self.window = None if args.no_window else SourceWindow(estimator.name, args, frame_source)

        # Deferred retarget attach: when --retarget is requested without
        # --source-jtr-npz, the solver is built once the actor rest skeleton is
        # locked from the first N valid betas (see below).
        self._defer_retarget = (adapter.retargeter is None and args.retarget)
        self._betas_buf = []                # accumulates estimator.last_betas until lock

    def _maybe_lock_source_skeleton(self):
        """Collect one valid frame's betas; once N are gathered, build source_Jtr
        and attach the retarget solver. No-op unless deferred."""
        if not self._defer_retarget:
            return
        betas = getattr(self.estimator, "last_betas", None)
        if betas is None:
            return
        self._betas_buf.append(np.asarray(betas, dtype=np.float64).reshape(-1))
        if len(self._betas_buf) < self.args.source_betas_frames:
            return
        betas_mean = np.mean(np.stack(self._betas_buf, axis=0), axis=0)
        source_Jtr = source_jtr_from_betas(betas_mean)
        self.adapter.retargeter = build_retargeter(
            self.args, self.adapter.Jtr_target, source_Jtr)
        self._defer_retarget = False
        self._betas_buf = []
        print(f"[INIT] Retarget ON (proportion={self.args.proportion}); "
              f"source_Jtr locked from "
              f"{self.args.source_betas_frames} frames.", flush=True)

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
        if _PROFILE:
            _pc = time.perf_counter()
        frame = self.frame_source.read()
        if _PROFILE:
            _t_cap = (time.perf_counter() - _pc) * 1000.0
            _t_est = _t_rtg = _t_rnd = 0.0

        # ── 2) Run the estimator only every Nth frame (--romp-every-n); reuse the
        #      last avatar on the others. Estimation (~95 ms) dominates the frame,
        #      so on skip frames we do NO estimate and NO deform: the C++ viewer
        #      keeps rendering the last posed buffer while the source window still
        #      advances every frame. N=1 (default) = every frame (unchanged). ────
        run_estimator = (self._romp_ctr % self.args.romp_every_n == 0) or (self.last_tensors is None)
        self._romp_ctr += 1

        if run_estimator:
            # ── estimate -> One-Euro -> pose (missing handled) -> deform + pack ─
            if _PROFILE:
                _pe = time.perf_counter()
            raw_pose, raw_trans = self.estimator.estimate(frame)
            if _PROFILE:
                _t_est = (time.perf_counter() - _pe) * 1000.0
            # Amplify the estimator's own neck+head yaw before smoothing (--head-amplify).
            if self.head_amplify and raw_pose is not None:
                from pipeline.headpose import amplify_head_yaw
                raw_pose = amplify_head_yaw(raw_pose, self.args.head_amplify_gain,
                                            self.args.head_amplify_max)
            # Override neck(12)+head(15) from the face before smoothing (--head-pose).
            if self.head_pose is not None and raw_pose is not None:
                raw_pose = self.head_pose.apply(raw_pose, frame)
            pose, trans, status = self.processor.process((raw_pose, raw_trans), t0)
            if pose is not None:
                # Lock the actor rest skeleton and attach the retarget solver once
                # enough betas are seen (no-op unless deferred / already attached).
                self._maybe_lock_source_skeleton()
                # Optional deterministic IK retarget (--retarget). No-op passthrough
                # when disabled: pose/trans unchanged, correction is None.
                if _PROFILE:
                    _pr = time.perf_counter()
                pose, trans, correction = self.adapter.retarget(pose, trans)
                cam = self.adapter.to_camera(pose, self.identity, trans,
                                             extra_trans=correction)
                if _PROFILE:
                    _t_rtg = (time.perf_counter() - _pr) * 1000.0
                    torch.cuda.synchronize()
                    _pd = time.perf_counter()
                tensors = self.renderer.render(cam)
                if _PROFILE:
                    torch.cuda.synchronize()
                    _t_rnd = (time.perf_counter() - _pd) * 1000.0
                self.last_tensors = tensors
            elif self.last_tensors is not None:
                if self.adapter.retargeter is not None:
                    self.adapter.retargeter.reset()   # drop stale foot locks
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

        if _PROFILE and run_estimator:
            _PROF["cap"].append(_t_cap)
            _PROF["est"].append(_t_est)
            _PROF["rtg"].append(_t_rtg)
            _PROF["rnd"].append(_t_rnd)
            _PROF["tot"].append(produce_ms)
            if len(_PROF["tot"]) % _PROFILE_EVERY == 0:
                _prof_report()

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
    p.add_argument("--realtime", choices=["auto", "on", "off"], default="auto",
                   help="Drop stale frames to track a live source in realtime. "
                        "A background thread keeps only the newest captured "
                        "frame, so the (slower) pipeline never falls behind. "
                        "auto = on for --source webcam, off for --source video.")
    p.add_argument("--camera-fps", type=float, default=None,
                   help="Best-effort webcam capture FPS (CAP_PROP_FPS). "
                        "Ignored by backends that don't honor it.")
    p.add_argument("--camera-width", type=int, default=None,
                   help="Best-effort webcam capture width (CAP_PROP_FRAME_WIDTH).")
    p.add_argument("--camera-height", type=int, default=None,
                   help="Best-effort webcam capture height (CAP_PROP_FRAME_HEIGHT).")
    p.add_argument("--identity", type=int, default=0, help="Target MISTA identity index (0-7).")
    p.add_argument("--estimator", choices=["romp", "bev", "pare", "hybrik"], default="romp",
                   help="Pose estimation backend. 'romp' (default) = current "
                        "behavior; 'bev' uses BEV (ROMP's depth-reasoning successor; "
                        "slower, but its HRNet backbone honors --trt/--onnx via "
                        "onnxruntime); 'pare' uses the vendored PARE network; 'hybrik' "
                        "uses the vendored HybrIK network (higher accuracy, slower).")

    # PARE backend options (only used when --estimator pare). Defaults point at
    # the weights downloaded under the vendored submodule (submodules/PARE).
    p.add_argument("--pare-ckpt", type=str,
                   default=os.path.join(_ROOT, "submodules/PARE/data/pare/checkpoints/pare_w_3dpw_checkpoint.ckpt"),
                   help="PARE checkpoint (.ckpt). Only used with --estimator pare.")
    p.add_argument("--pare-cfg", type=str,
                   default=os.path.join(_ROOT, "submodules/PARE/data/pare/checkpoints/pare_w_3dpw_config.yaml"),
                   help="PARE model config (.yaml). Only used with --estimator pare.")
    p.add_argument("--pare-crop-size", type=int, default=224,
                   help="PARE input crop size (default 224).")
    p.add_argument("--pare-scale", type=float, default=1.0,
                   help="PARE bbox scale factor for the full-frame crop (default 1.0).")

    # HybrIK backend options (only used when --estimator hybrik). Defaults point at
    # the ResNet-34 config + checkpoint (the light backbone — usable on the RTX 1080;
    # swap to an HRNet-W48 config/ckpt for max accuracy at lower FPS). Files live
    # under the vendored submodule (submodules/HybrIK).
    p.add_argument("--hybrik-ckpt", type=str,
                   default=os.path.join(_ROOT, "submodules/HybrIK/pretrained_models/hybrik_res34.pth"),
                   help="HybrIK checkpoint (.pth). Only used with --estimator hybrik.")
    p.add_argument("--hybrik-cfg", type=str,
                   default=os.path.join(_ROOT, "submodules/HybrIK/configs/256x192_adam_lr1e-3-res34_smpl_3d_cam_2x_mix_w_pw3d.yaml"),
                   help="HybrIK model config (.yaml). Only used with --estimator hybrik. "
                        "Must match the checkpoint (default = the ResNet-34 w/ 3DPW cfg).")
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

    # TensorRT backend for ROMP (A/B toggle). --trt routes ROMP's SAME ONNX model
    # through onnxruntime's TensorRT execution provider instead of the CUDA one;
    # the pipeline is otherwise byte-identical, so this is a clean A/B against the
    # plain-CUDA ONNX baseline. TRT needs the ONNX model, so --trt implies ONNX.
    # ROMP-only (ignored for --estimator pare).
    p.add_argument("--trt", dest="trt", action="store_true",
                   help="Run ROMP's ONNX model via the TensorRT EP (A/B vs CUDA).")
    p.add_argument("--no-trt", dest="trt", action="store_false",
                   help="Use the plain-CUDA ONNX baseline (default).")
    p.set_defaults(trt=False)
    p.add_argument("--trt-fp16", dest="trt_fp16", action="store_true",
                   help="Enable TensorRT FP16 mode (default; big speedup).")
    p.add_argument("--no-trt-fp16", dest="trt_fp16", action="store_false",
                   help="Force TensorRT FP32 (numerically safer, slower).")
    p.set_defaults(trt_fp16=True)
    p.add_argument("--trt-cache-dir", type=str,
                   default=os.path.join(os.path.expanduser("~"), ".romp", "trt_cache"),
                   help="Directory for the serialized TensorRT engine + timing cache "
                        "(built once, reused on later launches).")
    p.add_argument("--trt-lib-dir", type=str,
                   default=os.environ.get("TENSORRT_LIB_DIR"),
                   help="Path to the TensorRT 8.6 (CUDA 11.8) 'lib' folder containing "
                        "nvinfer.dll etc. Added to the DLL search path so onnxruntime's "
                        "TensorRT EP can load. Defaults to $TENSORRT_LIB_DIR.")

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
    p.add_argument("--debug-head", action="store_true",
                   help="Print per-frame neck(12)/head(15) rotation magnitude and "
                        "yaw (raw vs filtered) to diagnose under-rotated head turns.")
    p.add_argument("--debug-head-csv", type=str, default=None,
                   help="Optional CSV path to log the --debug-head values per frame "
                        "for offline plotting.")

    # Head-yaw amplification (zero-dep): scale the estimator's OWN neck+head yaw,
    # which is correctly-signed but too small. Independent of --head-pose (face).
    p.add_argument("--head-amplify", action="store_true",
                   help="Amplify the estimator's own neck(12)+head(15) yaw (no face "
                        "detection, no new deps). Correct sign, no twitch; limited range.")
    p.add_argument("--head-amplify-gain", type=float, default=6.0,
                   help="Multiplier on the estimator's head/neck yaw (--head-amplify).")
    p.add_argument("--head-amplify-max", type=float, default=45.0,
                   help="Clamp (deg) on each amplified joint's yaw (--head-amplify).")

    # Face-driven head pose (YuNet + solvePnP). Body estimators collapse head yaw,
    # so this OVERRIDES SMPL neck(12)+head(15) from the face. Off by default.
    p.add_argument("--head-pose", action="store_true",
                   help="Drive the avatar's neck+head from a face-pose source "
                        "(YuNet + solvePnP), overriding the estimator's head joints.")
    p.add_argument("--head-yunet-model", type=str, default=None,
                   help="Path to face_detection_yunet_2023mar.onnx (OpenCV Zoo). "
                        "Required with --head-pose.")
    p.add_argument("--head-gain", type=float, default=1.0,
                   help="Scalar applied to the face-derived head angles.")
    p.add_argument("--head-split", type=float, default=0.5,
                   help="Fraction of the head turn assigned to the neck joint (12); "
                        "the rest goes to the head joint (15).")
    p.add_argument("--head-neutral-frames", type=int, default=15,
                   help="Frames used to auto-calibrate the frontal neutral (assumes "
                        "the actor starts roughly facing the camera).")
    p.add_argument("--head-max-deg", type=float, default=70.0,
                   help="Clamp on each head angle (deg) before gain.")
    p.add_argument("--head-ema", type=float, default=0.5,
                   help="EMA factor for the face angles (0<..<=1; lower = smoother, "
                        "more lag). Suppresses solvePnP jitter/twitches.")
    p.add_argument("--head-slew-deg", type=float, default=12.0,
                   help="Max change per processed frame (deg) for the face yaw; "
                        "lower = fewer twitches but slower fast turns.")
    p.add_argument("--head-a-max", type=float, default=1.0,
                   help="Reject a face frame whose nose/eye asymmetry exceeds this "
                        "(|a|>a_max = garbage/near-profile landmarks); the head holds "
                        "its last value instead of snapping. Lower = stricter.")
    p.add_argument("--head-reset-after", type=int, default=30,
                   help="Consecutive no-face frames before the head recenters to the "
                        "body pose and the frontal neutral recalibrates. Below this, "
                        "the last head yaw is held (no twitch on brief detection drops).")
    p.add_argument("--head-yaw-sign", type=float, choices=[1.0, -1.0], default=1.0,
                   help="Sign for the yaw axis (flip if the head turns the wrong way).")
    p.add_argument("--head-pitch-sign", type=float, choices=[1.0, -1.0], default=1.0,
                   help="(Unused: yaw-only head driving.)")
    p.add_argument("--head-roll-sign", type=float, choices=[1.0, -1.0], default=1.0,
                   help="(Unused: yaw-only head driving.)")

    # Root motion: thread ROMP's cam_trans into bone_transforms so a walk
    # translates the avatar instead of rendering in place. Off by default; the
    # cam_trans -> canonical-frame mapping is tunable live via the flags below.
    p.add_argument("--root-motion", action="store_true",
                   help="Drive avatar translation from the estimator's cam_trans "
                        "(ROMP only). Default off = pinned root (in-place).")
    p.add_argument("--root-scale", type=float, default=1.0,
                   help="Scalar applied to cam_trans before adding to the root.")
    p.add_argument("--root-axis", type=str, default="x,y,z",
                   help="Axis permute+sign mapping cam_trans -> canonical frame, "
                        "e.g. 'x,y,z' (identity) or 'x,-z,y'.")
    p.add_argument("--root-horizontal", action="store_true",
                   help="Zero the depth (Z) component after mapping: keep only "
                        "ground-plane walking, drop noisy toward/away motion.")

    # Deterministic IK retargeting (A/B toggle). Off by default = current
    # pipeline unchanged (the "A" arm). --retarget enables the "B" arm: foot
    # contact / anti-skate + proportion retarget between the filter and adapter.
    add_retarget_args(p)
    return p.parse_args()


def build_trans_xform(args):
    """Closure mapping a raw (3,) cam_trans into the canonical frame, or None.

    Returns None unless --root-motion is set (so the adapter pins the root).
    Applies, in order: axis permute+sign (--root-axis), scale (--root-scale),
    and optional depth zeroing (--root-horizontal).
    """
    if not args.root_motion:
        return None

    idx, sign = [], []
    for tok in args.root_axis.split(","):
        tok = tok.strip().lower()
        s = -1.0 if tok.startswith("-") else 1.0
        axis = tok.lstrip("+-")
        if axis not in ("x", "y", "z"):
            raise SystemExit(f"--root-axis: bad token '{tok}' (use x/y/z with optional -)")
        idx.append({"x": 0, "y": 1, "z": 2}[axis])
        sign.append(s)
    if len(idx) != 3:
        raise SystemExit("--root-axis must have exactly 3 comma-separated axes")

    idx = np.asarray(idx, dtype=np.int64)
    sign = np.asarray(sign, dtype=np.float32)
    scale = float(args.root_scale)
    horizontal = args.root_horizontal

    def xform(trans):
        t = np.asarray(trans, dtype=np.float32).reshape(-1)
        out = (t[idx] * sign) * scale
        if horizontal:
            out[2] = 0.0
        return out.astype(np.float32)

    return xform


def main():
    args = parse_args()
    if args.source == "video" and not args.video:
        raise SystemExit("--source video requires --video PATH")
    if not (0 <= args.identity <= 7):
        raise SystemExit("--identity must be in 0..7")
    if args.romp_every_n < 1:
        raise SystemExit("--romp-every-n must be >= 1")
    if args.trt and args.estimator != "romp":
        print(f"[INIT] --trt only applies to the ROMP backend; ignored for "
              f"{args.estimator}.")
    if args.root_motion and args.estimator == "hybrik" and not args.root_horizontal:
        print("[INIT] --root-motion with HybrIK: X and vertical motion track ROMP, "
              "but HybrIK's monocular depth (Z) is noisy. Add --root-horizontal to "
              "drop depth for a clean jump/lateral drive.")
    if args.root_motion and args.estimator == "pare":
        print("[INIT] --root-motion needs a world translation; the PARE backend "
              "returns none, so the root stays pinned for this backend.")
    if args.trt and not args.onnx:
        # TensorRT runs ROMP's ONNX graph; --no-onnx is incompatible. Force ONNX
        # on (rather than aborting) so A/B scripts can just add/remove --trt.
        print("[INIT] --trt requires the ONNX model; enabling ONNX (ignoring --no-onnx).")
        args.onnx = True

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
    trans_xform = build_trans_xform(args)
    if trans_xform is not None:
        print(f"[INIT] Root motion ON (scale={args.root_scale}, axis={args.root_axis}, "
              f"horizontal={args.root_horizontal}).")
    # Proportion retarget needs the actor rest skeleton (source_Jtr). With
    # --source-jtr-npz it is known up front; otherwise it is locked from the first
    # N valid frames of live betas (LiveVRSource._maybe_lock_source_skeleton), so
    # the solver is attached lazily and the pose path is unchanged until then.
    source_Jtr = None
    if args.retarget and args.source_jtr_npz:
        source_Jtr = np.asarray(np.load(args.source_jtr_npz)["Jtr"], dtype=np.float64)
        print(f"[INIT] Loaded source_Jtr from {args.source_jtr_npz}.")
    defer_retarget = args.retarget and source_Jtr is None
    retargeter = None if defer_retarget else build_retargeter(args, Jtr_target, source_Jtr)
    if retargeter is not None:
        print(f"[INIT] Retarget ON (proportion={args.proportion}).")
    elif defer_retarget:
        print(f"[INIT] Retarget pending: locking actor betas over "
              f"the first {args.source_betas_frames} valid frames ...")
    adapter = MistaPoseAdapter(template_cam, Jtr_target, b02v_inv, renderer.device,
                               trans_xform=trans_xform, retargeter=retargeter)
    processor = PoseProcessor(args)
    if args.head_amplify:
        print(f"[INIT] Head-yaw amplify ON (gain={args.head_amplify_gain}, "
              f"max_deg={args.head_amplify_max}).")
    # Optional face-driven head pose (overrides SMPL neck+head; --head-pose).
    head_pose = None
    if args.head_pose:
        from pipeline.headpose import FaceHeadPose
        head_pose = FaceHeadPose(
            args.head_yunet_model, gain=args.head_gain, split=args.head_split,
            signs=(args.head_yaw_sign, args.head_pitch_sign, args.head_roll_sign),
            neutral_frames=args.head_neutral_frames, max_deg=args.head_max_deg,
            ema=args.head_ema, slew_deg=args.head_slew_deg,
            reset_after=args.head_reset_after, a_max=args.head_a_max,
            debug=args.debug_head)
        print(f"[INIT] Face head-pose ON (yaw-only, gain={args.head_gain}, "
              f"split={args.head_split}, neutral_frames={args.head_neutral_frames}, "
              f"ema={args.head_ema}, slew_deg={args.head_slew_deg}, "
              f"reset_after={args.head_reset_after}).")
    source = LiveVRSource(frame_source, estimator, processor, adapter, renderer, args,
                          head_pose=head_pose)

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
