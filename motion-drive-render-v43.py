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
from scipy.spatial.transform import Rotation

# Make sure the MISTA package root (this file's directory) is importable.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# MISTA / viewer building blocks (reused as-is).
from render import build_scene                              # checkpoint + scene loader
from dataset.zjumocap import ZJUMoCapDataset               # for _recompute_bone_transforms
from vr_viewer import run_server                            # generic streaming loop (UNCHANGED)
from vr_viewer.colormlp_export import (
    export_color_mlp,
    _view_indep_feat_dim as view_indep_feat_dim,
)
# The MISTA VR adapter: we subclass its source and reuse its feature helpers verbatim.
from render_vr_v1_modular import (
    MistaVRSource,
    extract_view_indep_features,
    extract_R_bwd,
    cov3D_in_vr_frame,
)

# ROMP pose estimator (installed package).
import romp


# Sentinel deform/texture iteration: past every training/delay/export threshold so no
# iteration-keyed disk writes fire (the VR deform path already avoids render()'s export
# gate, but we stay consistent with Phase 1).
RENDER_ITER = 10 ** 9


# --------------------------------------------------------------------------- #
# One-Euro temporal filter  (duplicated verbatim from motion-driven-render.py;
# v43 is intentionally self-contained so the Phase-1 file stays untouched)
# --------------------------------------------------------------------------- #
class OneEuroFilter:
    """
    Causal One-Euro low-pass filter, vectorized over a NumPy array. Filters each
    element using only current+previous samples (no future frames). State persists
    across calls. Runs on the CPU on the tiny (<=72,) float32 pose — no GPU transfer.
    """

    def __init__(self, min_cutoff=0.8, beta=0.05, d_cutoff=1.0, freq=12.5):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.freq = float(freq)
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @property
    def initialized(self) -> bool:
        return self.x_prev is not None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        x = np.asarray(x, dtype=np.float32)
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x.copy()
        if t is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if (not np.isfinite(dt)) or dt <= 1e-4 or dt > 1.0:
                dt = 1.0 / self.freq
        else:
            dt = 1.0 / self.freq
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat.astype(np.float32)


class RotationOneEuroFilter:
    """
    Rotation-aware One-Euro filter (duplicated verbatim from motion-driven-render.py).
    Filters rotations in QUATERNION space with sign (hemisphere) continuity, then
    converts back to axis-angle, avoiding the axis-angle double-cover/wrap
    discontinuities that briefly flip the avatar. Input/output are axis-angle
    vectors (J*3,); drop-in replacement for OneEuroFilter.
    """

    def __init__(self, min_cutoff=0.8, beta=0.05, d_cutoff=1.0, freq=12.5):
        self._oe = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)

    def reset(self):
        self._oe.reset()

    @property
    def initialized(self) -> bool:
        return self._oe.initialized

    def __call__(self, x, t=None):
        aa = np.asarray(x, dtype=np.float32).reshape(-1)
        J = aa.shape[0] // 3
        q = Rotation.from_rotvec(aa.reshape(J, 3)).as_quat().astype(np.float32)  # (J,4) xyzw
        if self._oe.x_prev is not None:
            q_ref = np.asarray(self._oe.x_prev, dtype=np.float32).reshape(J, 4)
            flip = np.sum(q * q_ref, axis=1) < 0.0
            q[flip] *= -1.0
        qf = self._oe(q.reshape(-1), t).reshape(J, 4)
        n = np.linalg.norm(qf, axis=1, keepdims=True)
        n[n < 1e-8] = 1.0
        qf = qf / n
        return Rotation.from_quat(qf).as_rotvec().astype(np.float32).reshape(-1)


# --------------------------------------------------------------------------- #
# Pose -> Camera fields  (duplicated verbatim from motion-driven-render.py)
# --------------------------------------------------------------------------- #
def pose_to_camera_fields(smpl_thetas_72, Jtr_target, b02v_inv, device):
    """
    Convert a single ROMP SMPL pose (72-d axis-angle) into the tensors MISTA's
    deformer consumes: `rots` (1,24,9) and `bone_transforms` (24,4,4). Mirrors
    dataset/zjumocap.py getitem() exactly. Fixed camera => trans = 0.
    """
    thetas = np.asarray(smpl_thetas_72, dtype=np.float32).reshape(-1)
    root_orient = thetas[0:3]
    pose_body = thetas[3:66]     # 63 = 21 joints
    pose_hand = thetas[66:72]    # 6  = 2 joints (zero-padded by ROMP)

    pose_full = np.concatenate([root_orient, pose_body, pose_hand], axis=-1)
    pose_mat_full = Rotation.from_rotvec(pose_full.reshape([-1, 3])).as_matrix()  # (24,3,3)
    pose_mat = pose_mat_full[1:, ...].copy()                                      # (23,3,3)
    pose_rot = np.concatenate(
        [np.expand_dims(np.eye(3), axis=0), pose_mat], axis=0
    ).reshape([-1, 9])                                                            # (24,9)
    rots = torch.from_numpy(pose_rot).float().unsqueeze(0).to(device)            # (1,24,9)

    bt = ZJUMoCapDataset._recompute_bone_transforms(
        root_orient, pose_body, pose_hand, Jtr_target
    )                                                                            # (24,4,4)
    bt = (bt @ b02v_inv).astype(np.float32)
    bone_transforms = torch.from_numpy(bt).to(device)                            # (24,4,4)

    return rots, bone_transforms


# --------------------------------------------------------------------------- #
# Config (same compose pattern as motion-driven-render.py)
# --------------------------------------------------------------------------- #
def load_mista_config(load_ckpt: str, identity: int):
    configs_dir = os.path.join(_ROOT, "configs")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=configs_dir):
        config = compose(config_name="config_5d")

    OmegaConf.set_struct(config, False)
    config.mode = "test"                 # test split -> real ZJU frames + per-identity Jtr
    config.appearance_identity = int(identity)
    config.load_ckpt = load_ckpt
    config.dataset.preload = False
    config.wandb_disable = True
    if config.get("export", None) is not None:
        config.export.enable = False     # never write PLY snapshots
    return config


# --------------------------------------------------------------------------- #
# ROMP-driven VR source: reuse MistaVRSource deform/feature path, live-source poses
# --------------------------------------------------------------------------- #
class RompMistaVRSource(MistaVRSource):
    """
    Subclass of the MISTA VR adapter that replaces the static predict-sequence
    `smpl_cams` with a LIVE webcam/video -> ROMP -> One-Euro -> pose_to_camera_fields
    step, and shows the source frame in a Python OpenCV window. Everything downstream
    (deformation, feature extraction, IPC transport, C++ ColorMLP/rasterization) is the
    existing path, untouched.
    """

    def __init__(self, scene, template_cam, Jtr_target, b02v_inv,
                 K, model_bytes, N_max, cap, romp_model, args):
        # Reuse all of MistaVRSource's setup; n_frames=1 placeholder (frame_idx is
        # ignored for sourcing — we grab the next live frame each call).
        super().__init__(scene, [template_cam], RENDER_ITER, K, model_bytes, N_max)

        self.template_cam = template_cam
        self.Jtr_target = Jtr_target
        self.b02v_inv = b02v_inv
        self.cap = cap
        self.romp = romp_model
        self.args = args
        self.identity = int(args.identity)

        # One-Euro filters (global orientation + body/hand joints), persistent state.
        oe = dict(min_cutoff=args.min_cutoff, beta=args.beta,
                  d_cutoff=args.derivative_cutoff, freq=args.smooth_frequency)
        FilterCls = RotationOneEuroFilter if args.filter_space == "quat" else OneEuroFilter
        self.filt_global = FilterCls(**oe)
        self.filt_body = FilterCls(**oe)

        self.last_filtered = None          # last valid filtered pose (72,)
        self.missing_count = 0
        self.last_tensors = None           # last produced 5-tuple (avatar freeze on miss)
        self.prev_raw = None               # for --debug frame-to-frame deltas
        self.prev_filt = None

        self.win = "ROMP source (drives C++/SIBR avatar)"
        if not args.no_window:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        self._t_prev_frame = None
        self._fps_ema = None

    # -- pose extraction + One-Euro filtering + missing handling -------------- #
    def _get_filtered_pose(self, frame, t_now):
        with torch.no_grad():
            out = self.romp(frame)

        raw = None
        if out is not None and out.get("smpl_thetas", None) is not None \
                and len(out["smpl_thetas"]) > 0:
            raw = np.asarray(out["smpl_thetas"][0], dtype=np.float32).reshape(-1)

        if raw is not None:
            self.missing_count = 0
            if self.args.smooth:
                g = self.filt_global(raw[0:3], t_now)
                b = self.filt_body(raw[3:72], t_now)
                pose = np.concatenate([g, b]).astype(np.float32)
            else:
                pose = raw
            self.last_filtered = pose
            status = "OK"
        else:
            self.missing_count += 1
            if self.missing_count >= self.args.reset_after:
                self.filt_global.reset()
                self.filt_body.reset()
                self.last_filtered = None
            if self.last_filtered is not None and self.missing_count <= self.args.reuse_frames:
                pose = self.last_filtered
                status = f"REUSE {self.missing_count}"
            else:
                pose = None
                status = "NO POSE"

        if self.args.debug and raw is not None:
            rd = float(np.linalg.norm(raw - self.prev_raw)) if self.prev_raw is not None else 0.0
            fd = (float(np.linalg.norm(pose - self.prev_filt))
                  if (self.prev_filt is not None and pose is not None) else 0.0)
            print(f"[SMOOTH] raw dpose={rd:7.4f}  filt dpose={fd:7.4f}  "
                  f"miss={self.missing_count}", flush=True)
            self.prev_raw = raw
            self.prev_filt = pose

        return pose, status

    # -- deform + feature extraction (identical to MistaVRSource.produce_frame) -- #
    def _deform_and_pack(self, cam):
        with torch.no_grad():
            cam_c, _ = self.converter.pose_correction(cam, self.iteration)
            deformed_pc, _ = self.converter.deformer(
                self.gaussians, cam_c, self.iteration, compute_loss=False
            )
            torch.cuda.synchronize()

            feat = extract_view_indep_features(self.color_mlp, deformed_pc, cam_c)
            R_bwd_f = extract_R_bwd(deformed_pc, self.cano, self.axis_fix)

            xyz = deformed_pc.get_xyz @ self.axis_fix.T
            if self.avatar_center0 is None:
                self.avatar_center0 = xyz.mean(dim=0, keepdim=True).clone()
            xyz = xyz - self.avatar_center0
            opacity = deformed_pc.get_opacity.squeeze(-1)
            cov6 = deformed_pc.get_covariance()
            cov3D = cov3D_in_vr_frame(cov6, self.axis_fix)
        return xyz, feat, R_bwd_f, opacity, cov3D

    # -- source-frame window (Python-owned); q/Esc stops the server cleanly ---- #
    def _show_source(self, frame, status, produce_ms):
        if self.args.no_window:
            return
        now = time.perf_counter()
        if self._t_prev_frame is not None:
            dt = now - self._t_prev_frame
            inst = 1.0 / dt if dt > 0 else 0.0
            self._fps_ema = inst if self._fps_ema is None else 0.9 * self._fps_ema + 0.1 * inst
        self._t_prev_frame = now

        disp = frame.copy()
        fps = self._fps_ema if self._fps_ema is not None else 0.0
        line1 = f"FPS: {fps:5.1f}   produce: {produce_ms:6.1f} ms   id: {self.identity}   pose: {status}"
        sp = f" [{self.args.filter_space}]" if self.args.smooth else ""
        line2 = (f"smooth {'ON' if self.args.smooth else 'OFF'}{sp}  "
                 f"miss={self.missing_count}  -> C++/SIBR viewer")
        cv2.putText(disp, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 220), 1, cv2.LINE_AA)
        if status == "NO POSE":
            cv2.putText(disp, "No pose detected", (10, disp.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:       # 'q' or Esc -> clean server stop
            raise KeyboardInterrupt("source window quit")

    def produce_frame(self, frame_idx: int):
        t0 = time.perf_counter()

        # ── Live identity switch from the C++ GUI (CTL0) -> re-decode once ─────
        if self.pending_id is not None:
            with torch.no_grad():
                self.scene.update_gaussians_from_migs(int(self.pending_id))
            self.identity = int(self.pending_id)
            self.pending_id = None

        # ── 1) Live source frame ──────────────────────────────────────────────
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # End of video (or camera failure): stop the server cleanly.
            raise KeyboardInterrupt("end of stream")

        # ── 2) ROMP -> One-Euro -> pose (missing handled) ─────────────────────
        pose, status = self._get_filtered_pose(frame, t0)

        # ── 3-5) Build live camera, deform, pack. On no-pose (beyond reuse),
        #         re-send the last produced attributes so the avatar just holds. ─
        if pose is not None:
            rots, bone_transforms = pose_to_camera_fields(
                pose, self.Jtr_target, self.b02v_inv, self.device
            )
            cam = self.template_cam.copy()
            cam.update(rots=rots, bone_transforms=bone_transforms)
            cam.person_id = self.identity
            tensors = self._deform_and_pack(cam)
            self.last_tensors = tensors
        elif self.last_tensors is not None:
            tensors = self.last_tensors      # freeze avatar on lost target
        else:
            # No pose yet and nothing cached: send the canonical (undeformed) avatar
            # once so the buffer is valid. Uses the template camera as-is.
            tensors = self._deform_and_pack(self.template_cam)
            self.last_tensors = tensors

        produce_ms = (time.perf_counter() - t0) * 1000.0

        # ── 6) Show the source frame (same call => synced with the pose sent) ──
        self._show_source(frame, status, produce_ms)

        # ── 7) Hand the 5 attribute tensors to the server for IPC to C++ ──────
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
    p.add_argument("--load-ckpt", type=str, required=True,
                   help="Path to the MISTA 8-identity checkpoint (.pth).")
    p.add_argument("--port", type=int, default=6012,
                   help="TCP port the C++/SIBR viewer connects to (default 6012).")
    p.add_argument("--no-window", action="store_true",
                   help="Disable the Python source-frame window (stream to C++ only).")

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

    # ---- ROMP ----
    print("[INIT] Initializing ROMP ...")
    romp_model = romp.ROMP(romp.romp_settings(input_args=[]))
    print("[INIT] ROMP ready.")

    # ---- Input source ----
    if args.source == "webcam":
        cap = cv2.VideoCapture(args.camera_index)
    else:
        cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input source ({args.source}).")

    source = RompMistaVRSource(
        scene, template_cam, Jtr_target, b02v_inv,
        K, model_bytes, N_max, cap, romp_model, args,
    )

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
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
