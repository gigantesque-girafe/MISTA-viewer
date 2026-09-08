"""PoseProcessor: One-Euro smoothing + missing-detection policy."""

import numpy as np

from pipeline.filters import OneEuroFilter, RotationOneEuroFilter


class PoseProcessor:
    """One-Euro temporal smoothing + missing-detection policy.

    Turns the estimator's raw per-frame (72,) axis-angle pose (or None) into a
    smoothed pose to drive the avatar, applying reuse-on-miss and reset-after
    behaviour. Holds all the filter/missing state that used to live on the
    source class. `missing_count` is exposed for the status overlay.
    """

    def __init__(self, args):
        self.args = args
        oe = dict(min_cutoff=args.min_cutoff, beta=args.beta,
                  d_cutoff=args.derivative_cutoff, freq=args.smooth_frequency)
        FilterCls = RotationOneEuroFilter if args.filter_space == "quat" else OneEuroFilter
        self.filt_global = FilterCls(**oe)
        self.filt_body = FilterCls(**oe)
        # Translation is euclidean -> plain vector One-Euro (no quaternion path).
        self.filt_trans = OneEuroFilter(**oe)

        self.last_filtered = None          # last valid filtered pose (72,)
        self.last_filtered_trans = None    # last valid filtered trans (3,) or None
        self.missing_count = 0
        self.prev_raw = None               # for --debug frame-to-frame deltas
        self.prev_filt = None
        self._head_csv = None              # lazy file handle for --debug-head-csv
        self._head_frame = 0               # frame counter for the head diagnostic

    def process(self, raw, t_now):
        """((raw_pose (72,)|None, raw_trans (3,)|None), timestamp)
        -> (pose (72,)|None, trans (3,)|None, status str)."""
        raw_pose, raw_trans = raw
        if raw_pose is not None:
            self.missing_count = 0
            if self.args.smooth:
                g = self.filt_global(raw_pose[0:3], t_now)
                b = self.filt_body(raw_pose[3:72], t_now)
                pose = np.concatenate([g, b]).astype(np.float32)
                trans = (self.filt_trans(raw_trans, t_now).astype(np.float32)
                         if raw_trans is not None else None)
            else:
                pose = raw_pose
                trans = raw_trans
            self.last_filtered = pose
            self.last_filtered_trans = trans
            status = "OK"
        else:
            self.missing_count += 1
            if self.missing_count >= self.args.reset_after:
                self.filt_global.reset()
                self.filt_body.reset()
                self.filt_trans.reset()
                self.last_filtered = None
                self.last_filtered_trans = None
            if self.last_filtered is not None and self.missing_count <= self.args.reuse_frames:
                pose = self.last_filtered
                trans = self.last_filtered_trans
                status = f"REUSE {self.missing_count}"
            else:
                pose = None
                trans = None
                status = "NO POSE"

        if self.args.debug and raw_pose is not None:
            rd = float(np.linalg.norm(raw_pose - self.prev_raw)) if self.prev_raw is not None else 0.0
            fd = (float(np.linalg.norm(pose - self.prev_filt))
                  if (self.prev_filt is not None and pose is not None) else 0.0)
            print(f"[SMOOTH] raw dpose={rd:7.4f}  filt dpose={fd:7.4f}  "
                  f"miss={self.missing_count}", flush=True)
            self.prev_raw = raw_pose
            self.prev_filt = pose

        if getattr(self.args, "debug_head", False) and raw_pose is not None:
            self._debug_head(raw_pose, pose)

        return pose, trans, status

    def _debug_head(self, raw_pose, pose):
        """Log neck(12)/head(15) rotation magnitude + yaw, raw vs filtered.

        Diagnostic only (guarded by --debug-head). The yaw proxy is the SMPL-local
        Y-axis component of the axis-angle in degrees: a head left/right turn is a
        rotation about the body-up axis (~Y in SMPL local frame), so this tracks the
        over-the-shoulder turn. SMPL splits a head turn across neck(12)+head(15), so
        we also report the summed yaw.
        """
        def aa(p, j):
            return None if p is None else np.asarray(p[3 * j:3 * j + 3], dtype=np.float32)

        def mag(v):
            return 0.0 if v is None else float(np.rad2deg(np.linalg.norm(v)))

        def yaw(v):
            return 0.0 if v is None else float(np.rad2deg(v[1]))

        nr, hr = aa(raw_pose, 12), aa(raw_pose, 15)
        nf, hf = aa(pose, 12), aa(pose, 15)
        sum_yaw_raw = yaw(nr) + yaw(hr)
        sum_yaw_filt = yaw(nf) + yaw(hf)
        print(f"[HEAD] neck raw={mag(nr):5.1f}deg(yaw {yaw(nr):+6.1f}) "
              f"filt={mag(nf):5.1f}deg(yaw {yaw(nf):+6.1f})  "
              f"head raw={mag(hr):5.1f}deg(yaw {yaw(hr):+6.1f}) "
              f"filt={mag(hf):5.1f}deg(yaw {yaw(hf):+6.1f})  "
              f"sum_yaw raw={sum_yaw_raw:+6.1f} filt={sum_yaw_filt:+6.1f}", flush=True)

        csv_path = getattr(self.args, "debug_head_csv", None)
        if csv_path:
            if self._head_csv is None:
                self._head_csv = open(csv_path, "w", newline="")
                self._head_csv.write(
                    "frame,neck_raw_deg,neck_raw_yaw,neck_filt_deg,neck_filt_yaw,"
                    "head_raw_deg,head_raw_yaw,head_filt_deg,head_filt_yaw,"
                    "sum_yaw_raw,sum_yaw_filt\n")
            self._head_csv.write(
                f"{self._head_frame},{mag(nr):.3f},{yaw(nr):.3f},{mag(nf):.3f},{yaw(nf):.3f},"
                f"{mag(hr):.3f},{yaw(hr):.3f},{mag(hf):.3f},{yaw(hf):.3f},"
                f"{sum_yaw_raw:.3f},{sum_yaw_filt:.3f}\n")
            self._head_csv.flush()
        self._head_frame += 1
