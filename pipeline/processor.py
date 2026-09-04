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

        return pose, trans, status
