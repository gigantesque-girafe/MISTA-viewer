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

        self.last_filtered = None          # last valid filtered pose (72,)
        self.missing_count = 0
        self.prev_raw = None               # for --debug frame-to-frame deltas
        self.prev_filt = None

    def process(self, raw, t_now):
        """(raw (72,)|None, timestamp) -> (pose (72,)|None, status str)."""
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
