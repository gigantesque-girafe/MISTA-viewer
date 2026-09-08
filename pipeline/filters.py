"""
Causal One-Euro temporal filters (duplicated verbatim from motion-driven-render.py;
v43 is intentionally self-contained so the Phase-1 file stays untouched).
"""

import numpy as np
from scipy.spatial.transform import Rotation


class OneEuroFilter:
    """
    Causal One-Euro low-pass filter, vectorized over a NumPy array. Filters each
    element using only current+previous samples (no future frames). State persists
    across calls. Runs on the CPU on the tiny (<=72,) float32 pose — no GPU transfer.
    """

    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0, freq=12.5):
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
    Rotation-aware One-Euro filter.
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
