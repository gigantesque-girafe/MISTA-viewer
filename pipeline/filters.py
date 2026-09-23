"""
Causal One-Euro temporal filters.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class OneEuroFilter:
    """Causal One-Euro low-pass filter, vectorized over a NumPy array.

    @note: Filters each element using only current and previous samples (no
        future frames); state persists across calls. Runs on CPU.
    """

    def __init__(self, min_cutoff=0.5, beta=0.05, d_cutoff=1.0, freq=12.5):
        """@brief Set filter parameters and reset state.
        @param min_cutoff: minimum cutoff frequency (Hz); lower = smoother, more lag.
        @param beta: speed coefficient; higher = less lag on fast motion, more jitter.
        @param d_cutoff: cutoff frequency (Hz) for the derivative low-pass.
        @param freq: assumed sample frequency (Hz), used when `t` is not given
            to `__call__` or when consecutive timestamps are degenerate.
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.freq = float(freq)
        self.reset()

    def reset(self):
        """@brief Clear filter state so the next call is treated as the first sample."""
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    @property
    def initialized(self) -> bool:
        """@brief Return whether the filter has processed at least one sample."""
        return self.x_prev is not None

    @staticmethod
    def _alpha(cutoff, dt):
        """@brief Compute the exponential smoothing factor for a given cutoff and timestep.
        @param cutoff: cutoff frequency (Hz).
        @param dt: timestep (s).
        @return: float smoothing factor in (0, 1].
        """
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None):
        """Filter one new sample.

        @param x: array-like, any shape, cast to float32.
        @param t: optional timestamp (s); if None, or if consecutive calls'
            `dt` is non-finite, <= 1e-4, or > 1.0, `dt` falls back to `1/freq`.
        @return: np.ndarray, same shape as `x`, float32, filtered. On the
            first call, returns `x` unchanged (state initialization).
        """
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
    """Rotation-aware One-Euro filter; drop-in replacement for OneEuroFilter
    taking/returning axis-angle vectors.

    @note: Filters in quaternion space with sign (hemisphere) continuity, then
        converts back to axis-angle, avoiding the axis-angle double-cover/wrap
        discontinuities that briefly flip the avatar.
    """

    def __init__(self, min_cutoff=0.8, beta=0.05, d_cutoff=1.0, freq=12.5):
        """@brief Construct the underlying OneEuroFilter with these parameters (see OneEuroFilter.__init__)."""
        self._oe = OneEuroFilter(min_cutoff, beta, d_cutoff, freq)

    def reset(self):
        """@brief Clear filter state so the next call is treated as the first sample."""
        self._oe.reset()

    @property
    def initialized(self) -> bool:
        """@brief Return whether the filter has processed at least one sample."""
        return self._oe.initialized

    def __call__(self, x, t=None):
        """Filter one new rotation sample.

        @param x: array-like of length J*3, axis-angle vectors for J joints.
        @param t: optional timestamp (s); see `OneEuroFilter.__call__`.
        @return: np.ndarray of shape (J*3,), float32, filtered axis-angle vectors.
        """
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
