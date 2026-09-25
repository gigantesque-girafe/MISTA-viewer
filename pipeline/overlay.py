"""draw_skeleton: cheap 2D joint/bone overlay from ROMP's `last_pj2d`.

Draws directly on the already-captured source frame (no avatar render, no mesh).
Style follows `_draw_joints` in `motion-drive-composite.py` (green dots on the
estimator's 2D joints), extended with bone lines via the SMPL-24 kinematic chain.
"""

import cv2
import numpy as np

from pipeline.adapter import SMPL_24_PARENTS

POINT_COLOR = (0, 255, 0)      # green, matches motion-drive-composite.py's _draw_joints
LINE_COLOR = (0, 200, 255)     # amber, distinguishes bones from joints


def draw_skeleton(frame, pj2d, parents=SMPL_24_PARENTS,
                  point_color=POINT_COLOR, line_color=LINE_COLOR,
                  radius=3, thickness=2):
    """Draw joint dots + parent-child bone lines on `frame`, in place.

    @param frame: np.ndarray, HxWx3 BGR, modified in place.
    @param pj2d: (J,2) array of 2D joint pixel coords (e.g. `RompEstimator.last_pj2d`),
        or None (no-op). Only the first `len(parents)` rows are used.
    @param parents: parent index per joint (-1 = root), SMPL-24 order by default.
    @return: `frame`, for convenience.
    """
    if pj2d is None:
        return frame
    pj2d = np.asarray(pj2d)
    n = min(len(pj2d), len(parents))
    h, w = frame.shape[:2]

    def _valid(pt):
        x, y = pt
        return np.isfinite(x) and np.isfinite(y) and -w <= x <= 2 * w and -h <= y <= 2 * h

    for j in range(n):
        p = parents[j]
        if p < 0 or p >= n:
            continue
        a, b = pj2d[j], pj2d[p]
        if _valid(a) and _valid(b):
            cv2.line(frame, (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))), line_color, thickness, cv2.LINE_AA)

    for j in range(n):
        pt = pj2d[j]
        if _valid(pt):
            cv2.circle(frame, (int(round(pt[0])), int(round(pt[1]))),
                       radius, point_color, -1, cv2.LINE_AA)

    return frame
