"""SourceWindow: Python-owned OpenCV preview of the driving frame."""

import time

import cv2


class SourceWindow:
    """Python-owned OpenCV preview of the driving frame + status overlay.

    Purely presentational; the only side effect on the pipeline is that pressing
    'q'/Esc raises KeyboardInterrupt to stop the server cleanly.
    """

    def __init__(self, estimator_name, args, frame_source=None):
        """Create the OpenCV window.

        @param estimator_name: pose-estimator display name, used in the window title.
        @param args: parsed CLI namespace; reads `filter_space`, `smooth`,
            `romp_every_n` in `show()`.
        @param frame_source: optional pipeline.frame_source.FrameSource, used
            to render capture-mode instrumentation (grabbed/dropped/input-age)
            when provided.
        """
        self.args = args
        self.frame_source = frame_source
        self.win = f"{estimator_name} source (drives C++/SIBR avatar)"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        self._t_prev_frame = None
        self._dt_ema = None

    def show(self, frame, status, produce_ms, identity, missing_count):
        """Render one frame with a status overlay and pump the OpenCV event loop.

        @param frame: np.ndarray, BGR frame to display (not modified in place).
        @param status: pose status string (e.g. "OK", "REUSE <n>", "NO POSE",
            "SKIP <n>/<n>"), shown in the overlay.
        @param produce_ms: last `produce_frame()` duration in ms, shown in the overlay.
        @param identity: current identity index, shown in the overlay.
        @param missing_count: consecutive missed-detection count, shown in the overlay.
        @throws KeyboardInterrupt: if the user presses 'q' or Esc, so the
            caller's server loop stops cleanly.
        """
        now = time.perf_counter()
        if self._t_prev_frame is not None:
            dt = now - self._t_prev_frame
            # Smooth the frame TIME (not 1/dt): averaging instantaneous rates is
            # dominated by cheap skip frames and wildly overstates throughput.
            # FPS = 1 / mean(dt) is the true frames-per-second.
            self._dt_ema = dt if self._dt_ema is None else 0.9 * self._dt_ema + 0.1 * dt
        self._t_prev_frame = now

        disp = frame.copy()
        fps = (1.0 / self._dt_ema) if self._dt_ema else 0.0
        line1 = f"FPS: {fps:5.1f}   produce: {produce_ms:6.1f} ms   id: {identity}   pose: {status}"
        sp = f" [{self.args.filter_space}]" if self.args.smooth else ""
        nev = f"  romp 1/{self.args.romp_every_n}" if self.args.romp_every_n > 1 else ""
        line2 = (f"smooth {'ON' if self.args.smooth else 'OFF'}{sp}{nev}  "
                 f"miss={missing_count}  -> C++/SIBR viewer")
        cv2.putText(disp, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(disp, line2, (10, 48), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 220, 220), 1, cv2.LINE_AA)
        fs = self.frame_source
        if fs is not None:
            mode = "DROP" if getattr(fs, "realtime", False) else "SEQ"
            line3 = (f"cap {mode}  grabbed={getattr(fs, 'grabbed', 0)}  "
                     f"dropped={getattr(fs, 'dropped', 0)}  "
                     f"input-age={getattr(fs, 'last_age_ms', 0.0):5.1f} ms")
            cv2.putText(disp, line3, (10, 72), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 180, 255), 1, cv2.LINE_AA)
        if status == "NO POSE":
            cv2.putText(disp, "No pose detected", (10, disp.shape[0] - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow(self.win, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:       # 'q' or Esc -> clean server stop
            raise KeyboardInterrupt("source window quit")
