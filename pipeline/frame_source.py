"""FrameSource: owns the cv2.VideoCapture and yields BGR frames.

Two capture modes:

* **Sequential** (video files, or --realtime off): a plain blocking
  ``cap.read()`` — every frame is consumed in order.
* **Realtime** (webcam, or --realtime on): a background daemon thread grabs
  frames as fast as the device delivers them and keeps only the *newest* one.
  ``read()`` returns that newest unseen frame, blocking until one arrives that
  is newer than the last it returned. Frames that pile up while the (slower)
  inference pipeline is busy are dropped, so the avatar tracks the live source
  with low, stable latency instead of falling progressively behind.

`read()` returns the next frame, or raises KeyboardInterrupt at end of
stream / camera failure so the server loop unwinds cleanly.

Instrumentation (readable any time, for the overlay / logs):
  * ``realtime``       — bool, whether latest-frame dropping is active.
  * ``dropped``        — cumulative stale frames the grabber skipped.
  * ``grabbed``        — cumulative frames the grabber pulled off the device.
  * ``last_age_ms``    — age of the frame the last ``read()`` handed out
                         (time from device grab to consumption); a rising or
                         high value means input latency the drop is meant to
                         kill. 0.0 in sequential mode.
"""

import threading
import time

import cv2


class FrameSource:
    """Owns the cv2.VideoCapture and yields BGR frames, in sequential or
    realtime (latest-frame) capture mode; see module docstring.

    @note: Exposes `realtime`, `dropped`, `grabbed`, `last_age_ms` for
        instrumentation (see module docstring for meaning).
    """

    def __init__(self, args):
        """Open the capture device and start the background grabber thread if realtime.

        @param args: parsed CLI namespace; reads `source` ("webcam"/"video"),
            `camera_index`, `video`, `realtime` ("auto"/"on"/"off"), and (webcam
            realtime mode only) `camera_fps`, `camera_width`, `camera_height`.
        @throws SystemExit: if the capture device fails to open.
        """
        if args.source == "webcam":
            cap = cv2.VideoCapture(args.camera_index)
        else:
            cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open input source ({args.source}).")
        self._cap = cap

        # Resolve --realtime {auto,on,off}: auto => on for webcam, off for video.
        realtime = getattr(args, "realtime", "auto")
        if realtime == "auto":
            realtime = args.source == "webcam"
        else:
            realtime = realtime == "on"
        self.realtime = bool(realtime)

        # Instrumentation — always defined; stay 0 in sequential mode.
        self.dropped = 0
        self.grabbed = 0
        self.last_age_ms = 0.0

        if not self.realtime:
            return

        # ── Realtime latest-frame capture ────────────────────────────────────
        # Best-effort device tuning (no-op / ignored on some backends).
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if args.source == "webcam":
            for prop, val in (
                (cv2.CAP_PROP_FPS, getattr(args, "camera_fps", None)),
                (cv2.CAP_PROP_FRAME_WIDTH, getattr(args, "camera_width", None)),
                (cv2.CAP_PROP_FRAME_HEIGHT, getattr(args, "camera_height", None)),
            ):
                if val:
                    try:
                        cap.set(prop, float(val))
                    except Exception:
                        pass

        self._latest = None            # newest grabbed frame (BGR ndarray)
        self._latest_t = 0.0           # perf_counter when it was grabbed
        self._seq = 0                  # incremented on every successful grab
        self._last_returned_seq = 0    # seq of the frame read() last handed out
        self._eos = False              # camera failure / end of stream
        self._stop = False             # release() requested shutdown
        self._cond = threading.Condition()
        self._thread = threading.Thread(
            target=self._grab_loop, name="FrameSourceGrabber", daemon=True)
        self._thread.start()

    def _grab_loop(self):
        """Background thread body: continuously grab frames, keeping only the most recent one.

        @note: Runs until `release()` sets `self._stop`, or the device fails
            (sets `self._eos`). Updates `self.grabbed` and notifies
            `self._cond` on each successful grab.
        """
        while True:
            ok, frame = self._cap.read()
            t = time.perf_counter()
            with self._cond:
                if self._stop:
                    return
                if not ok or frame is None:
                    self._eos = True
                    self._cond.notify_all()
                    return
                self._latest = frame
                self._latest_t = t
                self._seq += 1
                self.grabbed += 1
                self._cond.notify_all()

    def read(self):
        """Return the next frame to process.

        @return: np.ndarray, BGR frame. In sequential mode, the next frame in
            order. In realtime mode, blocks until a frame newer than the last
            one returned is available, updating `self.dropped` and
            `self.last_age_ms`.
        @throws KeyboardInterrupt: at end of stream or on camera failure, so
            the server loop unwinds cleanly.
        """
        if not self.realtime:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                raise KeyboardInterrupt("end of stream")
            return frame

        with self._cond:
            while self._seq <= self._last_returned_seq and not self._eos:
                self._cond.wait()
            if self._seq <= self._last_returned_seq and self._eos:
                raise KeyboardInterrupt("end of stream")
            # Count frames skipped between the last returned one and this newest.
            self.dropped += self._seq - self._last_returned_seq - 1
            self._last_returned_seq = self._seq
            self.last_age_ms = (time.perf_counter() - self._latest_t) * 1000.0
            return self._latest

    def release(self):
        """@brief Stop the grabber thread (if running) and release the capture device."""
        if self.realtime:
            with self._cond:
                self._stop = True
                self._cond.notify_all()
            self._thread.join(timeout=1.0)
        self._cap.release()
