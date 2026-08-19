"""FrameSource: owns the cv2.VideoCapture and yields BGR frames."""

import cv2


class FrameSource:
    """Owns the cv2.VideoCapture and yields BGR frames.

    `read()` returns the next frame, or raises KeyboardInterrupt at end of
    stream / camera failure so the server loop unwinds cleanly.
    """

    def __init__(self, args):
        if args.source == "webcam":
            cap = cv2.VideoCapture(args.camera_index)
        else:
            cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Failed to open input source ({args.source}).")
        self._cap = cap

    def read(self):
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise KeyboardInterrupt("end of stream")
        return frame

    def release(self):
        self._cap.release()
