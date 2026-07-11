"""Camera source wrapper.

Isolates *where frames come from* (webcam, USB, RTSP/IP CCTV) from the rest of
the app. When you add a second camera later, you just create a second Camera
object — nothing else changes.
"""

import cv2


class Camera:
    def __init__(self, source=0):
        self.source = source
        self._cap = None

    def open(self):
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source}")
        return self

    def read(self):
        """Return the next frame, or None if the stream ended/failed."""
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self):
        if self._cap is not None:
            self._cap.release()

    # Allow `with Camera(...) as cam:` usage for clean shutdown.
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.release()
