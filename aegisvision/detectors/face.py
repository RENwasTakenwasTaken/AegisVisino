"""Face detector.

Uses OpenCV's built-in Haar cascade so it runs with ZERO extra downloads to
get you started. When you move to production, swap this class's internals for
InsightFace/ArcFace — the rest of the app won't notice because it still just
returns a list of Detection objects.
"""

import cv2

from .base import Detector, Detection


class FaceDetector(Detector):
    name = "face"

    def __init__(self, config):
        self.cfg = config
        # Ships inside the opencv-python package, no separate file needed.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")

    def process(self, frame) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.cfg.scale_factor,
            minNeighbors=self.cfg.min_neighbors,
            minSize=(self.cfg.min_size, self.cfg.min_size),
        )
        return [Detection(label="face", box=(int(x), int(y), int(w), int(h)))
                for (x, y, w, h) in faces]
