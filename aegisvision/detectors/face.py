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

    def __init__(self, config, quality_gate=None):
        self.cfg = config
        self.quality = quality_gate  # optional FaceQualityGate; None = accept all
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
        detections = []
        for (x, y, w, h) in faces:
            box = (int(x), int(y), int(w), int(h))
            # Haar has no landmarks/embeddings, so only the completeness (edge +
            # aspect) check applies — no occlusion score is available here.
            if self.quality is not None:
                complete, _ = self.quality.is_complete(frame.shape, box, None)
                if not complete:
                    continue
            detections.append(Detection(label="face", box=box))
        return detections
