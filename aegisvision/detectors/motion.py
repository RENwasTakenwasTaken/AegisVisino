"""Frame-differentiation motion detector.

Compares the current frame against the previous one. Where enough pixels
changed, we report motion. This is the cheap "wake up" trigger that decides
whether the expensive detectors (face, threat) should even run.
"""

import cv2

from .base import Detector, Detection


class MotionDetector(Detector):
    name = "motion"

    def __init__(self, config):
        self.cfg = config
        self._prev_gray = None  # the previous frame, for comparison

    def _prepare(self, frame):
        """Convert to grayscale + blur so we compare shapes, not noise."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        k = self.cfg.blur_kernel
        return cv2.GaussianBlur(gray, (k, k), 0)

    def process(self, frame) -> list[Detection]:
        gray = self._prepare(frame)

        # First ever frame: nothing to compare against yet.
        if self._prev_gray is None:
            self._prev_gray = gray
            return []

        # 1. Absolute difference between now and the previous frame.
        delta = cv2.absdiff(self._prev_gray, gray)
        # 2. Anything that changed more than the threshold becomes white.
        thresh = cv2.threshold(delta, self.cfg.diff_threshold, 255,
                               cv2.THRESH_BINARY)[1]
        # 3. Dilate to join broken-up blobs into solid regions.
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Roll the window forward for next time.
        self._prev_gray = gray

        # 4. Find the changed regions and keep only the big ones.
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            if cv2.contourArea(c) < self.cfg.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(Detection(label="motion", box=(x, y, w, h)))
        return detections
