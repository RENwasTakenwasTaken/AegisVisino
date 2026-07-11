"""Camera tamper / cover detector.

Detects when the camera has been blinded — covered by a hand, tape, cloth,
spray, or knocked out of focus. A covered lens produces a FLAT, DETAIL-LESS
image: low spatial variation (std) AND low edge content (Laplacian variance).
That pair is the signal, regardless of whether the cover is dark or bright.

We deliberately avoid using brightness alone: a legitimately dark night scene
is dark but still has texture/noise; a covered lens has neither.

To avoid false alarms from someone briefly stepping right up to the lens, the
flat condition must PERSIST for `sustained_frames` processed frames before we
call it tampered.
"""

import cv2


class CameraTamperDetector:
    name = "tamper"

    def __init__(self, config):
        self.cfg = config
        self._covered_streak = 0

    def check(self, frame):
        """Return (covered: bool, info: dict)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std = float(gray.std())                              # spatial variation
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())  # edge / detail
        mean = float(gray.mean())                            # brightness (info only)

        # Flat AND detail-less -> the lens is blocked or defocused.
        flat = std < self.cfg.max_std and lap < self.cfg.max_laplacian
        self._covered_streak = self._covered_streak + 1 if flat else 0
        covered = self._covered_streak >= self.cfg.sustained_frames

        if self.cfg.debug:
            print(f"[TAMPER] std={std:.1f} lap={lap:.1f} mean={mean:.1f} "
                  f"streak={self._covered_streak} covered={covered}")
        return covered, {"std": std, "laplacian": lap, "mean": mean}
