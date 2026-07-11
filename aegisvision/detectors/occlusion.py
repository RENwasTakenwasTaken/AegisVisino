"""Region-based 'covered face' detector.

Where the embedding-norm quality score only says "this face is low quality",
this looks specifically at the nose-and-mouth region and asks "is it actually
skin?". A mask / scarf / helmet / balaclava is NOT skin-coloured, so its skin
ratio collapses — a direct signal that the lower face is covered.

Uses InsightFace's 5 landmarks (left eye, right eye, nose, left mouth corner,
right mouth corner) to locate the region, then measures skin pixels in YCrCb
(more lighting-robust than plain RGB/HSV).

Limits (be aware): a BARE HAND is skin-coloured so it may not trigger here;
and skin detection drifts with lighting / skin tone. That's why the pipeline
combines this with the embedding-norm signal instead of trusting it alone.
"""

import cv2
import numpy as np

# YCrCb skin range — a widely-used, reasonably tone-robust default.
_SKIN_LOW = np.array([0, 133, 77], dtype=np.uint8)
_SKIN_HIGH = np.array([255, 173, 127], dtype=np.uint8)


class OcclusionAnalyzer:
    def __init__(self, config):
        self.cfg = config

    def _lower_face_region(self, frame_shape, kps):
        """Bounding box of the nose->below-mouth area, from landmarks."""
        h_img, w_img = frame_shape[:2]
        _le, _re, nose, lm, rm = kps
        mouth_y = (lm[1] + rm[1]) / 2.0
        left = int(min(lm[0], rm[0]))
        right = int(max(lm[0], rm[0]))
        width = max(1, right - left)
        # Widen sideways, and span from the nose to a bit below the mouth.
        left -= int(0.35 * width)
        right += int(0.35 * width)
        top = int(nose[1])
        bottom = int(mouth_y + (mouth_y - nose[1]) * 1.2)
        # Clamp to the frame.
        left, top = max(0, left), max(0, top)
        right, bottom = min(w_img, right), min(h_img, bottom)
        return left, top, right, bottom

    def _skin_ratio(self, region):
        """Fraction of pixels in the region that look like skin (0.0 - 1.0)."""
        ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, _SKIN_LOW, _SKIN_HIGH)
        return float(mask.mean() / 255.0)

    def is_covered(self, frame, box, kps):
        """Return (covered: bool, info: dict).

        Needs landmarks; without them we can't locate the mouth region, so we
        abstain (return False) and let the norm signal decide.
        """
        if not self.cfg.enabled or kps is None or len(kps) < 5:
            return False, {}

        left, top, right, bottom = self._lower_face_region(frame.shape, kps)
        if right - left < 2 or bottom - top < 2:
            return False, {}

        ratio = self._skin_ratio(frame[top:bottom, left:right])
        covered = ratio < self.cfg.skin_ratio_threshold
        if self.cfg.debug:
            print(f"[OCCLUSION] lower-face skin_ratio={ratio:.2f} "
                  f"(<{self.cfg.skin_ratio_threshold} => covered={covered})")
        return covered, {"skin_ratio": ratio}
