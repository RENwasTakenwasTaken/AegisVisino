"""Face quality gate.

Decides whether a detected face is "clean enough" to use. The whole point:
a face that's cut off by the frame edge, or missing landmarks, produces a
distorted embedding — which then looks like a brand-new person to the dedup
logic. So we reject those faces up front instead of trying to recognise them.

This is the same idea Google Photos uses: only complete, well-formed faces
get enrolled; everything else is ignored.
"""


class FaceQualityGate:
    def __init__(self, config):
        self.cfg = config

    def check(self, frame_shape, box, kps=None):
        """Return (accepted: bool, reason: str).

        frame_shape : the frame's .shape (h, w, ...)
        box         : (x, y, w, h)
        kps         : optional list of 5 (x, y) facial landmarks (InsightFace)
        """
        h_img, w_img = frame_shape[:2]
        x, y, w, h = box
        m = self.cfg.edge_margin_px

        # 1. Is the bounding box jammed against a frame edge? -> cut-off face.
        if x <= m or y <= m or (x + w) >= (w_img - m) or (y + h) >= (h_img - m):
            return False, "touches frame edge"

        # 2. A half-face box is unusually narrow/wide. Sanity-check the ratio.
        if h <= 0:
            return False, "degenerate box"
        ratio = w / h
        if not (self.cfg.min_aspect <= ratio <= self.cfg.max_aspect):
            return False, f"bad aspect ratio {ratio:.2f}"

        # 3. Every facial landmark must sit inside the frame. If the nose or an
        #    eye is off-screen, the face is incomplete regardless of the box.
        if kps is not None and self.cfg.require_landmarks_inside:
            for (px, py) in kps:
                if px < m or py < m or px > (w_img - m) or py > (h_img - m):
                    return False, "landmark outside frame"

        return True, "ok"
