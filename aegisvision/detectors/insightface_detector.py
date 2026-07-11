"""Face detector + embedder using InsightFace.

Unlike the Haar detector (which only finds boxes), this one ALSO produces a
512-d "embedding" per face — a numeric fingerprint. Two images of the same
person give near-identical embeddings, which is what makes per-person dedup
and suspect-face-search possible.

The embedding is attached to each Detection via `extra["embedding"]`, so the
rest of the app stays unchanged — it's still just a list of Detection objects.
"""

import numpy as np

from .base import Detector, Detection


class InsightFaceDetector(Detector):
    name = "face"

    def __init__(self, config, quality_gate=None, occlusion=None):
        self.cfg = config
        self.quality = quality_gate  # optional FaceQualityGate; None = accept all
        self.occlusion = occlusion   # optional OcclusionAnalyzer (skin-region)
        # Imported lazily so the app still starts if InsightFace isn't installed
        # (e.g. when running the lightweight Haar engine instead).
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name=config.model_name,               # e.g. "buffalo_l" / "buffalo_sc"
            providers=list(config.providers),     # CPU on the Uno Q by default
        )
        # ctx_id = -1 forces CPU; 0+ selects a GPU if you have one.
        self._app.prepare(ctx_id=config.ctx_id,
                          det_size=(config.det_size, config.det_size))

    def process(self, frame) -> list[Detection]:
        detections = []
        for f in self._app.get(frame):
            if f.det_score < self.cfg.min_score:
                continue
            x1, y1, x2, y2 = f.bbox.astype(int)
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            # InsightFace gives 5 landmarks (eyes, nose, mouth corners).
            kps = f.kps.tolist() if getattr(f, "kps", None) is not None else None
            # Quality/occlusion proxy: magnitude of the RAW (un-normalised)
            # ArcFace embedding. High = clean face, low = occluded/blurry.
            quality = (float(np.linalg.norm(f.embedding))
                       if getattr(f, "embedding", None) is not None else None)

            # Quality gate. Two distinct outcomes:
            #  - incomplete (cut off at edge) -> camera artifact, drop silently.
            #  - concealed (in-frame but masked/covered) -> keep + flag; the
            #    pipeline turns this into a "concealed person" alert.
            concealed = False
            if self.quality is not None:
                complete, reason = self.quality.is_complete(frame.shape, box, kps)
                if not complete:
                    print(f"[SKIP] {reason}")
                    continue
                concealed = self._is_concealed(frame, box, kps, quality)

            detections.append(Detection(
                label="face",
                confidence=float(f.det_score),
                box=box,
                # normed_embedding is L2-normalised, so cosine similarity
                # between two of them is just a dot product.
                extra={"embedding": f.normed_embedding, "kps": kps,
                       "quality": quality, "concealed": concealed},
            ))
        return detections

    def _is_concealed(self, frame, box, kps, quality) -> bool:
        """Combine the two 'covered face' signals per the occlusion config:
          - norm  : embedding-magnitude quality gate (general low quality)
          - skin  : lower-face skin-region analysis (specific: mouth covered)
        """
        norm_flag = self.quality.is_concealed(quality) if self.quality else False

        # No skin analyzer configured -> fall back to the norm signal alone.
        if self.occlusion is None:
            return norm_flag

        skin_flag, _ = self.occlusion.is_covered(frame, box, kps)

        mode = self.occlusion.cfg.combine
        if mode == "skin":
            return skin_flag
        if mode == "norm":
            return norm_flag
        if mode == "and":
            return norm_flag and skin_flag
        return norm_flag or skin_flag  # "or" (default)
