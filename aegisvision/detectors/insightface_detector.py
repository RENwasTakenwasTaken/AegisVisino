"""Face detector + embedder using InsightFace.

Unlike the Haar detector (which only finds boxes), this one ALSO produces a
512-d "embedding" per face — a numeric fingerprint. Two images of the same
person give near-identical embeddings, which is what makes per-person dedup
and suspect-face-search possible.

The embedding is attached to each Detection via `extra["embedding"]`, so the
rest of the app stays unchanged — it's still just a list of Detection objects.
"""

from .base import Detector, Detection


class InsightFaceDetector(Detector):
    name = "face"

    def __init__(self, config):
        self.cfg = config
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
            detections.append(Detection(
                label="face",
                confidence=float(f.det_score),
                box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                # normed_embedding is L2-normalised, so cosine similarity
                # between two of them is just a dot product.
                extra={"embedding": f.normed_embedding},
            ))
        return detections
