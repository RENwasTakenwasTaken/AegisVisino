"""Saves detected faces to disk, with per-person de-duplication.

Instead of dumping every crop, we compare each new face's embedding against
the people we've already seen. Same person -> we log a "seen again" and skip.
New person -> we assign a new ID, make a folder for them, and save the crop.

This is the foundation for entry logging and suspect-face-search: every person
becomes one folder + one embedding you can later match against.

NOTE: the known-people registry currently lives in memory, so it resets when
the app restarts. Persisting embeddings to disk (e.g. a .npy per person) is the
natural next step — it won't change this class's interface.
"""

import time
from pathlib import Path

import cv2
import numpy as np


class FaceStore:
    def __init__(self, config):
        self.cfg = config
        self.folder = Path(config.folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        # Each entry: {"id": int, "embeddings": [np.ndarray, ...], "count": int}
        # We keep MULTIPLE embeddings per person (different angles/lighting) so a
        # single odd frame can't spawn a duplicate identity.
        self._known = []
        self._next_id = 1

    # ---- geometry helpers -------------------------------------------------
    def _crop(self, frame, box):
        x, y, w, h = box
        pad = int(max(w, h) * self.cfg.padding)
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        return frame[y1:y2, x1:x2]

    # ---- recognition ------------------------------------------------------
    def _match(self, embedding):
        """Return the known person this embedding matches, or None if new.

        We compare against the BEST of each person's stored embeddings, so any
        one of their past angles is enough to recognise them.
        Embeddings are L2-normalised, so cosine similarity == dot product.
        """
        if embedding is None or not self._known:
            return None
        best_person, best_sim = None, -1.0
        for person in self._known:
            sim = max(float(np.dot(embedding, e)) for e in person["embeddings"])
            if sim > best_sim:
                best_person, best_sim = person, sim
        if best_sim >= self.cfg.recognition_threshold:
            return best_person
        return None

    def _register(self, embedding):
        person = {"id": self._next_id, "embeddings": [embedding], "count": 0}
        self._known.append(person)
        self._next_id += 1
        return person

    def _add_sample(self, person, embedding):
        """Grow a person's embedding set, capped, so recognition stays robust."""
        if embedding is None:
            return
        if len(person["embeddings"]) < self.cfg.max_samples_per_person:
            person["embeddings"].append(embedding)

    # ---- main entry point -------------------------------------------------
    def save(self, frame, faces) -> int:
        """Process detected faces. Returns how many NEW people were saved."""
        saved = 0
        stamp = time.strftime("%Y%m%d_%H%M%S")

        for i, face in enumerate(faces):
            embedding = face.extra.get("embedding")
            match = self._match(embedding)

            if match is not None:
                # Known person — note we saw them again, and enrich their
                # embedding set so future recognition is even more robust.
                match["count"] += 1
                self._add_sample(match, embedding)
                print(f"[SEEN] person_{match['id']:03d} "
                      f"(seen {match['count']}x)")
                continue

            # New person (or no embedding available at all).
            person = self._register(embedding) if embedding is not None else None
            crop = self._crop(frame, face.box)
            if crop.size == 0:
                continue

            if person is not None:
                person_dir = self.folder / f"person_{person['id']:03d}"
                person_dir.mkdir(parents=True, exist_ok=True)
                path = person_dir / f"{stamp}_{i}.jpg"
                print(f"[NEW] person_{person['id']:03d} -> {path}")
            else:
                # Fallback: no embeddings (e.g. Haar engine) -> flat dump.
                path = self.folder / f"face_{stamp}_{i}.jpg"
                print(f"[SAVED] {path}")

            cv2.imwrite(str(path), crop)
            saved += 1
        return saved
