"""Saves detected faces to disk.

For now this just crops each face's bounding box and writes it to the "people"
folder. Later, this same class is where face-recognition/embedding + de-dup
("is this a person we've already logged?") will live — the pipeline won't need
to change, because it just hands us (frame, faces).
"""

import time
from pathlib import Path

import cv2


class FaceStore:
    def __init__(self, config):
        self.cfg = config
        self.folder = Path(config.folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._last_save = 0.0  # for the cooldown, so we don't spam the disk

    def _crop(self, frame, box):
        """Crop the face with a little padding, clamped to the frame edges."""
        x, y, w, h = box
        pad = int(max(w, h) * self.cfg.padding)
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        return frame[y1:y2, x1:x2]

    def save(self, frame, faces) -> int:
        """Save each detected face. Returns how many were written."""
        if not faces:
            return 0

        # Cooldown: don't save more than once every `cooldown_seconds`.
        now = time.time()
        if now - self._last_save < self.cfg.cooldown_seconds:
            return 0
        self._last_save = now

        stamp = time.strftime("%Y%m%d_%H%M%S")
        saved = 0
        for i, face in enumerate(faces):
            crop = self._crop(frame, face.box)
            if crop.size == 0:
                continue
            path = self.folder / f"face_{stamp}_{i}.jpg"
            cv2.imwrite(str(path), crop)
            saved += 1
            print(f"[SAVED] {path}")
        return saved
