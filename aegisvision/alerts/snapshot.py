"""Alert output that saves the triggering frame to disk.

Gives you the evidence image for every alert — a concealed-person snapshot goes
to alerts/concealed_person/<timestamp>.jpg. Invaluable for post-event review.
"""

import time
from pathlib import Path

import cv2

from .base import Alerter, Alert


class SnapshotAlerter(Alerter):
    name = "snapshot"

    def __init__(self, folder="alerts"):
        self.folder = Path(folder)

    def send(self, alert: Alert) -> None:
        if alert.frame is None:
            return
        out_dir = self.folder / alert.event_type
        out_dir.mkdir(parents=True, exist_ok=True)

        frame = alert.frame.copy()
        # Mark what triggered the alert, if we know where it was.
        if alert.box is not None:
            x, y, w, h = alert.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 140, 255), 2)

        path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(path), frame)
        print(f"[SNAPSHOT] {path}")
