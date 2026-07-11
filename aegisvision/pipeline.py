"""The pipeline: ties the pieces together and encodes the core logic.

Core rule for now:  detect motion  ->  only if motion, run face detection.

This is where you compose features. To add threat/fire detection later, you
register another detector and add one line deciding when it runs.
"""

import cv2

from .camera import Camera
from .detectors.motion import MotionDetector
from .detectors.face import FaceDetector
from .storage.face_store import FaceStore


class SecurityPipeline:
    def __init__(self, config):
        self.cfg = config
        self.motion = MotionDetector(config.motion)
        self.face = FaceDetector(config.face)
        self.face_store = FaceStore(config.face_store) if config.save_faces else None
        self._frame_index = 0

    def _should_process(self) -> bool:
        """Frame-skipping to save CPU on the embedded board."""
        self._frame_index += 1
        return self._frame_index % self.cfg.process_every_n_frames == 0

    def process_frame(self, frame):
        """Run one frame through the logic. Returns (motion, faces)."""
        motion = self.motion.process(frame)

        faces = []
        if motion:
            # Motion is the gate. Face detection (expensive) only runs
            # when something actually moved.
            faces = self.face.process(frame)
            if faces:
                print(f"[ALERT] Motion + {len(faces)} face(s) detected")
                if self.face_store is not None:
                    self.face_store.save(frame, faces)
            else:
                print("[INFO] Motion detected, no face")

        return motion, faces

    @staticmethod
    def _draw(frame, motion, faces):
        for d in motion:
            x, y, w, h = d.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        for d in faces:
            x, y, w, h = d.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "FACE", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame

    def run(self):
        with Camera(self.cfg.camera_source) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    print("[INFO] Stream ended.")
                    break

                if self._should_process():
                    motion, faces = self.process_frame(frame)
                    if self.cfg.show_window:
                        frame = self._draw(frame, motion, faces)

                if self.cfg.show_window:
                    cv2.imshow("AegisVision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if self.cfg.show_window:
                cv2.destroyAllWindows()
