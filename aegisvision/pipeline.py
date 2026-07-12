"""The pipeline: ties the pieces together and encodes the core logic.

Core rule for now:  detect motion  ->  only if motion, run face detection.

This is where you compose features. To add threat/fire detection later, you
register another detector and add one line deciding when it runs.
"""

import cv2

from .camera import Camera
from .detectors.motion import MotionDetector
from .detectors.tamper import CameraTamperDetector
from .storage.face_store import FaceStore
from .alerts.manager import AlertManager
from .alerts.console import ConsoleAlerter
from .alerts.snapshot import SnapshotAlerter


def build_face_detector(config):
    """Pick the face engine from config. Adding a new engine = one more branch."""
    from .detectors.quality import FaceQualityGate
    gate = FaceQualityGate(config.quality)  # shared quality filter for any engine
    if config.face_engine == "insightface":
        from .detectors.insightface_detector import InsightFaceDetector
        from .detectors.occlusion import OcclusionAnalyzer
        occlusion = OcclusionAnalyzer(config.occlusion)  # skin-region covered-face check
        return InsightFaceDetector(config.insightface, gate, occlusion)
    if config.face_engine == "haar":
        from .detectors.face import FaceDetector
        return FaceDetector(config.face, gate)
    raise ValueError(f"Unknown face_engine: {config.face_engine}")


def build_alert_manager(config, server=None):
    """Assemble the alert outputs. Add the 12V siren / mobile push here later."""
    alerters = [ConsoleAlerter()]
    if config.alert.snapshot:
        alerters.append(SnapshotAlerter(config.alert.snapshot_folder))
    if server is not None:
        from .alerts.api_alerter import ApiAlerter
        alerters.append(ApiAlerter(server))  # feeds the mobile app over Wi-Fi
    return AlertManager(alerters, cooldown_seconds=config.alert.cooldown_seconds)


class SecurityPipeline:
    def __init__(self, config):
        self.cfg = config
        self.motion = MotionDetector(config.motion)
        self.face = build_face_detector(config)
        self.tamper = CameraTamperDetector(config.tamper) if config.tamper.enabled else None
        self.weapon = self._build_weapon_detector(config)
        self.face_store = FaceStore(config.face_store) if config.save_faces else None
        # Start the on-board alert server (for the mobile app) if enabled.
        self.server = None
        if config.server.enabled:
            from .server.alert_server import AlertServer
            self.server = AlertServer(config.server)
            self.server.start()
        self.alerts = build_alert_manager(config, self.server)
        self._covered = False
        self._frame_index = 0

    @staticmethod
    def _build_weapon_detector(config):
        """Pick the weapon-detection backend. ONNX (no torch) is the default;
        'ultralytics' is available if you ever need the .pt/PyTorch path."""
        if not config.yolo.enabled:
            return None
        if config.yolo.backend == "onnx":
            from .detectors.yolo_onnx import YOLOOnnxDetector
            return YOLOOnnxDetector(config.yolo)
        if config.yolo.backend == "ultralytics":
            from .detectors.yolo_detector import YOLODetector
            return YOLODetector(config.yolo)
        raise ValueError(f"Unknown yolo backend: {config.yolo.backend}")

    def _should_process(self) -> bool:
        """Frame-skipping to save CPU on the embedded board."""
        self._frame_index += 1
        return self._frame_index % self.cfg.process_every_n_frames == 0

    def process_frame(self, frame):
        """Run one frame through the logic. Returns (motion, faces, weapons)."""
        # Camera-cover check first: if we're blinded, there's nothing else to
        # see — alert and skip the rest of the pipeline.
        if self.tamper is not None:
            self._covered, _ = self.tamper.check(frame)
            if self._covered:
                if self.cfg.alert_on_tamper:
                    self.alerts.fire("camera_covered",
                                     "Camera appears covered / tampered",
                                     frame=frame)
                return [], [], []

        # Weapon detection runs EVERY frame, independent of motion/faces — a
        # threat is critical and can't wait for the motion gate.
        weapons = []
        if self.weapon is not None:
            weapons = self.weapon.process(frame)
            if weapons and self.cfg.alert_on_weapon:
                names = ", ".join(sorted({w.label for w in weapons}))
                self.alerts.fire("weapon", f"Weapon detected: {names}",
                                 frame=frame, box=weapons[0].box)

        motion = self.motion.process(frame) if self.cfg.motion_gating else []

        # Run face detection when motion gating is off, OR when motion fired.
        faces = []
        if not self.cfg.motion_gating or motion:
            faces = self.face.process(frame)

            # Split faces: concealed (masked/covered) vs clean.
            concealed = [f for f in faces if f.extra.get("concealed")]
            clean = [f for f in faces if not f.extra.get("concealed")]

            # Concealed person -> alert (do NOT enroll them as an identity).
            if concealed and self.cfg.alert_on_concealed:
                self.alerts.fire(
                    "concealed_person",
                    f"{len(concealed)} concealed face(s) detected",
                    frame=frame,
                    box=concealed[0].box,
                )

            # Clean faces -> log/recognise as normal.
            if clean and self.face_store is not None:
                self.face_store.save(frame, clean)

        return motion, faces, weapons

    def _draw(self, frame, motion, faces, weapons):
        if self._covered:
            cv2.putText(frame, "CAMERA COVERED", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            return frame  # nothing else to draw while blinded
        for d in motion:
            x, y, w, h = d.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        for d in faces:
            x, y, w, h = d.box
            if d.extra.get("concealed"):
                colour, label = (0, 140, 255), "CONCEALED"   # orange
            else:
                colour, label = (0, 0, 255), "FACE"          # red
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        for d in weapons:
            x, y, w, h = d.box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3)  # magenta
            cv2.putText(frame, f"{d.label.upper()} {d.confidence:.2f}",
                        (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        return frame

    def run(self):
        with Camera(self.cfg.camera_source) as cam:
            while True:
                frame = cam.read()
                if frame is None:
                    print("[INFO] Stream ended.")
                    break

                if self._should_process():
                    motion, faces, weapons = self.process_frame(frame)
                    if self.cfg.show_window:
                        frame = self._draw(frame, motion, faces, weapons)

                if self.cfg.show_window:
                    cv2.imshow("AegisVision", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if self.cfg.show_window:
                cv2.destroyAllWindows()
