"""Weapon / threat-object detector using Ultralytics YOLO.

Runs an object-detection model each frame and keeps only the classes we treat
as threats (knife, gun, ...). Returns them as Detection objects, so the pipeline
turns any hit into a "weapon" alert through the same alert manager as everything
else.

MODEL NOTE:
  - The default COCO model (yolov8n) already detects "knife" -> works now.
  - "gun" is NOT a COCO class. To detect firearms you must point model_path at
    a weapon-trained model (train on a public gun dataset, or use community
    weapon weights). Until then only the classes your model actually has will
    fire — check the [YOLO] startup line for which target classes were found.
"""

from .base import Detector, Detection


class YOLODetector(Detector):
    name = "weapon"

    def __init__(self, config):
        self.cfg = config
        # Lazy import so the app still starts if ultralytics isn't installed.
        from ultralytics import YOLO

        self._model = YOLO(config.model_path)
        self._names = self._model.names  # {class_id: class_name}

        # Map our wanted class NAMES to the model's class IDs, so we can filter
        # at inference time (cheaper than detecting everything then discarding).
        wanted = {t.lower() for t in config.target_classes}
        self._target_ids = [i for i, n in self._names.items()
                            if n.lower() in wanted]
        found = [self._names[i] for i in self._target_ids]
        print(f"[YOLO] model={config.model_path} threat classes found: "
              f"{found or 'NONE — model has none of target_classes'}")

    def process(self, frame) -> list[Detection]:
        results = self._model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_threshold,
            classes=self._target_ids or None,  # None = no id filter (avoid empty)
            device=self.cfg.device,
            verbose=False,
        )
        detections = []
        if not self._target_ids:
            return detections  # model can't see any of our threats; nothing to do

        for b in results[0].boxes:
            cls_id = int(b.cls[0])
            conf = float(b.conf[0])
            name = self._names[cls_id]
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if self.cfg.debug:
                print(f"[WEAPON] {name} {conf:.2f}")
            detections.append(Detection(label=name, confidence=conf, box=box,
                                        extra={"threat": True}))
        return detections
