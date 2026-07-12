"""Weapon / threat detector running a YOLOv8 ONNX model on onnxruntime.

No PyTorch, no ultralytics on the board — it reuses the same onnxruntime that
InsightFace already runs on. That's why this is the default backend: smaller
install, faster on the QRB2210.

The cost is that we do ourselves what ultralytics did internally — letterbox the
frame, then decode + NMS the raw output. The Detection output is identical to
the ultralytics backend, so the pipeline can't tell them apart.

Get the model file once, on your LAPTOP:
    yolo export model=yolov8n.pt format=onnx      # -> yolov8n.onnx
then copy yolov8n.onnx to the board.
"""

import ast

import cv2
import numpy as np

from .base import Detector, Detection

# Fallback class names if the ONNX file has no embedded metadata (a model
# exported by ultralytics DOES embed them, so this is rarely used).
_COCO = {i: n for i, n in enumerate([
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"])}


class YOLOOnnxDetector(Detector):
    name = "weapon"

    def __init__(self, config):
        self.cfg = config
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            config.model_path, providers=list(config.providers))
        self._input_name = self._session.get_inputs()[0].name

        # Use the model's fixed input size if it has one, else the config value.
        ishape = self._session.get_inputs()[0].shape  # [1, 3, H, W]
        self._imgsz = ishape[2] if isinstance(ishape[2], int) else config.imgsz

        # Class names: prefer names embedded by the ultralytics exporter.
        meta = self._session.get_modelmeta().custom_metadata_map
        if "names" in meta:
            self._names = ast.literal_eval(meta["names"])
        else:
            self._names = _COCO

        wanted = {t.lower() for t in config.target_classes}
        self._target_ids = [i for i, n in self._names.items()
                            if str(n).lower() in wanted]
        found = [self._names[i] for i in self._target_ids]
        print(f"[YOLO-ONNX] model={config.model_path} size={self._imgsz} "
              f"threat classes found: {found or 'NONE'}")

    def _letterbox(self, img):
        """Resize keeping aspect ratio, pad to a square. Returns padded image
        plus the scale + padding needed to map boxes back to the original."""
        h, w = img.shape[:2]
        r = min(self._imgsz / h, self._imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self._imgsz, self._imgsz, 3), 114, dtype=np.uint8)
        dh, dw = (self._imgsz - nh) // 2, (self._imgsz - nw) // 2
        canvas[dh:dh + nh, dw:dw + nw] = resized
        return canvas, r, dw, dh

    def process(self, frame) -> list[Detection]:
        if not self._target_ids:
            return []  # model has none of our threat classes

        img, r, dw, dh = self._letterbox(frame)
        # BGR->RGB, HWC->CHW, scale to 0-1, add batch dim.
        blob = np.ascontiguousarray(
            img[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0

        out = self._session.run(None, {self._input_name: blob})[0]  # [1, 4+C, N]
        preds = out[0].T                              # [N, 4+C]
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(len(scores_all)), class_ids]

        # Keep only confident detections of the classes we care about.
        keep = (confidences >= self.cfg.conf_threshold) & \
               np.isin(class_ids, self._target_ids)
        boxes_xywh, class_ids, confidences = \
            boxes_xywh[keep], class_ids[keep], confidences[keep]
        if len(boxes_xywh) == 0:
            return []

        # Undo the letterbox: center xywh -> original-image x,y,w,h.
        cx, cy, bw, bh = boxes_xywh.T
        x = (cx - bw / 2 - dw) / r
        y = (cy - bh / 2 - dh) / r
        rects = np.stack([x, y, bw / r, bh / r], axis=1)

        idxs = cv2.dnn.NMSBoxes(rects.tolist(), confidences.tolist(),
                                self.cfg.conf_threshold, self.cfg.iou_threshold)
        detections = []
        for i in np.array(idxs).flatten():
            bx, by, bw_, bh_ = rects[i]
            name = self._names[int(class_ids[i])]
            conf = float(confidences[i])
            if self.cfg.debug:
                print(f"[WEAPON] {name} {conf:.2f}")
            detections.append(Detection(label=str(name), confidence=conf,
                                        box=(int(bx), int(by), int(bw_), int(bh_)),
                                        extra={"threat": True}))
        return detections
