"""Vehicle detection with YOLOX through ONNX Runtime (no deep-learning framework needed).

Replaces the 2021 darkflow/TensorFlow 1.x YOLOv2 setup.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

# COCO classes kept, grouped like the 2021 counts (cars, motorbikes, trucks) with bicycles
# and buses added: two-wheelers with motorbikes, buses with trucks.
VEHICLE_CLASSES = {1: "two-wheeler", 2: "car", 3: "two-wheeler", 5: "heavy", 7: "heavy"}
GROUPS = ["car", "two-wheeler", "heavy"]
GROUP_ID = {name: i for i, name in enumerate(GROUPS)}


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float]:
    """Resize keeping the aspect ratio and pad to a square (YOLOX's own preprocessing)."""
    h, w = image.shape[:2]
    ratio = min(size / h, size / w)
    resized = cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, ratio


def decode_yolox(raw: np.ndarray, size: int, strides=(8, 16, 32)) -> np.ndarray:
    """Turn YOLOX grid outputs (N, 85) into absolute centre/size boxes in input pixels."""
    grids, expanded = [], []
    for stride in strides:
        n = size // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
        expanded.append(np.full((n * n, 1), stride))
    grid = np.concatenate(grids).astype(np.float32)
    stride = np.concatenate(expanded).astype(np.float32)
    out = raw.astype(np.float32).copy()
    out[:, :2] = (out[:, :2] + grid) * stride
    out[:, 2:4] = np.exp(out[:, 2:4]) * stride
    return out


def postprocess(
    decoded: np.ndarray, ratio: float, conf: float = 0.3, iou: float = 0.45
) -> sv.Detections:
    """Keep vehicle classes above ``conf``, apply NMS and map boxes back to the image."""
    class_ids = decoded[:, 5:].argmax(axis=1)
    scores = decoded[:, 4] * decoded[np.arange(len(decoded)), 5 + class_ids]
    keep = (scores >= conf) & np.isin(class_ids, list(VEHICLE_CLASSES))
    if not keep.any():
        return sv.Detections.empty()
    boxes, scores, class_ids = decoded[keep, :4], scores[keep], class_ids[keep]
    xyxy = (
        np.column_stack(
            (
                boxes[:, 0] - boxes[:, 2] / 2,
                boxes[:, 1] - boxes[:, 3] / 2,
                boxes[:, 0] + boxes[:, 2] / 2,
                boxes[:, 1] + boxes[:, 3] / 2,
            )
        )
        / ratio
    )
    xywh = np.column_stack((xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]))
    # Class-agnostic NMS: a vehicle is one vehicle even if two classes fire on it.
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)
    groups = np.array([GROUP_ID[VEHICLE_CLASSES[int(c)]] for c in class_ids[idx]], dtype=int)
    return sv.Detections(
        xyxy=xyxy[idx].astype(np.float32),
        confidence=scores[idx].astype(np.float32),
        class_id=groups,
    )


class YOLOXDetector:
    def __init__(
        self, model_path: Path, input_size: int = 640, conf: float = 0.3, iou: float = 0.45
    ):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.size, self.conf, self.iou = input_size, conf, iou

    def __call__(self, frame_bgr: np.ndarray) -> sv.Detections:
        image, ratio = letterbox(frame_bgr, self.size)
        blob = image.transpose(2, 0, 1)[None].astype(np.float32)
        raw = self.session.run(None, {self.input_name: blob})[0][0]
        detections = postprocess(decode_yolox(raw, self.size), ratio, self.conf, self.iou)
        h, w = frame_bgr.shape[:2]
        if len(detections):
            detections.xyxy[:, [0, 2]] = detections.xyxy[:, [0, 2]].clip(0, w)
            detections.xyxy[:, [1, 3]] = detections.xyxy[:, [1, 3]].clip(0, h)
        return detections
