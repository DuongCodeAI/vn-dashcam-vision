"""Chạy detector YOLO11 đã export ONNX chỉ bằng numpy + onnxruntime.

Không dùng ultralytics lúc suy luận: đỡ kéo torch (~1GB) lên laptop và dễ nhúng vào
viet-copilot. Phải tự làm 3 việc mà ultralytics vẫn giấu đi:
1. letterbox: resize giữ tỉ lệ + pad về 640x640 (hoặc imgsz lúc train)
2. decode output (1, 4+nc, N): cx, cy, w, h + điểm từng lớp, KHÔNG có objectness như YOLOv5
3. NMS
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    box: np.ndarray  # x1, y1, x2, y2 theo toạ độ ảnh gốc
    score: float
    cls: int


def letterbox(img: np.ndarray, size: int = 640, pad_value: int = 114) -> tuple[np.ndarray, float, tuple[int, int]]:
    """img HxWx3 (BGR/RGB đều được) -> ảnh size x size, tỉ lệ scale, (pad_x, pad_y)."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else img
    out = np.full((size, size, 3), pad_value, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = resized
    return out, r, (px, py)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5) -> list[int]:
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        ious = iou_one_to_many(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_thr]
    return keep


def decode(output: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.5,
           class_agnostic: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """output (1, 4+nc, N) -> boxes xyxy (theo ảnh letterbox), scores, classes.

    class_agnostic NMS: biển báo hay đứng sát nhau (cột có 2-3 biển), nhưng một vật
    thể không thể vừa là P.130 vừa là P.131a -> NMS chung mọi lớp.
    """
    pred = output[0].T  # (N, 4+nc)
    cls_scores = pred[:, 4:]
    cls = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(cls)), cls]
    m = scores > conf_thr
    if not m.any():
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    xywh, scores, cls = pred[m, :4], scores[m], cls[m]
    boxes = np.stack([xywh[:, 0] - xywh[:, 2] / 2, xywh[:, 1] - xywh[:, 3] / 2,
                      xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    if class_agnostic:
        keep = nms(boxes, scores, iou_thr)
    else:
        keep = []
        for c in np.unique(cls):
            idx = np.where(cls == c)[0]
            keep += [int(idx[k]) for k in nms(boxes[idx], scores[idx], iou_thr)]
    return boxes[keep], scores[keep], cls[keep]


def unletterbox(boxes: np.ndarray, r: float, pad: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, shape[1])
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, shape[0])
    return out


class YoloOnnx:
    def __init__(self, path: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, threads: int | None = None):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou

    def __call__(self, img_bgr: np.ndarray) -> list[Detection]:
        lb, r, pad = letterbox(img_bgr, self.imgsz)
        x = lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, HWC->CHW
        out = self.sess.run(None, {self.inp: np.ascontiguousarray(x)})[0]
        boxes, scores, cls = decode(out, self.conf, self.iou)
        boxes = unletterbox(boxes, r, pad, img_bgr.shape[:2])
        return [Detection(b, float(s), int(c)) for b, s, c in zip(boxes, scores, cls, strict=True)]
