"""mAP@0.5 tự tính, để so pipeline 2 tầng (detector + SignNet) với YOLO 1 tầng trên CÙNG thước đo.

`model.val()` của ultralytics chỉ đo được YOLO; pipeline 2 tầng nằm ngoài ultralytics
nên phải tự tính. AP kiểu VOC (nội suy mọi điểm), IoU 0.5.
"""

from collections import defaultdict

import numpy as np

from .yolo_onnx import iou_one_to_many


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(((r[idx + 1] - r[idx]) * p[idx + 1]).sum())


def map50(preds: dict[str, list[tuple[np.ndarray, float, int]]], gts: dict[str, list[tuple[np.ndarray, int]]],
          iou_thr: float = 0.5) -> tuple[float, dict[int, float]]:
    """preds[img] = [(box, score, cls)], gts[img] = [(box, cls)] -> (mAP, AP từng lớp có trong gt)."""
    by_cls_pred = defaultdict(list)
    n_gt = defaultdict(int)
    for img, items in preds.items():
        for box, s, c in items:
            by_cls_pred[c].append((s, img, box))
    gt_by = defaultdict(lambda: defaultdict(list))
    for img, items in gts.items():
        for box, c in items:
            gt_by[c][img].append(box)
            n_gt[c] += 1
    aps = {}
    for c in n_gt:
        dets = sorted(by_cls_pred.get(c, []), key=lambda x: -x[0])
        used = {img: np.zeros(len(b), bool) for img, b in gt_by[c].items()}
        tp = np.zeros(len(dets))
        for k, (_, img, box) in enumerate(dets):
            g = gt_by[c].get(img)
            if not g:
                continue
            ious = iou_one_to_many(box, np.array(g))
            j = int(ious.argmax())
            if ious[j] >= iou_thr and not used[img][j]:
                tp[k] = 1
                used[img][j] = True
        ctp = np.cumsum(tp)
        rec = ctp / n_gt[c]
        prec = ctp / np.arange(1, len(dets) + 1) if len(dets) else np.zeros(0)
        aps[c] = average_precision(rec, prec) if len(dets) else 0.0
    return (float(np.mean(list(aps.values()))) if aps else 0.0), aps
