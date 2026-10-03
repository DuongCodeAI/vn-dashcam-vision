"""Tiền xử lý crop dùng chung cho train (torch) và suy luận (onnxruntime) - phải giống hệt nhau."""

import numpy as np

CROP = 64
PAD = 0.15
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def normalize_batch(batch_bgr: np.ndarray) -> np.ndarray:
    """uint8 (N,H,W,3) BGR -> float32 (N,3,H,W) RGB đã chuẩn hoá."""
    x = (batch_bgr[..., ::-1].astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def crop_box(img: np.ndarray, box, pad: float = PAD, size: int = CROP) -> np.ndarray | None:
    """box x1,y1,x2,y2 (pixel) -> crop vuông size x size, nới rộng pad mỗi phía như lúc tạo dữ liệu."""
    import cv2

    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * (1 + 2 * pad), (y2 - y1) * (1 + 2 * pad)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a, b = int(max(cx - bw / 2, 0)), int(max(cy - bh / 2, 0))
    c, d = int(min(cx + bw / 2, W)), int(min(cy + bh / 2, H))
    if c - a < 4 or d - b < 4:
        return None
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)
