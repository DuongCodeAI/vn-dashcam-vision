"""Augmentation cho crop biển báo + copy-paste lớp hiếm.

Ảnh thật từ xe máy (dự định quay để test) bị: rung -> nhoè chuyển động, nghiêng khi vào cua,
ngược sáng, ban đêm. Nên augmentation nhấn mạnh đúng mấy thứ đó. KHÔNG lật ngang:
lật "cấm rẽ trái" thành "cấm rẽ phải" là đổi nhãn.
"""

import numpy as np


def motion_blur(img: np.ndarray, k: int, angle_deg: float) -> np.ndarray:
    import cv2

    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def random_crop_aug(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """img: crop biển báo HxWx3 uint8 -> crop đã biến đổi (cùng kích thước)."""
    import cv2

    h, w = img.shape[:2]
    out = img.copy()
    # xoay nhẹ (xe nghiêng khi vào cua) + lệch tâm/scale (bbox detector không khít)
    ang = rng.uniform(-12, 12)
    sc = rng.uniform(0.85, 1.15)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    m[:, 2] += rng.uniform(-0.08, 0.08, 2) * [w, h]
    out = cv2.warpAffine(out, m, (w, h), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.4:  # rung xe máy
        out = motion_blur(out, int(rng.integers(3, 8)), rng.uniform(0, 180))
    if rng.random() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 1.5))
    # sáng/tối/tương phản (ban đêm, ngược sáng)
    alpha, beta = rng.uniform(0.5, 1.4), rng.uniform(-40, 40)
    out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255)
    if rng.random() < 0.2:  # nhiễu cảm biến lúc thiếu sáng
        out = out + rng.normal(0, 8, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.15:  # bị che một phần (cành cây, cột điện)
        y, x = rng.integers(0, h), rng.integers(0, w)
        out[y:y + h // 4, x:x + w // 4] = rng.integers(0, 255)
    return out


def paste_sign(bg: np.ndarray, sign: np.ndarray, rng: np.random.Generator,
               min_size: int = 16, max_size: int = 96) -> tuple[np.ndarray, list[float]]:
    """Dán một crop biển báo lên ảnh nền đường phố (copy-paste augmentation cho lớp hiếm).

    Trả về ảnh mới + bbox YOLO chuẩn hoá (cx, cy, w, h). Biển thường ở nửa trên-bên phải
    khung hình camera hành trình (lề phải), nên vị trí dán lệch về đó cho giống thật.
    """
    import cv2

    H, W = bg.shape[:2]
    size = int(rng.integers(min_size, max_size))
    ratio = sign.shape[0] / max(sign.shape[1], 1)
    sw, sh = size, max(int(size * ratio), 4)
    s = cv2.resize(random_crop_aug(sign, rng), (sw, sh), interpolation=cv2.INTER_AREA)
    x = int(rng.uniform(0.45, 0.95) * (W - sw)) if W > sw else 0
    y = int(rng.uniform(0.05, 0.55) * (H - sh)) if H > sh else 0
    out = bg.copy()
    # mép mềm để không lộ đường cắt
    mask = np.ones((sh, sw), np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), max(size / 40, 0.5))[..., None]
    roi = out[y:y + sh, x:x + sw].astype(np.float32)
    out[y:y + sh, x:x + sw] = (mask * s + (1 - mask) * roi).astype(np.uint8)
    return out, [(x + sw / 2) / W, (y + sh / 2) / H, sw / W, sh / H]
