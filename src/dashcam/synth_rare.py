"""Sinh thêm ảnh huấn luyện cho lớp hiếm bằng copy-paste.

Lấy crop của lớp hiếm từ tập TRAIN (không bao giờ từ val/test), dán lên ảnh nền là các
ảnh train khác, ghi nhãn YOLO tương ứng. Dùng cho cả detector 1 lớp (học thêm hình dạng
biển lạ) lẫn baseline 52 lớp.

python -m dashcam.synth_rare --data datasets/vnts --rare 50 --per-class 150
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .augment import paste_sign


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/vnts")
    ap.add_argument("--rare", type=int, default=50, help="lớp có ít hơn số này crop train là lớp hiếm")
    ap.add_argument("--per-class", type=int, default=150, help="số ảnh tổng hợp cho mỗi lớp hiếm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.data)
    names = json.loads((root / "names.json").read_text("utf-8"))
    tr = np.load(root / "crops" / "train.npz")
    x, y = tr["x"], tr["y"]
    counts = np.bincount(y, minlength=len(names))
    rare = [c for c in range(len(names)) if 0 < counts[c] < args.rare]
    bgs = sorted((root / "yolo_ncls" / "images" / "train").iterdir())
    print(f"{len(rare)} lớp hiếm: {[names[c] for c in rare]}")

    n = 0
    for c in rare:
        pool = np.where(y == c)[0]
        for _ in range(args.per_class):
            bg_path = bgs[rng.integers(len(bgs))]
            bg = cv2.imread(str(bg_path))
            if bg is None:
                continue
            img, (cx, cy, w, h) = paste_sign(bg, x[rng.choice(pool)], rng)
            stem = f"synth_{names[c]}_{n:05d}".replace(".", "_")
            for d, cls in (("yolo_ncls", c), ("yolo_1cls", 0)):
                cv2.imwrite(str(root / d / "images" / "train" / f"{stem}.jpg"), img)
                # PHẢI giữ nhãn gốc của ảnh nền: bỏ đi thì các biển có sẵn trong ảnh thành "nền",
                # dạy detector bỏ qua biển thật. (Biển dán có thể đè lên biển cũ - chấp nhận, hiếm.)
                old = (root / d / "labels" / "train" / f"{bg_path.stem}.txt").read_text()
                (root / d / "labels" / "train" / f"{stem}.txt").write_text(old + f"{cls} {cx} {cy} {w} {h}\n")
            n += 1
    print(f"đã sinh {n} ảnh")


if __name__ == "__main__":
    main()
