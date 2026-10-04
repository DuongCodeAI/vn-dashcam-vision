"""Chuẩn bị dataset VNTS (Kaggle: maitam/vietnamese-traffic-signs, CC BY-SA 4.0) cho 2 tầng.

Đầu ra:
  yolo_1cls/   ảnh + nhãn YOLO, mọi biển gộp thành 1 lớp "sign"   -> train detector tầng 1
  yolo_ncls/   ảnh + nhãn YOLO giữ 52 lớp                           -> baseline YOLO 1 tầng
  crops/{train,val,test}.npz  crop 64x64 + nhãn                     -> train classifier tầng 2

Chia train/val/test: dùng file split có sẵn của dataset nếu tìm thấy. Nếu không, chia theo
"nhóm" = tên file bỏ số thứ tự ở cuối, vì ảnh dashcam liên tiếp từ cùng một video gần như
giống hệt nhau; chia ngẫu nhiên từng ảnh thì test bị rò rỉ từ train, điểm đẹp giả.

python -m dashcam.prepare_vnts --src /kaggle/input/vietnamese-traffic-signs --out datasets/vnts
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_names(src: Path) -> list[str]:
    """Tên lớp: data.yaml (names) / classes.txt / obj.names."""
    for y in src.rglob("*.yaml"):
        txt = y.read_text("utf-8", errors="ignore")
        if "names" in txt:
            import yaml

            names = yaml.safe_load(txt)["names"]
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    for pat in ("classes.txt", "obj.names", "*.names"):
        for f in src.rglob(pat):
            return [ln.strip() for ln in f.read_text("utf-8").splitlines() if ln.strip()]
    raise FileNotFoundError("không thấy file tên lớp (data.yaml / classes.txt)")


def group_key(stem: str) -> str:
    return re.sub(r"[_\-]?\d+$", "", stem) or stem


def split_by_group(stems: list[str], val: float = 0.1, test: float = 0.1) -> dict[str, str]:
    out = {}
    for s in stems:
        h = int(hashlib.md5(group_key(s).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out[s] = "test" if h < test else "val" if h < test + val else "train"
    return out


def read_split_lists(src: Path) -> dict[str, str] | None:
    """File split của dataset: mỗi dòng là đường dẫn ảnh, tên file chứa train/val/test."""
    res = {}
    for f in src.rglob("*.txt"):
        name = f.stem.lower()
        sp = next((k for k, keys in (("train", ("train",)), ("val", ("val", "valid")), ("test", ("test",)))
                   if any(x in name for x in keys)), None)
        if sp is None or f.parent.name.lower() in ("labels",):
            continue
        lines = f.read_text("utf-8", errors="ignore").splitlines()
        if lines and any(ln.strip().lower().endswith(tuple(IMG_EXT)) for ln in lines[:5]):
            for ln in lines:
                if ln.strip():
                    res[Path(ln.strip()).stem] = sp
    return res or None


def add_val(split: dict[str, str], stems: list[str], val: float = 0.1) -> dict[str, str]:
    """VNTS chỉ có train_files.txt / test_files.txt (không có val) -> tách val từ phần train
    theo nhóm, test giữ nguyên. Ảnh không nằm trong file split nào thì tính là train."""
    by_group = split_by_group(stems, val=val, test=0.0)
    out = {}
    for s in stems:
        sp = split.get(s, "train")
        out[s] = "val" if sp == "train" and by_group[s] == "val" else sp
    return out


def read_labels(p: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if p.exists():
        for ln in p.read_text().splitlines():
            parts = ln.split()
            if len(parts) >= 5:
                rows.append((int(float(parts[0])), *map(float, parts[1:5])))
    return rows


def crop(img: np.ndarray, cx, cy, w, h, pad: float = 0.15, size: int = 64) -> np.ndarray | None:
    import cv2

    H, W = img.shape[:2]
    bw, bh = w * W * (1 + 2 * pad), h * H * (1 + 2 * pad)
    x1, y1 = int(cx * W - bw / 2), int(cy * H - bh / 2)
    x2, y2 = int(cx * W + bw / 2), int(cy * H + bh / 2)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def _link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            os.symlink(src, dst)
        except OSError:  # Windows không có quyền symlink -> copy
            import shutil

            shutil.copy(src, dst)


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="datasets/vnts")
    ap.add_argument("--crop-size", type=int, default=64)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)

    names = find_names(src)
    images = [p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT]
    split = read_split_lists(src)
    how = "file split của dataset"
    if not split or not any(p.stem in split for p in images):
        split = split_by_group([p.stem for p in images])
        how = "nhóm theo tên file (tránh rò rỉ frame cùng video)"
    elif "val" not in split.values():
        split = add_val(split, [p.stem for p in images])
        how += " (val tách 10% từ train)"
    print(f"{len(images)} ảnh, {len(names)} lớp, chia theo {how}")

    crops = defaultdict(lambda: ([], []))
    cls_count: dict[str, Counter] = defaultdict(Counter)
    for p in images:
        sp = split.get(p.stem, "train")
        lab = p.with_suffix(".txt")
        if not lab.exists():
            cand = list(p.parent.parent.glob(f"labels/{p.stem}.txt"))
            lab = cand[0] if cand else lab
        rows = read_labels(lab)
        _link(p, out / "yolo_ncls" / "images" / sp / p.name)
        _link(p, out / "yolo_1cls" / "images" / sp / p.name)
        for d in ("yolo_ncls", "yolo_1cls"):
            (out / d / "labels" / sp).mkdir(parents=True, exist_ok=True)
        (out / "yolo_ncls" / "labels" / sp / f"{p.stem}.txt").write_text(
            "".join(f"{c} {x} {y} {w} {h}\n" for c, x, y, w, h in rows))
        (out / "yolo_1cls" / "labels" / sp / f"{p.stem}.txt").write_text(
            "".join(f"0 {x} {y} {w} {h}\n" for _, x, y, w, h in rows))
        if rows:
            img = cv2.imread(str(p))
            for c, x, y, w, h in rows:
                cr = crop(img, x, y, w, h, size=args.crop_size)
                if cr is not None:
                    crops[sp][0].append(cr)
                    crops[sp][1].append(c)
                    cls_count[sp][names[c]] += 1

    for d, nm in (("yolo_ncls", names), ("yolo_1cls", ["sign"])):
        (out / d / "data.yaml").write_text(
            f"path: {(out / d).resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
            f"names: {json.dumps(dict(enumerate(nm)), ensure_ascii=False)}\n", "utf-8")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    for sp, (xs, ys) in crops.items():
        np.savez_compressed(out / "crops" / f"{sp}.npz", x=np.stack(xs), y=np.array(ys))
    (out / "names.json").write_text(json.dumps(names, ensure_ascii=False), "utf-8")
    stats = {sp: dict(c.most_common()) for sp, c in cls_count.items()}
    (out / "class_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), "utf-8")
    tr = cls_count["train"]
    print("số crop:", {sp: len(v[0]) for sp, v in crops.items()})
    print("lớp nhiều nhất:", tr.most_common(5))
    print("lớp ít nhất:", tr.most_common()[-8:])


if __name__ == "__main__":
    main()
