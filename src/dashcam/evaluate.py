"""Đánh giá trên ảnh tĩnh có nhãn YOLO: so YOLO 1 tầng vs 2 tầng, fp32 vs int8, đo tốc độ.

python -m dashcam.evaluate --images datasets/vnts/yolo_ncls/images/test --labels datasets/vnts/yolo_ncls/labels/test \
    --models models --threads 4 --out results

Tách AP lớp hiếm (ít hơn --rare mẫu train) ra riêng: đó là chỗ 2 tầng được kỳ vọng hơn 1 tầng.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .map_eval import map50
from .pipeline import SignPipeline
from .yolo_onnx import YoloOnnx


def load_gt(img_path: Path, label_dir: Path, shape) -> list:
    H, W = shape[:2]
    p = label_dir / f"{img_path.stem}.txt"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            c, x, y, w, h = ln.split()[:5]
            x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
            out.append((np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]), int(c)))
    return out


def run_system(name, make_pipe, paths, label_dir, warmup: int = 3):
    import cv2

    preds, gts, times = {}, {}, []
    for k, p in enumerate(paths):
        pipe = make_pipe()  # ảnh tĩnh: mỗi ảnh một pipeline mới, không bỏ phiếu qua frame
        img = cv2.imread(str(p))
        t0 = time.perf_counter()
        res = pipe.process(img)
        if k >= warmup:
            times.append(time.perf_counter() - t0)
        preds[p.stem] = [(b, float(s) * float(pr.max()), int(pr.argmax()))
                         for b, s, pr in zip(res.boxes, res.det_scores, res.probs, strict=True) if pr is not None]
        gts[p.stem] = load_gt(p, label_dir, img.shape)
    m, aps = map50(preds, gts)
    return {"system": name, "map50": m, "ap": aps, "ms_per_img": 1000 * float(np.mean(times)) if times else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--class-stats", default=None, help="class_stats.json từ prepare_vnts để tách lớp hiếm")
    ap.add_argument("--rare", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    md = Path(args.models)
    names = json.loads((md / "names.json").read_text("utf-8"))
    imgsz = json.loads((md / "config.json").read_text())["imgsz"] if (md / "config.json").exists() else 640
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths = paths[: args.limit]

    def det(f):
        return YoloOnnx(str(md / f), imgsz=imgsz, conf=0.15, threads=args.threads)

    systems = []
    for suffix in ("", ".int8"):
        if (md / f"detector_ncls{suffix}.onnx").exists():
            systems.append((f"1 tầng YOLO-52{suffix or ' fp32'}",
                            lambda s=suffix: SignPipeline(det(f"detector_ncls{s}.onnx"), names)))
        if (md / f"detector_1cls{suffix}.onnx").exists() and (md / f"signnet{suffix}.onnx").exists():
            systems.append((f"2 tầng YOLO-1 + SignNet{suffix or ' fp32'}",
                            lambda s=suffix: SignPipeline(det(f"detector_1cls{s}.onnx"), names,
                                                          str(md / f"signnet{s}.onnx"), threads=args.threads)))

    rare = set()
    if args.class_stats:
        tr = json.loads(Path(args.class_stats).read_text("utf-8")).get("train", {})
        rare = {i for i, n in enumerate(names) if tr.get(n, 0) < args.rare}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, md_lines = [], ["| hệ thống | mAP@0.5 | mAP@0.5 lớp hiếm | ms/ảnh (CPU) |", "|---|---|---|---|"]
    for name, make in systems:
        r = run_system(name, make, paths, Path(args.labels))
        rare_aps = [v for c, v in r["ap"].items() if c in rare]
        r["map50_rare"] = float(np.mean(rare_aps)) if rare_aps else None
        rows.append(r)
        rare_txt = "-" if r["map50_rare"] is None else f"{r['map50_rare']:.3f}"
        md_lines.append(f"| {name} | {r['map50']:.3f} | {rare_txt} | {r['ms_per_img']:.0f} |")
        print(md_lines[-1])
    for r in rows:
        r["ap"] = {names[c]: round(v, 4) for c, v in r["ap"].items()}
    (out / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), "utf-8")
    (out / "eval.md").write_text("\n".join(md_lines) + "\n", "utf-8")


if __name__ == "__main__":
    main()
