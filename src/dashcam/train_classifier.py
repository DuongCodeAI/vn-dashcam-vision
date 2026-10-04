"""Train SignNet trên crop 64x64 (Kaggle T4 vài phút, CPU laptop cũng chạy được nhưng chậm).

python -m dashcam.train_classifier --crops datasets/vnts/crops --out runs/cls --epochs 40

Xử lý lệch lớp (P.130 hơn 1000 mẫu, có lớp chỉ vài chục):
- sampler lấy mẫu theo tần suất^0.5 (không cân bằng tuyệt đối, tránh học thuộc lớp hiếm)
- cross-entropy có trọng số "effective number" + label smoothing 0.05
- augmentation mạnh (xem augment.py), không lật ngang
So sánh 3 cấu hình trong notebook: CE thường / + sampler / + sampler + trọng số.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .augment import random_crop_aug
from .classifier import SignNet, class_weights, count_params
from .preprocess import normalize_batch


def to_tensor(batch_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(normalize_batch(batch_hwc))


class CropDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, train: bool, seed: int = 0):
        self.x, self.y, self.train = x, y, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = self.x[i]
        if self.train:
            img = random_crop_aug(img, self.rng)
        return img, int(self.y[i])


def seed_worker(_):
    # worker được fork từ tiến trình chính nên mang y nguyên self.rng -> mọi worker, mọi epoch lặp lại
    # đúng một chuỗi phép augment. torch đổi seed worker mỗi epoch -> lấy seed đó tạo rng mới.
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**32)


def collate(batch):
    imgs = np.stack([b[0] for b in batch])
    return to_tensor(imgs), torch.tensor([b[1] for b in batch])


@torch.no_grad()
def predict(model, x: np.ndarray, device, bs: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(F.softmax(model(to_tensor(x[i:i + bs]).to(device)), -1).cpu().numpy())
    model.train()
    return np.concatenate(out)


def metrics(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    pred = probs.argmax(1)
    acc = float((pred == y).mean())
    f1s, recalls = [], {}
    for c in range(n_classes):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        recalls[c] = r
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"acc": acc, "macro_f1": float(np.mean(f1s)), "per_class_recall": recalls}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="datasets/vnts/crops")
    ap.add_argument("--names", default="datasets/vnts/names.json")
    ap.add_argument("--out", default="runs/cls")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--sampler", choices=["none", "sqrt"], default="sqrt")
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-min", type=float, default=None, help="giới hạn cứng thời gian train (phút)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = json.loads(Path(args.names).read_text("utf-8"))
    nc = len(names)
    tr, va = np.load(Path(args.crops) / "train.npz"), np.load(Path(args.crops) / "val.npz")
    xtr, ytr, xva, yva = tr["x"], tr["y"], va["x"], va["y"]
    counts = np.bincount(ytr, minlength=nc)
    print(f"train {len(ytr)} val {len(yva)}; lớp ít nhất {counts[counts > 0].min()} mẫu, nhiều nhất {counts.max()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignNet(nc, args.width).to(device)
    print(f"SignNet {count_params(model) / 1e6:.2f}M tham số")

    ds = CropDataset(xtr, ytr, train=True)
    sampler = None
    if args.sampler == "sqrt":
        w = 1.0 / np.sqrt(np.maximum(counts[ytr], 1))
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ytr), replacement=True)
    dl = DataLoader(ds, batch_size=args.bs, sampler=sampler, shuffle=sampler is None, collate_fn=collate,
                    num_workers=args.workers, drop_last=True, worker_init_fn=seed_worker)
    cw = class_weights(counts.tolist()).to(device) if args.weighted_loss else None
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(dl))

    best, log = -1.0, []
    t0 = time.time()
    for ep in range(args.epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb, weight=cw, label_smoothing=0.05)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        m = metrics(predict(model, xva, device), yva, nc)
        log.append({"epoch": ep + 1, "loss": round(loss.item(), 4), "acc": m["acc"], "macro_f1": m["macro_f1"],
                    "min": round((time.time() - t0) / 60, 1)})
        print(log[-1])
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"model": model.state_dict(), "names": names, "width": args.width}, out / "best.pt")
        if args.max_min and time.time() - t0 > args.max_min * 60:
            print(f"hết {args.max_min} phút, dừng ở epoch {ep + 1}")
            break
    (out / "log.json").write_text(json.dumps(log, indent=1), "utf-8")
    print("best macro_f1", round(best, 4))


if __name__ == "__main__":
    main()
