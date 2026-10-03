"""CNN phân loại biển báo, tự thiết kế (không dùng backbone pretrained).

Vì sao tầng 2 riêng thay vì để YOLO đoán luôn 52 lớp?
- YOLO11n phải học đồng thời "ở đâu có biển" và "biển gì" trên ảnh 640px, biển xa chỉ ~15px.
  Crop ra rồi phóng lên 64x64 cho classifier thì chi tiết (gạch chéo P.130 vs P.131a, số trên
  biển tốc độ) rõ hơn nhiều.
- Lớp hiếm (vài chục mẫu) dễ xử lý ở classifier: oversample, loss có trọng số, copy-paste.
- Classifier ~0.6M tham số, chạy ~1ms/crop trên CPU.

Kiến trúc: 4 khối conv (conv-BN-ReLU x2 + maxpool), kênh 32-64-128-256, global avg pool, dropout, FC.
Kiểu VGG thu nhỏ, dễ giải thích, đủ cho ảnh 64x64.
"""

import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SignNet(nn.Module):
    def __init__(self, n_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(3, w), _block(w, 2 * w), _block(2 * w, 4 * w), _block(4 * w, 8 * w))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(8 * w, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def class_weights(counts: list[int], beta: float = 0.999) -> torch.Tensor:
    """Trọng số theo "effective number of samples" (Cui et al., CVPR 2019).

    1/n thì lớp 5 mẫu nặng gấp 200 lần lớp 1000 mẫu -> train rất loạn. Effective number
    bão hoà dần: thêm mẫu cho lớp đã nhiều thì ít thông tin mới.
    """
    c = torch.tensor(counts, dtype=torch.float32).clamp(min=1)
    eff = (1 - beta ** c) / (1 - beta)
    w = 1.0 / eff
    return w / w.sum() * len(counts)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
