import numpy as np
import pytest

from dashcam.map_eval import average_precision, map50
from dashcam.prepare_vnts import group_key, split_by_group


def test_map_perfect_and_miss():
    b = np.array([0, 0, 10, 10.0])
    gts = {"a": [(b, 0)], "b": [(b, 1)]}
    preds = {"a": [(b, 0.9, 0)], "b": [(b + 50, 0.9, 1)]}  # lớp 1 đoán lệch chỗ
    m, aps = map50(preds, gts)
    assert aps[0] == pytest.approx(1.0)
    assert aps[1] == 0.0
    assert m == pytest.approx(0.5)


def test_ap_duplicate_detection_counts_as_fp():
    b = np.array([0, 0, 10, 10.0])
    m, aps = map50({"a": [(b, 0.9, 0), (b, 0.8, 0)]}, {"a": [(b, 0)]})
    assert aps[0] == pytest.approx(1.0)  # FP xếp sau TP không làm giảm AP
    assert average_precision(np.array([0.5, 1.0]), np.array([1.0, 0.5])) == pytest.approx(0.75)


def test_group_split_keeps_video_together():
    stems = [f"video12_{i:04d}" for i in range(50)] + [f"clipB-{i}" for i in range(50)]
    sp = split_by_group(stems)
    assert len({sp[s] for s in stems if s.startswith("video12")}) == 1
    assert group_key("video12_0007") == "video12"


def test_signnet_forward_and_weights():
    torch = pytest.importorskip("torch")
    from dashcam.classifier import SignNet, class_weights, count_params

    m = SignNet(52)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 52)
    assert count_params(m) < 1_500_000
    w = class_weights([1000, 10])
    assert w[1] > w[0]
    assert float(w[1] / w[0]) < 100  # không nổ như 1/n


def test_normalize_batch_shape():
    from dashcam.preprocess import normalize_batch

    x = normalize_batch(np.zeros((3, 64, 64, 3), np.uint8))
    assert x.shape == (3, 3, 64, 64) and x.dtype == np.float32
