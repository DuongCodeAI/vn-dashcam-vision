import numpy as np

from dashcam.tracker import ByteTrackLite
from dashcam.voting import TrackVoter
from dashcam.yolo_onnx import decode, nms, unletterbox


def _yolo_out(dets, nc=3, n=20):
    """dets: list (cx, cy, w, h, cls, score) -> output giả (1, 4+nc, n)."""
    out = np.zeros((1, 4 + nc, n), dtype=np.float32)
    for i, (cx, cy, w, h, c, s) in enumerate(dets):
        out[0, :4, i] = [cx, cy, w, h]
        out[0, 4 + c, i] = s
    return out


def test_decode_and_nms_merges_duplicates():
    out = _yolo_out([(100, 100, 40, 40, 1, 0.9), (102, 101, 40, 40, 2, 0.6), (300, 300, 30, 30, 0, 0.8)])
    boxes, scores, cls = decode(out, conf_thr=0.25, iou_thr=0.5)
    assert len(boxes) == 2
    assert set(cls.tolist()) == {1, 0}
    assert np.allclose(boxes[0], [80, 80, 120, 120])


def test_decode_empty():
    boxes, scores, cls = decode(_yolo_out([]), conf_thr=0.25)
    assert boxes.shape == (0, 4)


def test_nms_keeps_separate():
    b = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    assert sorted(nms(b, np.array([0.5, 0.9]))) == [0, 1]


def test_unletterbox():
    b = np.array([[110.0, 20.0, 210.0, 120.0]])
    out = unletterbox(b, r=0.5, pad=(10, 20), shape=(1000, 1000))
    assert np.allclose(out, [[200, 0, 400, 200]])


def test_tracker_keeps_id_through_low_score_frame():
    tr = ByteTrackLite(high=0.5, low=0.15)
    a = tr.update(np.array([[100, 100, 140, 140.]]), np.array([0.9]))
    tid = a[0][0].id
    # frame bị nhoè: score thấp nhưng vẫn phải nối vào track cũ
    b = tr.update(np.array([[103, 101, 143, 141.]]), np.array([0.25]))
    assert b and b[0][0].id == tid
    c = tr.update(np.array([[106, 102, 146, 142.]]), np.array([0.8]))
    assert c[0][0].id == tid and c[0][0].hits == 3


def test_low_score_does_not_start_track():
    tr = ByteTrackLite()
    assert tr.update(np.array([[0, 0, 10, 10.]]), np.array([0.2])) == []


def test_voter_needs_agreement_across_frames():
    names = ["P.130", "P.131a"]
    v = TrackVoter(names, min_frames=3, min_conf=0.6)
    tr = ByteTrackLite()
    events = []
    # frame 1 nhầm sang P.130, các frame sau đúng P.131a
    seq = [np.array([0.6, 0.4]), np.array([0.2, 0.8]), np.array([0.1, 0.9]), np.array([0.15, 0.85])]
    for f, p in enumerate(seq, 1):
        assigned = tr.update(np.array([[100 + f, 100, 140 + f, 140.]]), np.array([0.9]), [(p, 0.9)])
        events += v.step(assigned, frame=f, fps=10)
    assert len(events) == 1
    assert events[0].code == "P.131a" and events[0].frame == 3
