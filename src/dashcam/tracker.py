"""Tracker kiểu ByteTrack rút gọn (không Kalman).

Ý tưởng chính của ByteTrack: đừng vứt detection điểm thấp. Biển báo ở xa / bị nhoè
do xe rung chỉ có score ~0.2-0.3; nếu lọc bỏ ngay thì track bị đứt rồi tạo ID mới,
và bước bỏ phiếu nhiều frame mất tác dụng. Nên ghép 2 lượt:
  1) detection điểm cao  <-> tất cả track (IoU)
  2) detection điểm thấp <-> các track chưa được ghép ở lượt 1
Detection điểm thấp KHÔNG được tạo track mới.

Không có Kalman vì camera hành trình đi thẳng là chính, biển báo dịch chuyển chậm
giữa 2 frame liên tiếp ở 15-30 fps -> IoU với vị trí cũ là đủ.
"""

from dataclasses import dataclass, field

import numpy as np

from .yolo_onnx import iou_one_to_many


@dataclass
class Track:
    id: int
    box: np.ndarray
    score: float
    cls_hist: list = field(default_factory=list)  # lịch sử (frame, payload) để bỏ phiếu
    hits: int = 1
    age: int = 0  # số frame liên tiếp không thấy
    start_frame: int = 0


def _greedy_match(tracks: list[Track], boxes: np.ndarray, thr: float):
    """-> (cặp (track, det), track chưa ghép, det chưa ghép). Greedy theo IoU giảm dần, đủ tốt khi ít vật thể."""
    if not tracks or len(boxes) == 0:
        return [], list(range(len(tracks))), list(range(len(boxes)))
    iou = np.stack([iou_one_to_many(t.box, boxes) for t in tracks])  # (T, D)
    pairs = []
    used_t, used_d = set(), set()
    for flat in np.argsort(-iou, axis=None):
        ti, di = divmod(int(flat), iou.shape[1])
        if iou[ti, di] < thr:
            break
        if ti in used_t or di in used_d:
            continue
        pairs.append((ti, di))
        used_t.add(ti)
        used_d.add(di)
    return pairs, [i for i in range(len(tracks)) if i not in used_t], [j for j in range(len(boxes)) if j not in used_d]


class ByteTrackLite:
    def __init__(self, high: float = 0.5, low: float = 0.15, iou_thr: float = 0.3, max_age: int = 15):
        self.high, self.low, self.iou_thr, self.max_age = high, low, iou_thr, max_age
        self.tracks: list[Track] = []
        self._next = 1
        self.frame = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray, payloads: list | None = None) -> list[tuple[Track, int]]:
        """Trả về list (track, chỉ số detection) cho các detection đã gán vào track ở frame này."""
        self.frame += 1
        payloads = payloads if payloads is not None else [None] * len(boxes)
        hi = np.where(scores >= self.high)[0]
        lo = np.where((scores >= self.low) & (scores < self.high))[0]
        out: list[tuple[Track, int]] = []

        pairs, un_t, un_d = _greedy_match(self.tracks, boxes[hi] if len(hi) else np.zeros((0, 4)), self.iou_thr)
        for ti, k in pairs:
            out.append(self._assign(self.tracks[ti], hi[k], boxes, scores, payloads))
        remain = [self.tracks[i] for i in un_t]
        pairs2, un_t2, _ = _greedy_match(remain, boxes[lo] if len(lo) else np.zeros((0, 4)), self.iou_thr)
        for ti, k in pairs2:
            out.append(self._assign(remain[ti], lo[k], boxes, scores, payloads))
        for i in un_t2:
            remain[i].age += 1
        for k in un_d:  # chỉ detection điểm cao mới mở track mới
            d = hi[k]
            t = Track(self._next, boxes[d].copy(), float(scores[d]), start_frame=self.frame)
            t.cls_hist.append((self.frame, payloads[d]))
            self._next += 1
            self.tracks.append(t)
            out.append((t, int(d)))
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return out

    def _assign(self, t: Track, d: int, boxes, scores, payloads) -> tuple[Track, int]:
        t.box = boxes[d].copy()
        t.score = float(scores[d])
        t.hits += 1
        t.age = 0
        t.cls_hist.append((self.frame, payloads[d]))
        return t, int(d)
