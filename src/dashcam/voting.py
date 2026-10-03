"""Bỏ phiếu loại biển báo qua nhiều frame của cùng một track rồi mới phát sự kiện.

Một frame đơn lẻ hay nhầm P.130 (cấm dừng và đỗ) với P.131a (cấm đỗ) vì chỉ khác
một gạch chéo; biển ở xa thì nhầm lung tung. Trong viet-copilot, phát nhầm sự kiện
= trợ lý nói sai luật cho tài xế -> thà chậm 0.3s còn hơn sai.

Cách gộp: trung bình log-xác suất các frame (= tích xác suất, chuẩn hoá theo số frame).
Frame bị nhoè (score detector thấp) đóng góp ít hơn nhờ trọng số = score detector.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SignEvent:
    code: str
    name: str
    conf: float
    track_id: int
    frame: int
    n_frames: int
    box: list[float]
    t_sec: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TrackVoter:
    def __init__(self, class_names: list[str], min_frames: int = 3, min_conf: float = 0.6,
                 display_names: dict[str, str] | None = None):
        self.names = class_names
        self.min_frames = min_frames
        self.min_conf = min_conf
        self.display = display_names or {}
        self.emitted: set[int] = set()

    def decide(self, hist: list[tuple[int, tuple[np.ndarray, float]]]) -> tuple[int, float] | None:
        """hist: [(frame, (probs (C,), det_score)), ...] -> (lớp, độ tin) hoặc None nếu chưa đủ chắc."""
        items = [p for _, p in hist if p is not None]
        if len(items) < self.min_frames:
            return None
        logp = np.stack([np.log(np.clip(pr, 1e-6, 1.0)) for pr, _ in items])
        w = np.array([max(s, 1e-3) for _, s in items])[:, None]
        agg = (logp * w).sum(0) / w.sum()
        probs = np.exp(agg - agg.max())
        probs /= probs.sum()
        c = int(probs.argmax())
        return c, float(probs[c])

    def step(self, assigned, frame: int, fps: float | None = None) -> list[SignEvent]:
        """assigned: list (track, det_idx) từ tracker. Mỗi track chỉ phát sự kiện 1 lần."""
        events = []
        for track, _ in assigned:
            if track.id in self.emitted:
                continue
            res = self.decide(track.cls_hist)
            if res is None or res[1] < self.min_conf:
                continue
            c, conf = res
            code = self.names[c]
            self.emitted.add(track.id)
            events.append(SignEvent(code=code, name=self.display.get(code, code), conf=round(conf, 3),
                                    track_id=track.id, frame=frame, n_frames=len(track.cls_hist),
                                    box=[round(float(v), 1) for v in track.box],
                                    t_sec=round(frame / fps, 2) if fps else None))
        return events
