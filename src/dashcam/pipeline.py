"""Pipeline camera hành trình: frame -> detector -> (classifier) -> tracker -> bỏ phiếu -> SignEvent.

Hai chế độ để so sánh:
- "two_stage": YOLO 1 lớp tìm biển + SignNet phân loại crop (mặc định)
- "one_stage": YOLO 52 lớp làm cả hai việc (baseline)

Chỉ cần numpy + onnxruntime + opencv. viet-copilot gọi:
    pipe = SignPipeline.load("models/")
    for frame in video: events = pipe.process(frame)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .preprocess import crop_box, normalize_batch
from .tracker import ByteTrackLite
from .voting import SignEvent, TrackVoter
from .yolo_onnx import YoloOnnx


@dataclass
class FrameResult:
    boxes: np.ndarray
    det_scores: np.ndarray
    probs: list  # xác suất lớp cho từng box (hoặc None)
    track_ids: list[int]
    events: list[SignEvent]
    ms: dict = field(default_factory=dict)


class SignPipeline:
    def __init__(self, detector: YoloOnnx, names: list[str], classifier_path: str | None = None,
                 display_names: dict[str, str] | None = None, threads: int | None = None, fps: float | None = None,
                 min_frames: int = 3, min_conf: float = 0.6):
        self.det = detector
        self.names = names
        self.cls_sess = None
        if classifier_path:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            if threads:
                opts.intra_op_num_threads = threads
            self.cls_sess = ort.InferenceSession(classifier_path, opts, providers=["CPUExecutionProvider"])
        self.tracker = ByteTrackLite()
        self.voter = TrackVoter(names, min_frames=min_frames, min_conf=min_conf, display_names=display_names)
        self.fps = fps
        self.frame = 0

    @classmethod
    def load(cls, folder: str | Path, mode: str = "two_stage", threads: int | None = 4, int8: bool = False,
             **kw) -> "SignPipeline":
        folder = Path(folder)
        ext = ".int8.onnx" if int8 else ".onnx"  # int8 nhanh hơn ~17% trên CPU laptop (README)
        names = json.loads((folder / "names.json").read_text("utf-8"))
        display = {}
        if (folder / "display_names.json").exists():
            display = json.loads((folder / "display_names.json").read_text("utf-8"))
        cfg = json.loads((folder / "config.json").read_text("utf-8")) if (folder / "config.json").exists() else {}
        if mode == "two_stage":
            det = YoloOnnx(str(folder / f"detector_1cls{ext}"), imgsz=cfg.get("imgsz", 640), conf=0.15, threads=threads)
            return cls(det, names, str(folder / f"signnet{ext}"), display, threads, **kw)
        det = YoloOnnx(str(folder / f"detector_ncls{ext}"), imgsz=cfg.get("imgsz", 640), conf=0.15, threads=threads)
        return cls(det, names, None, display, threads, **kw)

    def _classify(self, frame: np.ndarray, boxes: np.ndarray) -> list:
        crops, idx = [], []
        for i, b in enumerate(boxes):
            c = crop_box(frame, b)
            if c is not None:
                crops.append(c)
                idx.append(i)
        probs = [None] * len(boxes)
        if crops:
            logits = self.cls_sess.run(None, {self.cls_sess.get_inputs()[0].name: normalize_batch(np.stack(crops))})[0]
            e = np.exp(logits - logits.max(1, keepdims=True))
            p = e / e.sum(1, keepdims=True)
            for k, i in enumerate(idx):
                probs[i] = p[k]
        return probs

    def process(self, frame_bgr: np.ndarray) -> FrameResult:
        self.frame += 1
        t0 = time.perf_counter()
        dets = self.det(frame_bgr)
        t1 = time.perf_counter()
        boxes = np.array([d.box for d in dets]).reshape(-1, 4)
        scores = np.array([d.score for d in dets])
        if self.cls_sess is not None:
            probs = self._classify(frame_bgr, boxes)
        else:  # một tầng: coi lớp của YOLO là chắc chắn theo score
            probs = []
            for d in dets:
                p = np.full(len(self.names), (1 - d.score) / max(len(self.names) - 1, 1))
                p[d.cls] = d.score
                probs.append(p)
        t2 = time.perf_counter()
        payloads = [(p, s) if p is not None else None for p, s in zip(probs, scores, strict=True)]
        assigned = self.tracker.update(boxes, scores, payloads)
        events = self.voter.step(assigned, self.frame, self.fps)
        tids = [-1] * len(boxes)
        for t, di in assigned:
            tids[di] = t.id
        t3 = time.perf_counter()
        return FrameResult(boxes, scores, probs, tids, events,
                           {"det": (t1 - t0) * 1e3, "cls": (t2 - t1) * 1e3, "track": (t3 - t2) * 1e3})
