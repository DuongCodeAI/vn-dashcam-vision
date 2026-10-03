"""Dashboard Streamlit: tải video dashcam -> video có khung biển báo + dòng thời gian sự kiện.

streamlit run src/dashcam/app.py

Nếu cài thêm vn-traffic-law-rag thì mỗi sự kiện biển báo kèm luôn điều khoản xử phạt
(đây là đoạn sẽ dùng trong viet-copilot).
"""

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from dashcam.pipeline import SignPipeline

st.set_page_config(page_title="Dashcam biển báo", layout="wide")
st.title("Nhận diện biển báo từ camera hành trình")

models = st.sidebar.text_input("Thư mục model", "models")
mode = st.sidebar.selectbox("Chế độ", ["two_stage", "one_stage"])
every = st.sidebar.slider("Xử lý 1 trên N frame", 1, 5, 2)
vehicle = st.sidebar.selectbox("Loại xe (để tra luật)", ["xe máy", "ô tô"])
up = st.file_uploader("Video (.mp4)", type=["mp4", "mov", "avi"])


@st.cache_resource
def law():
    try:
        from vnlaw_rag.pipeline import LawRAG

        return LawRAG.from_dir(st.secrets.get("VNLAW_DATA", "../vn-traffic-law-rag/data"))
    except Exception:  # noqa: BLE001 - không có thì bỏ qua phần tra luật
        return None


def draw(frame, res, names):
    for b, pr, tid in zip(res.boxes, res.probs, res.track_ids, strict=True):
        x1, y1, x2, y2 = map(int, b)
        label = f"#{tid} {names[int(pr.argmax())]} {pr.max():.2f}" if pr is not None else f"#{tid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return frame


if up:
    tmp = Path(tempfile.mkdtemp()) / up.name
    tmp.write_bytes(up.read())
    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pipe = SignPipeline.load(models, mode=mode, fps=fps / every)
    view, side = st.columns([3, 2])
    img_slot = view.empty()
    stat = view.empty()
    side.subheader("Sự kiện")
    i, ms = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % every:
            continue
        t0 = time.perf_counter()
        res = pipe.process(frame)
        ms.append((time.perf_counter() - t0) * 1000)
        img_slot.image(draw(frame, res, pipe.names)[:, :, ::-1], use_container_width=True)
        stat.caption(f"frame {i} · {np.mean(ms[-30:]):.0f} ms/frame · det {res.ms['det']:.0f} cls {res.ms['cls']:.0f}")
        for ev in res.events:
            box = side.container(border=True)
            box.markdown(f"**{ev.code}** {ev.name} · {i / fps:.1f}s · tin cậy {ev.conf:.0%} ({ev.n_frames} frame)")
            rag = law()
            if rag is not None:
                info = rag.lookup_sign(ev.code, vehicle=vehicle, k=1)
                if info and info.hits:
                    box.caption(info.hits[0].citation)
                    box.write(info.hits[0].chunk["text"].split("\n", 1)[1][:300])
