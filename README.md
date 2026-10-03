# vn-dashcam-vision

Nhận diện **biển báo giao thông Việt Nam** từ camera hành trình, chạy trên CPU laptop (ONNX, không cần torch).
Mỗi biển chỉ được "báo" một lần, sau khi nhiều frame liên tiếp đồng ý với nhau, để dùng cho trợ lý lái xe
[viet-copilot](https://github.com/DuongCodeAI/viet-copilot): thấy biển "Cấm đỗ xe" thì tra luật ngay
([vn-traffic-law-rag](https://github.com/DuongCodeAI/vn-traffic-law-rag)).

```
video ─► YOLO11n (1 lớp "biển báo") ─► crop 64x64 ─► SignNet (CNN tự thiết kế, 52 lớp QCVN 41)
      ─► tracker kiểu ByteTrack ─► bỏ phiếu nhiều frame ─► SignEvent{code: "P.131a", conf, track_id}
```

## Kết quả

> Chưa chạy. Các bảng dưới sẽ được điền bằng số thật sau khi chạy notebook trên Kaggle.

Dataset: [VNTS](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) (3.216 ảnh, 52 lớp đặt tên theo mã QCVN, CC BY-SA 4.0),
chia train/val/test theo nhóm video để không rò rỉ frame gần giống nhau.

**1 tầng vs 2 tầng** (test VNTS, mAP@0.5, tốc độ trên CPU 4 luồng):

| hệ thống | mAP@0.5 | mAP@0.5 lớp hiếm (<50 mẫu) | ms/ảnh |
|---|---|---|---|
| YOLO11n 52 lớp, fp32 | | | |
| YOLO11n 52 lớp, int8 | | | |
| YOLO11n 1 lớp + SignNet, fp32 | | | |
| YOLO11n 1 lớp + SignNet, int8 | | | |

**SignNet với các cách xử lý lệch lớp** (crop test):

| cấu hình | accuracy | macro-F1 | recall lớp hiếm |
|---|---|---|---|
| cross-entropy thường | | | |
| + sampler căn bậc 2 tần suất | | | |
| + sampler + loss trọng số "effective number" | | | |

**Đường thật nhìn từ xe máy** (video tự quay, ~300-500 frame có nhãn):

| nhóm | 1 tầng | 2 tầng |
|---|---|---|
| tất cả | | |
| ngày / đêm | | |
| frame nét / nhoè | | |

## Điểm đáng chú ý

- **2 tầng thay vì YOLO 52 lớp**: biển ở xa chỉ ~15px trên ảnh 640px; crop rồi phóng lên 64x64 thì chi tiết
  (gạch chéo của P.130 vs P.131a, con số trên biển tốc độ) rõ hơn, và lớp hiếm dễ xử lý ở classifier.
- **Lệch lớp**: P.130 hơn 1.000 mẫu, có lớp vài chục. Thử sampler, loss có trọng số và copy-paste crop lớp hiếm
  lên ảnh nền khác (giữ nguyên nhãn gốc của ảnh nền).
- **Không lật ngang ảnh** khi augment: "cấm rẽ trái" lật thành "cấm rẽ phải".
- **Bỏ phiếu nhiều frame**: trung bình log-xác suất có trọng số theo điểm detector; frame nhoè ít tiếng nói hơn.
  Phát nhầm sự kiện = trợ lý nói sai luật cho tài xế, nên thà chậm vài trăm ms.
- **Tracker giữ detection điểm thấp** (ý tưởng ByteTrack): biển bị nhoè do xe rung có score ~0.2,
  bỏ đi là track đứt và phiếu bầu bị chia nhỏ.
- **Suy luận tự viết bằng numpy + onnxruntime** (letterbox, decode YOLO11, NMS): không cần ultralytics/torch lúc chạy.
- **Int8 bằng static quantization có calibration**: với CNN, quantize dynamic gần như không nhanh hơn.

## Chạy

```bash
pip install -e ".[infer]"
# tải models/ từ HF Hub: detector_1cls(.int8).onnx, signnet(.int8).onnx, names.json
pip install -e ".[app]" && streamlit run src/dashcam/app.py
```

```python
from dashcam.pipeline import SignPipeline
pipe = SignPipeline.load("models", fps=15)
for frame in frames:
    for ev in pipe.process(frame).events:
        print(ev.code, ev.conf, ev.t_sec)
```

Train lại: `notebooks/01` (detector, T4) → `02` (SignNet) → `03` (quantize + so sánh, CPU) → `04` (video xe máy).
Quay video thử: [docs/quay_video_xe_may.md](docs/quay_video_xe_may.md).

## Hạn chế

- VNTS chủ yếu ảnh ban ngày; đêm/mưa phụ thuộc augmentation.
- imgsz 640 để chạy được trên CPU; biển rất xa sẽ bị bỏ sót (960 tốt hơn nhưng chậm ~2.2 lần).
- Ultralytics (dùng để train) là AGPL-3.0.
