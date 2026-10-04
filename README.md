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

Dataset: [VNTS](https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs) (3.216 ảnh, 52 lớp đặt tên theo mã QCVN, CC BY-SA 4.0).
Giữ nguyên file chia train/test của dataset (2.552 / 639 ảnh), tách 10% train làm val. Test có 1.645 biển.

**Detector** (notebook 01, Colab T4, imgsz 640, mỗi detector giới hạn ~20 phút train):

| detector | epoch | mAP@0.5 test | mAP@0.5:0.95 test |
|---|---|---|---|
| YOLO11n 1 lớp "biển báo" | 19 | 0.986 | 0.752 |
| YOLO11n 52 lớp | 11 | 0.876 | 0.624 |

Hai số này đo hai việc khác nhau (1 lớp chỉ cần tìm đúng chỗ có biển, 52 lớp phải tìm và gọi đúng tên),
nên không so trực tiếp được; so sánh công bằng là bảng 1 tầng vs 2 tầng bên dưới.
Bản 52 lớp chỉ được 11 epoch vì cache ảnh ra đĩa (chậm hơn RAM ~1.5 lần, xem phần "Sự cố").

**1 tầng vs 2 tầng** (notebook 03: 300 ảnh test đầu tiên, mAP@0.5 tính bằng code tự viết trong `map_eval.py`,
ms/ảnh là cả pipeline detect + crop + phân loại, đo bằng onnxruntime trên CPU Colab Xeon 2.0GHz, 2 luồng):

| hệ thống | mAP@0.5 | mAP@0.5 lớp hiếm (<50 mẫu) | ms/ảnh |
|---|---|---|---|
| YOLO11n 52 lớp, fp32 | 0.801 | 0.769 | 133 |
| YOLO11n 52 lớp, int8 | 0.741 | 0.726 | 131 |
| YOLO11n 1 lớp + SignNet, fp32 | **0.962** | 0.950 | 198 |
| YOLO11n 1 lớp + SignNet, int8 | 0.945 | **0.955** | 182 |

- **2 tầng hơn 1 tầng 16 điểm mAP@0.5** (0.962 vs 0.801), lớp hiếm hơn 18 điểm.
- Tốc độ phụ thuộc số luồng nhiều hơn số tầng. Đo lại với **1 luồng** (100 ảnh test đầu):

  | hệ thống | mAP@0.5 | ms/ảnh |
  |---|---|---|
  | 52 lớp fp32 / int8 | 0.838 / 0.819 | 117 / 116 |
  | 1 lớp + SignNet fp32 / int8 | 0.962 / 0.945 | 121 / 116 |

  Với 1 luồng, thêm SignNet chỉ tốn ~4ms. Bản 2 luồng của 2 tầng chậm hẳn (198ms) có lẽ vì YOLO và SignNet là
  2 session onnxruntime, mỗi session 2 luồng, giành nhau 2 vCPU của Colab. Chạy thật nên để 1 luồng/session
  hoặc chạy 2 model song song trên 2 nhân. ~120ms/ảnh ≈ 8 ảnh/giây trên 1 nhân Colab.
- **Int8 trên CPU Colab gần như không nhanh hơn** (131 vs 133ms cho YOLO). Chưa kiểm tra nguyên nhân; nghi CPU
  của runtime này không có lệnh int8 (VNNI) nên onnxruntime không tăng tốc được. Lợi ích chắc chắn là file nhỏ hơn
  ~3 lần (detector 10.6 → 3.2MB, SignNet 4.7 → 1.2MB). Tốc độ trên CPU laptop chưa đo.
- Int8 làm YOLO 52 lớp tụt 6 điểm nhưng 2 tầng chỉ tụt 1.7 điểm (lớp hiếm chênh 0.5 điểm, coi như không đổi).

Số detector ở bảng này là lần train thứ 2 (runtime Colab mới, cùng cấu hình): test mAP@0.5 1 lớp 0.980, 52 lớp 0.875,
gần như trùng lần 1 ở bảng trên (0.986 / 0.876).

**SignNet với các cách xử lý lệch lớp** (notebook 02, 1.645 crop test, 22 lớp hiếm có < 50 crop train;
mỗi cấu hình 40 epoch, ~3 phút trên T4):

| cấu hình | accuracy | macro-F1 | recall lớp hiếm |
|---|---|---|---|
| cross-entropy thường | 0.992 | 0.970 | 0.930 |
| + sampler căn bậc 2 tần suất | 0.991 | **0.985** | 0.967 |
| + sampler + loss trọng số "effective number" | 0.978 | 0.964 | **0.981** |

Accuracy gần như không đổi vì bị lớp đông (P.130 có 765 crop) áp đảo; khác biệt nằm ở macro-F1 và lớp hiếm.
Sampler căn bậc 2 cho macro-F1 cao nhất nên được chọn làm model chính. Thêm loss có trọng số kéo recall lớp hiếm
lên 0.981 nhưng accuracy tụt 1.3 điểm: cộng trọng số lên cả loss lẫn sampler là bù lệch hai lần, model nghiêng
quá tay về lớp hiếm.
Lỗi còn lại chủ yếu là **biển tốc độ** (P.127 40/60/80 nhầm nhau, chỉ khác con số) và W.203c → W.224.
SignNet 1.19M tham số: ONNX fp32 4.7MB, int8 1.2MB.

**Đường thật nhìn từ xe máy**: chưa làm. Notebook 04 cần video tự quay + gán nhãn ~300-500 frame
([hướng dẫn quay](docs/quay_video_xe_may.md)); VNTS là ảnh dashcam ô tô nên đây là chỗ dễ tụt độ chính xác nhất.

### Sự cố khi train (ghi lại để lần sau khỏi mất giờ GPU)

- `cache="ram"` trên Colab free (12.7GB RAM): train xong detector 1 lớp thì **crash hết RAM ở bước validate cuối**,
  vì cache ảnh train (~3.6GB) bị nhân theo số worker của dataloader. Đổi sang `cache="disk"` + `workers=2`
  (Colab chỉ 2 nhân). Đánh đổi: đọc cache từ đĩa chậm hơn ~1.5 lần, nên detector 52 lớp chỉ được 11 epoch trong 18 phút.
- Sau crash, `last.pt` đã bị ultralytics xoá optimizer (train xong rồi). Gọi `resume=True` với checkpoint này **không báo lỗi
  mà âm thầm train mới bằng cấu hình mặc định** (batch 16, lật ngang 0.5...). Notebook giờ kiểm tra `optimizer is None`
  để bỏ qua thay vì resume.

## Điểm đáng chú ý

- **2 tầng thay vì YOLO 52 lớp**: biển ở xa chỉ ~15px trên ảnh 640px; crop rồi phóng lên 64x64 thì chi tiết
  (gạch chéo của P.130 vs P.131a, con số trên biển tốc độ) rõ hơn, và lớp hiếm dễ xử lý ở classifier.
- **Lệch lớp**: P.130 có 765 crop train, 8 lớp dưới 16 crop (ít nhất 2). Thử sampler, loss có trọng số và copy-paste crop lớp hiếm
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
# (04/10/2026: model của lần train này chưa lên HF vì tài khoản Colab dùng để train chưa có secret HF_TOKEN)
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
Chạy trên Colab: mở notebook từ GitHub (File → Open notebook → GitHub → DuongCodeAI/vn-dashcam-vision), chọn T4 GPU
(03 chạy tiếp trên cùng runtime, chỉ dùng CPU). VNTS tải bằng `kagglehub`, không cần tài khoản Kaggle;
Secret `HF_TOKEN` (tuỳ chọn) để đẩy model lên HF Hub của bạn.
Output nằm trong Google Drive (`MyDrive/ai-portfolio/vn-dashcam-vision`) nên chạy lần lượt 01 → 04 là notebook sau tự tìm thấy.
Quay video thử: [docs/quay_video_xe_may.md](docs/quay_video_xe_may.md).

## Hạn chế

- VNTS chủ yếu ảnh ban ngày; đêm/mưa phụ thuộc augmentation.
- imgsz 640 để chạy được trên CPU; biển rất xa sẽ bị bỏ sót (960 tốt hơn nhưng chậm ~2.2 lần).
- Ultralytics (dùng để train) là AGPL-3.0.
