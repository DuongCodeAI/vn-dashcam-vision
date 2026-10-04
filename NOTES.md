# Ghi chú quyết định

## Chọn dataset
- VNTS (maitam, Kaggle): 3.216 ảnh, 52 lớp, **tên lớp là mã QCVN** (P.130, P.131a, P.127...) → nối thẳng
  sang bảng tra luật của vn-traffic-law-rag. License CC BY-SA 4.0 rõ ràng, có sẵn trên Kaggle (khỏi tải về máy).
- Không dùng Zalo AI 2020: 6GB, chỉ 7 nhóm lớp, license không rõ.
- Mã biển trong QCVN 41:2024 giữ nguyên so với 2019 (đã đối chiếu P.130, P.131a, P.127). Có trang ghi
  "P.130 đổi thành P.131" là sai.

## Chia dữ liệu
- Ảnh dashcam liên tiếp gần như giống hệt nhau. Chia ngẫu nhiên từng ảnh → test "đã thấy" lúc train → mAP đẹp giả.
  Dùng file split của dataset nếu có, không thì chia theo nhóm = tên file bỏ số thứ tự cuối.
- VNTS chỉ có `split_dataset/train_files.txt` (2552) + `test_files.txt` (639), không có val → giữ nguyên test,
  tách 10% train làm val (thiếu val thì ultralytics và train_classifier đều lỗi).

## 2 tầng
- Tầng 1: YOLO11n 1 lớp. Chỉ cần học "có biển ở đây" → dữ liệu của mọi lớp dồn cho một việc.
- Tầng 2: SignNet ~1.2M tham số (4 khối conv kiểu VGG thu nhỏ). Không dùng backbone pretrained để tự kiểm soát
  kiến trúc và giải thích được; ảnh 64x64 đơn giản, không cần ResNet.
- Crop nới rộng 15% mỗi phía, giống hệt lúc tạo dữ liệu train và lúc suy luận (`preprocess.py` dùng chung).

## Lệch lớp
- Effective number (Cui et al. 2019) thay vì 1/n: 1/n làm lớp 5 mẫu nặng gấp 200 lần, train không ổn định.
- Sampler theo tần suất^0.5: cân bằng tuyệt đối thì lớp hiếm bị lặp lại hàng trăm lần mỗi epoch → học thuộc.
- Copy-paste: crop lớp hiếm (chỉ từ train) dán lên ảnh train khác. Lỗi suýt mắc: ghi file nhãn mới chỉ có biển
  vừa dán → các biển có sẵn trong ảnh nền thành "nền" → dạy detector bỏ qua biển thật. Phải nối nhãn cũ.

## Augmentation
- Không lật ngang (đổi nghĩa biển rẽ trái/phải). `fliplr=0` khi train YOLO.
- Nhoè chuyển động (xe máy rung), xoay ±12° (nghiêng khi vào cua), sáng/tối mạnh (đêm, ngược sáng), che một phần.

## Tracker + bỏ phiếu
- Không Kalman: dashcam đi thẳng là chính, biển dịch chậm giữa 2 frame → IoU với vị trí cũ là đủ.
- Ghép 2 lượt kiểu ByteTrack; detection điểm thấp không được mở track mới (tránh track rác).
- Gộp phiếu = trung bình log-xác suất có trọng số theo score detector (= tích xác suất) thay vì đếm phiếu
  argmax: giữ được thông tin "khá chắc" vs "rất chắc".
- Mỗi track chỉ phát sự kiện 1 lần, cần >= 3 frame và độ tin >= 0.6.

## Chạy trên CPU
- Tự viết letterbox/decode/NMS: YOLO11 output (1, 4+nc, N), không có objectness như YOLOv5; NMS gộp mọi lớp
  vì một vật không thể vừa là P.130 vừa là P.131a.
- Int8: static quantization (QDQ, per-channel) với ~200 ảnh calibration. Dynamic quantization chỉ lượng tử
  trọng số nên Conv gần như không nhanh hơn.
- YOLO int8: giữ fp32 phần hậu xử lý của head (DFL, decode hộp, concat output). Output nối toạ độ (0..640)
  với điểm lớp (0..1); quantize chung một scale uint8 thì điểm lớp về 0 → thử với yolo11n ra 0 detection.

## Việc cần làm
- Chạy notebook 01-03, điền bảng kết quả.
- Quay + gán nhãn video xe máy (notebook 04).
- Thử imgsz 960 cho detector 1 lớp nếu recall biển xa thấp.
