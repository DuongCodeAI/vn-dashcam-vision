# Quay video thử nghiệm trên xe máy

Mục đích: có một bộ test "đường thật" nhìn từ xe máy để đo model train trên dataset công khai tụt bao nhiêu.

## Chuẩn bị
- Giá kẹp điện thoại vào ghi-đông **có đệm chống rung**. Rung động xe máy có thể làm hỏng bộ chống rung
  quang học (OIS) của camera điện thoại.
- Cài đặt camera: 1080p, 30 fps, quay ngang, khoá lấy nét và phơi sáng (giữ ngón tay lên màn hình).
- Bấm quay **trước** khi chạy, không chạm điện thoại khi đang đi (vừa nguy hiểm vừa bị phạt).
  Cách dễ nhất: nhờ người ngồi sau cầm quay, hoặc đứng trên vỉa hè quay các biển báo.

## Nên quay gì (tổng 20-30 phút)
| điều kiện | gợi ý | đặt tên file |
|---|---|---|
| ngày, nội thành | đường nhiều biển cấm dừng/đỗ, cấm rẽ | `ngay_noithanh_01.mp4` |
| ngày, ngoại thành / quốc lộ | biển tốc độ, cảnh báo | `ngay_ngoaithanh_01.mp4` |
| đêm | đèn đường, ngược sáng đèn xe | `dem_noithanh_01.mp4` |
| mưa / sau mưa | nếu an toàn | `mua_noithanh_01.mp4` |

Tên file theo dạng `<thoigian>_<khuvuc>_<stt>.mp4`, script dùng nó để chia nhóm kết quả.

## Tách frame + làm mờ mặt
```bash
python -m dashcam.frames --video data/videos/*.mp4 --out data/frames --every 1.0
```
Ra `data/frames/images/*.jpg` và `data/frames/frames.csv` (thời điểm, độ nét, điều kiện).

## Gán nhãn (~300-500 frame có biển)
- Dùng Label Studio (chạy local) hoặc CVAT, export **YOLO format** với đúng thứ tự lớp trong `names.json`.
- Chỉ gán biển nhìn rõ được bằng mắt; biển quá xa/mờ không đọc nổi thì bỏ qua (ghi lại quy ước này).
- Frame không có biển nào vẫn giữ lại (để đo báo động giả).

## Quyền riêng tư
- Video và frame gốc **giữ riêng tư** (dataset Kaggle để private). Script chỉ tự làm mờ mặt người.
- Biển số xe khác không làm mờ tự động được -> không đăng frame gốc lên GitHub/HF. Ảnh minh hoạ trong README
  phải tự kiểm tra và làm mờ biển số trước.
