"""Tách frame từ video quay bằng điện thoại gắn xe máy -> bộ test "đường thật".

python -m dashcam.frames --video data/videos/ngay_noithanh_01.mp4 --out data/frames --every 1.0

- Lấy 1 frame mỗi `--every` giây (frame liên tiếp gần như trùng nhau, gán nhãn thừa).
- Tính độ nét (phương sai Laplacian) cho từng frame: xe máy rung làm nhoè, mình muốn
  biết độ chính xác tụt bao nhiêu theo độ nhoè, nên KHÔNG bỏ frame nhoè mà ghi điểm lại.
- Điều kiện (ngày/đêm/mưa, nội thành/ngoại thành) lấy từ tên file video: <thoigian>_<khuvuc>_<stt>.mp4
- Làm mờ mặt người trước khi lưu (Haar cascade có sẵn trong OpenCV, không cần tải model).
  Biển số xe khác KHÔNG tự làm mờ được -> xem docs/quay_video_xe_may.md, không công khai frame gốc.
"""

import argparse
import csv
from pathlib import Path


def sharpness(gray) -> float:
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def blur_faces(img, cascade):
    import cv2

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    for x, y, w, h in faces:
        roi = img[y:y + h, x:x + w]
        img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), max(w, h) / 6)
    return img, len(faces)


def conditions_from_name(stem: str) -> dict:
    parts = stem.lower().split("_")
    return {"time": parts[0] if parts else "", "area": parts[1] if len(parts) > 1 else ""}


def main():
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, nargs="+")
    ap.add_argument("--out", default="data/frames")
    ap.add_argument("--every", type=float, default=1.0)
    ap.add_argument("--no-face-blur", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    meta_path = out / "frames.csv"
    new = not meta_path.exists()
    with meta_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["file", "video", "t_sec", "sharpness", "time", "area", "faces_blurred"])
        for v in args.video:
            cap = cv2.VideoCapture(v)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            step = max(int(round(fps * args.every)), 1)
            cond = conditions_from_name(Path(v).stem)
            i = saved = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if i % step == 0:
                    s = sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    nf = 0
                    if not args.no_face_blur:
                        frame, nf = blur_faces(frame, cascade)
                    name = f"{Path(v).stem}_{i:06d}.jpg"
                    cv2.imwrite(str(out / "images" / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow([name, Path(v).name, round(i / fps, 2), round(s, 1), cond["time"], cond["area"], nf])
                    saved += 1
                i += 1
            print(f"{v}: {saved} frame")


if __name__ == "__main__":
    main()
