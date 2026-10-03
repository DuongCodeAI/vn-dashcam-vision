"""Xuất model sang ONNX và lượng tử hoá int8.

- SignNet: torch.onnx.export (batch động).
- YOLO: export bằng ultralytics trong notebook (`model.export(format="onnx")`), ở đây chỉ quantize.

Với CNN, quantize "dynamic" (chỉ trọng số) gần như không nhanh hơn vì phần nặng là Conv;
phải quantize "static": chạy vài trăm ảnh thật qua model để đo dải giá trị activation
(calibration) rồi lượng tử hoá cả activation. Đổi lại có thể tụt mAP -> đo lại sau quantize.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def export_signnet(ckpt: str, out: str, opset: int = 17):
    import torch

    from .classifier import SignNet

    st = torch.load(ckpt, map_location="cpu")
    m = SignNet(len(st["names"]), st.get("width", 32))
    m.load_state_dict(st["model"])
    m.eval()
    torch.onnx.export(m, (torch.zeros(1, 3, 64, 64),), out, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "n"}, "logits": {0: "n"}}, opset_version=opset, dynamo=False)
    return st["names"]


class _ImageReader:
    """CalibrationDataReader cho onnxruntime: đưa từng ảnh (đã tiền xử lý) vào."""

    def __init__(self, input_name: str, arrays: list[np.ndarray]):
        self.name = input_name
        self.it = iter(arrays)

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {self.name: x}


def calib_yolo_inputs(image_dir: str, imgsz: int, n: int = 200) -> list[np.ndarray]:
    import cv2

    from .yolo_onnx import letterbox

    paths = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n]
    out = []
    for p in paths:
        lb, _, _ = letterbox(cv2.imread(str(p)), imgsz)
        out.append(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0))
    return out


def calib_crop_inputs(npz: str, n: int = 500) -> list[np.ndarray]:
    from .preprocess import normalize_batch

    x = np.load(npz)["x"][:n]
    return [normalize_batch(x[i:i + 1]) for i in range(len(x))]


def quantize_static_int8(fp32: str, out: str, calib: list[np.ndarray]):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = str(Path(out).with_suffix(".pre.onnx"))
    quant_pre_process(fp32, pre)
    name = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(pre, out, _ImageReader(name, calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, per_channel=True)
    Path(pre).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signnet", default=None, help="runs/cls/best.pt")
    ap.add_argument("--yolo", action="append", default=[], help="đường dẫn yolo .onnx (fp32) cần quantize")
    ap.add_argument("--calib-images", default=None)
    ap.add_argument("--calib-crops", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.signnet:
        names = export_signnet(args.signnet, str(out / "signnet.onnx"))
        (out / "names.json").write_text(json.dumps(names, ensure_ascii=False), "utf-8")
        if args.calib_crops:
            quantize_static_int8(str(out / "signnet.onnx"), str(out / "signnet.int8.onnx"),
                                 calib_crop_inputs(args.calib_crops))
    for y in args.yolo:
        q = str(out / (Path(y).stem + ".int8.onnx"))
        quantize_static_int8(y, q, calib_yolo_inputs(args.calib_images, args.imgsz))
    (out / "config.json").write_text(json.dumps({"imgsz": args.imgsz}), "utf-8")
    for p in sorted(out.glob("*.onnx")):
        print(p.name, f"{p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
