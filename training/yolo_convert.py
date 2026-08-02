from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "yolov8n_roi_320_30.pt"


def main():
    # 权重文件不纳入 Git；训练后将 best.pt 复制为 MODEL_PATH。
    model = YOLO(str(MODEL_PATH))
    model.export(
        format="onnx",
        imgsz=(32, 320),
        batch=1,
        dynamic=False,
        simplify=False,
        opset=11,
        nms=False,
    )


if __name__ == "__main__":
    main()
