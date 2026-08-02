from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "data_set" / "data.yaml"


if __name__ == "__main__":
    model = YOLO(r"yolov8n.pt")
    model.train(
        # 640x60 ROI images + rect=True produce [B, 3, 32, 320].
        data = str(DATA_CONFIG),
        epochs = 100,
        imgsz = 320,
        rect = True,
        multi_scale = 0.0,
        mosaic = 0.0,
        batch = -1,
        cache = "ram",
        workers = 1,
    )