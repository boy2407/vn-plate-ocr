"""Isolated desktop training entry point for the four-corner plate pose model."""

from pathlib import Path

import torch
import yaml
import ultralytics.utils
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    ultralytics.utils.WEIGHTS_DIR = root  # Use the AMP-check weights transferred from Mac.
    data = root / "pose-073"
    config = yaml.safe_load((data / "dataset.yaml").read_text())
    config["path"] = str(data.resolve())
    (data / "dataset-desktop.yaml").write_text(yaml.safe_dump(config))
    print("VERSIONS", torch.__version__, "CUDA", torch.cuda.is_available(), flush=True)
    model = YOLO(str(root / "yolov8n-pose.pt"), task="pose")
    model.train(
        data=str(data / "dataset-desktop.yaml"),
        epochs=100,
        patience=20,
        imgsz=640,
        batch=32,
        workers=4,
        device=0,
        seed=73,
        project=str(root / "runs"),
        name="plate-pose-073",
        exist_ok=False,
        amp=True,
        plots=True,
        save=True,
        fliplr=0.5,
        flipud=0.0,
    )


if __name__ == "__main__":
    main()
