"""Export the demo's YOLO detector and pose model to OpenVINO for CPU inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def openvino_dir(weights: Path) -> Path:
    """Folder Ultralytics writes for ``weights``: ``best.pt`` -> ``best_openvino_model``."""
    return weights.with_name(f"{weights.stem}_openvino_model")


def prefer_openvino(weights: Path | None) -> Path | None:
    """The OpenVINO export next to a ``.pt`` when it exists, else the ``.pt`` itself."""
    if weights is None or Path(weights).suffix != ".pt":
        return weights
    exported = openvino_dir(Path(weights))
    return exported if exported.is_dir() else weights


def export(weights: Path, task: str) -> Path:
    # Dynamic input keeps the 320/640 setting and the 16:9 letterbox (384x640 instead of 640x640).
    YOLO(str(weights), task=task).export(format="openvino", imgsz=640, dynamic=True)
    return openvino_dir(weights)


def main():
    from custom_alpr.video_demo import DEFAULT_DETECTOR  # noqa: PLC0415
    from custom_alpr.web_app import DEFAULT_POSE  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    args = parser.parse_args()
    print(export(args.detector, "detect"))
    print(export(args.pose, "pose"))


if __name__ == "__main__":
    main()
