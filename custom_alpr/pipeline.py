"""Run locally trained YOLOv8n and ONNX OCR weights."""

# ruff: noqa: PLC0415

from pathlib import Path

import cv2
import numpy as np


class PlateReader:
    """Keep detector and recognizer independent, as in FastALPR."""

    def __init__(self, detector: Path, ocr: Path, config: Path, device: str = "cpu"):
        # Lazy imports allow data preparation without installing ML frameworks.
        from fast_plate_ocr import LicensePlateRecognizer  # pylint: disable=import-outside-toplevel
        from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

        for path in (detector, ocr, config):
            if not path.is_file():
                raise ValueError(f"Missing model/config file: {path}")
        self.detector = YOLO(str(detector), task="detect")
        self.ocr = LicensePlateRecognizer(
            onnx_model_path=str(ocr), plate_config_path=str(config), device="cpu"
        )
        self.device = device

    def predict(self, image: np.ndarray, confidence: float = 0.4) -> list[dict]:
        if image is None or image.size == 0:
            raise ValueError("Expected a nonempty BGR image")
        results = []
        prediction = self.detector.predict(
            source=image, conf=confidence, device=self.device, verbose=False
        )[0]
        if prediction.boxes is None:
            return results
        height, width = image.shape[:2]
        for box in prediction.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            mode = self.ocr.config.image_color_mode
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB if mode == "rgb" else cv2.COLOR_BGR2GRAY)
            text = self.ocr.run_one(crop, return_confidence=True)
            results.append(
                {
                    "box": [x1, y1, x2, y2],
                    "detection_confidence": float(box.conf.item()),
                    "text": text.plate,
                    "character_confidences": (
                        text.char_probs.tolist() if text.char_probs is not None else None
                    ),
                }
            )
        return results


def edit_distance(reference: str, prediction: str) -> int:
    """Levenshtein distance for character error rate, including insertions."""
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for col, actual in enumerate(prediction, 1):
            current.append(
                min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (expected != actual))
            )
        previous = current
    return previous[-1]
