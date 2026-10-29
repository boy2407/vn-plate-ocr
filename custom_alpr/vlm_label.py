"""Label queued plate crops with a local vision-language model (Qwen2.5-VL).

A second, independent labeller next to Codex: rows where both agree can be
accepted, rows where they differ go to a person. The model never sees OCR
output, only the crop, enlarged like the ``view/`` copies Codex reads.

    python -m custom_alpr.vlm_label --rows custom_alpr/data/collected/queue.csv --limit 150
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

from custom_alpr.plate_tracks import normalized_text, valid_plate

ROOT = Path(__file__).resolve().parents[1]
COLLECTED = ROOT / "custom_alpr/data/collected"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VIEW_HEIGHT = 192  # Same enlargement as the view/ copies given to Codex.
VERDICTS = {"label", "unreadable", "partial", "not_plate"}
FIELDS = ["crop_id", "verdict", "plate_text", "lines", "seconds", "raw"]
PROMPT = """This image is a crop from a traffic video that may contain a Vietnamese license plate.
Answer with one JSON object and nothing else:
{"verdict": "...", "plate_text": "...", "lines": 1 or 2 or null}

verdict:
- "label": it is a plate and you can read EVERY character with certainty.
- "unreadable": it is a plate but at least one character is uncertain (blur, glare, too small).
- "partial": it is a plate but part of it is cut off by the image border.
- "not_plate": not a license plate (logo, sign, text on a vehicle, clothing, building).
If unsure about any character, use "unreadable". Never guess.

plate_text (only for "label", else ""): uppercase letters and digits only, drop "-", "." and spaces,
keep "Đ". For a two-line plate read the top line then the bottom line.
Examples: "51G-123.45" -> "51G12345"; top "29-B1" bottom "123.45" -> "29B112345".
Format: 2 digits (province), 1-2 letters, optional digit, then 4-5 digits.
lines: number of text lines on the plate (1 or 2), null for not_plate."""


def parse_answer(raw: str) -> dict:
    """Verdict, plate text and lines from the model's reply; malformed replies are unreadable."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        answer = {}
    verdict = str(answer.get("verdict", "")).strip().lower()
    text = normalized_text(str(answer.get("plate_text") or ""))
    lines = answer.get("lines")
    lines = str(lines) if lines in (1, 2, "1", "2") else ""
    if verdict not in VERDICTS:
        return dict(verdict="unreadable", plate_text="", lines="")
    # A plate text outside the VN format is a misread, not a label.
    if verdict == "label" and not valid_plate(text):
        return dict(verdict="unreadable", plate_text="", lines=lines)
    return dict(verdict=verdict, plate_text=text if verdict == "label" else "", lines=lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, default=COLLECTED / "queue.csv", help="CSV with crop_id,image_path")
    parser.add_argument("--collected", type=Path, default=COLLECTED, help="Base folder of image_path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=COLLECTED / "vlm-labels.csv")
    return parser.parse_args()


def main() -> None:
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    args = parse_args()
    with args.rows.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]
    done = set()
    if args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as handle:
            done = {row["crop_id"] for row in csv.DictReader(handle)}
    rows = [row for row in rows if row["crop_id"] not in done]

    started = time.perf_counter()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map="cuda",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"model loaded in {time.perf_counter() - started:.0f}s; {len(rows)} crops to label")

    new = not args.output.exists()
    with args.output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        for index, row in enumerate(rows, 1):
            tick = time.perf_counter()
            image = Image.open(args.collected / row["image_path"]).convert("RGB")
            scale = VIEW_HEIGHT / image.height
            image = image.resize((max(1, round(image.width * scale)), VIEW_HEIGHT), Image.Resampling.BICUBIC)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            raw = processor.batch_decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]
            seconds = time.perf_counter() - tick
            writer.writerow(dict(crop_id=row["crop_id"], seconds=round(seconds, 2), raw=raw.strip(), **parse_answer(raw)))
            handle.flush()
            if index % 10 == 0 or index == len(rows):
                print(f"{index}/{len(rows)}  last {seconds:.2f}s")


if __name__ == "__main__":
    main()
