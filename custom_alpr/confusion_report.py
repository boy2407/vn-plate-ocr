"""Which characters the OCR confuses, at which crop heights, with and without layout decoding.

For every model x manifest it reads each crop twice: free greedy decoding and
``layout_decode`` (digits and letters fixed by the plate layout). It reports exact
plate accuracy per source-height bucket and the most frequent character confusions,
split into digit<->digit, letter<->letter and digit<->letter.

    python -m custom_alpr.confusion_report \
        --model E=custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt \
        --manifest kaggle-test=custom_alpr/data/ocr/kaggle-v9-prepared/test.csv
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader

from custom_alpr.custom_ocr import build_model
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

HEIGHT_BUCKETS = ((0, 16), (16, 24), (24, 32), (32, 10_000))  # source crop height, px


def named_paths(values: list[str]) -> dict[str, Path]:
    pairs = (value.split("=", 1) for value in values)
    return {name: Path(path) for name, path in pairs}


def substitutions(reference: str, prediction: str) -> list[tuple[str, str]]:
    """Aligned (true, read) character pairs that differ; insertions/deletions are skipped."""
    pairs = []
    matcher = difflib.SequenceMatcher(a=reference, b=prediction, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "replace" and a1 - a0 == b1 - b0:
            pairs += [(t, p) for t, p in zip(reference[a0:a1], prediction[b0:b1], strict=True)]
    return pairs


def pair_kind(true: str, read: str) -> str:
    if true.isdigit() and read.isdigit():
        return "digit-digit"
    if not true.isdigit() and not read.isdigit():
        return "letter-letter"
    return "digit-letter"


def bucket(height: int) -> str:
    low, high = next((low, high) for low, high in HEIGHT_BUCKETS if low <= height < high)
    return f"{low}-{high}px" if high < 10_000 else f">={low}px"


def summarize(rows: list[dict]) -> dict:
    """Accuracy per height bucket and confusion counts for one decoding."""
    by_bucket: dict[str, list[bool]] = {}
    pairs, kinds = Counter(), Counter()
    for row in rows:
        by_bucket.setdefault(bucket(row["height"]), []).append(row["reference"] == row["prediction"])
        for true, read in substitutions(row["reference"], row["prediction"]):
            pairs[f"{true}->{read}"] += 1
            kinds[pair_kind(true, read)] += 1
    return dict(
        exact=round(sum(r["reference"] == r["prediction"] for r in rows) / len(rows), 4),
        crops=len(rows),
        by_height={
            name: dict(crops=len(hits), exact=round(sum(hits) / len(hits), 4))
            for name in (bucket(low) for low, _ in HEIGHT_BUCKETS)
            if (hits := by_bucket.get(name))
        },
        confusion_kinds=dict(kinds),
        top_confusions=pairs.most_common(15),
    )


@torch.inference_mode()
def read_manifest(model, manifest: Path, batch_size: int) -> list[dict]:
    dataset = PlateOCRDataset(manifest)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=plate_ocr_collate)
    rows, index = [], 0
    for images, _, _, references in loader:
        free = model.generate(images)
        fitted = model.generate(images, layout=True)
        for reference, plain, layout in zip(references, free, fitted, strict=True):
            path = dataset.samples[index][0]
            with Image.open(path) as image:
                height = image.height
            rows.append(dict(image=str(path), height=height, reference=reference, free=plain, layout=layout))
            index += 1
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", required=True, help="name=checkpoint.pt")
    parser.add_argument("--manifest", action="append", required=True, help="name=manifest.csv")
    parser.add_argument("--output", type=Path, default=Path("custom_alpr/outputs/confusion-report"))
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    for model_name, checkpoint_path in named_paths(args.model).items():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = build_model(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        for manifest_name, manifest in named_paths(args.manifest).items():
            rows = read_manifest(model, manifest, args.batch_size)
            name = f"{model_name}/{manifest_name}"
            report[name] = {
                decoding: summarize([dict(row, prediction=row[decoding]) for row in rows])
                for decoding in ("free", "layout")
            }
            (args.output / f"{model_name}-{manifest_name}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
            )
            for decoding, summary in report[name].items():
                print(f"{name:28s} {decoding:6s} exact={summary['exact']:.3f} n={summary['crops']} "
                      f"kinds={summary['confusion_kinds']}")
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
