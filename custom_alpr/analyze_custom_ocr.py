"""Analyze character errors, confidence thresholds, and inference speed for PlateOCR2D."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_alpr.custom_ocr import (
    ALPHABET,
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    PlateOCR2D,
    PlateOCRParseq,
    build_model,
    decode_targets,
)
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

DEFAULT_DATA = Path("custom_alpr/data/ocr/kaggle-v9-prepared")
INSERTION = "<INSERTION>"
DELETION = "<DELETION>"
CONFIDENCE_THRESHOLDS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DATA / "test.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout-labels", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")
    for path in (args.checkpoint, args.manifest):
        if not path.is_file():
            parser.error(f"File does not exist: {path}")
    if args.layout_labels is not None and not args.layout_labels.is_file():
        parser.error(f"Layout label file does not exist: {args.layout_labels}")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def greedy_with_confidence(
    model: PlateOCR2D, images: torch.Tensor
) -> tuple[list[str], list[float]]:
    """Decode greedily and score each plate by geometric mean character-token probability."""
    if isinstance(model, PlateOCRParseq):
        return model.generate_with_confidence(images)
    memory = model.encode(images)
    batch_size = images.shape[0]
    generated = torch.full((batch_size, 1), BOS_INDEX, dtype=torch.long, device=images.device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
    log_probability_sum = torch.zeros(batch_size, dtype=torch.float32, device=images.device)
    character_counts = torch.zeros(batch_size, dtype=torch.long, device=images.device)

    for _ in range(model.max_output_length - 1):
        logits = model.decode(memory, generated)[:, -1]
        probabilities = logits.softmax(dim=-1)
        selected_probabilities, next_ids = probabilities.max(dim=-1)
        active_characters = (~finished) & (next_ids >= 3)
        log_probability_sum += torch.where(
            active_characters, selected_probabilities.clamp_min(1e-12).log(), 0.0
        )
        character_counts += active_characters.long()
        next_ids = torch.where(finished, torch.full_like(next_ids, PAD_INDEX), next_ids)
        generated = torch.cat((generated, next_ids[:, None]), dim=1)
        finished |= next_ids.eq(EOS_INDEX)
        if finished.all():
            break

    scores = torch.where(
        character_counts > 0,
        (log_probability_sum / character_counts.clamp_min(1)).exp(),
        0.0,
    )
    return decode_targets(generated), scores.detach().cpu().tolist()


def align_characters(reference: str, prediction: str) -> list[tuple[str, str]]:
    """Levenshtein alignment with deterministic diagonal, deletion, insertion tie order."""
    rows, columns = len(reference), len(prediction)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    traces = [[""] * (columns + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        costs[row][0] = row
        traces[row][0] = "delete"
    for column in range(1, columns + 1):
        costs[0][column] = column
        traces[0][column] = "insert"

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            diagonal = costs[row - 1][column - 1] + (reference[row - 1] != prediction[column - 1])
            deletion = costs[row - 1][column] + 1
            insertion = costs[row][column - 1] + 1
            cost, _, operation = min(
                (diagonal, 0, "diagonal"),
                (deletion, 1, "delete"),
                (insertion, 2, "insert"),
            )
            costs[row][column] = cost
            traces[row][column] = operation

    aligned = []
    row, column = rows, columns
    while row or column:
        operation = traces[row][column]
        if operation == "diagonal":
            aligned.append((reference[row - 1], prediction[column - 1]))
            row -= 1
            column -= 1
        elif operation == "delete":
            aligned.append((reference[row - 1], DELETION))
            row -= 1
        else:
            aligned.append((INSERTION, prediction[column - 1]))
            column -= 1
    aligned.reverse()
    return aligned


def edit_distance(reference: str, prediction: str) -> int:
    return sum(expected != actual for expected, actual in align_characters(reference, prediction))


def load_layout_labels(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as label_file:
        reader = csv.DictReader(label_file)
        if not {"image", "layout"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: expected image,layout columns")
        labels = {}
        for line, row in enumerate(reader, 2):
            layout = (row.get("layout") or "").strip().lower()
            if layout not in {"single", "double"}:
                raise ValueError(f"{path}:{line}: layout must be single or double")
            labels[Path(row["image"]).name] = layout
        return labels


def layout_metrics(records: list[dict], labels: dict[str, str]) -> dict:
    if not labels:
        return {
            "status": "unavailable",
            "reason": (
                "The Kaggle Type field is not a reliable one-row/two-row label. Supply a "
                "CSV with image,layout columns (single or double) to compute this breakdown."
            ),
        }
    groups = {layout: [] for layout in ("single", "double")}
    for record in records:
        if record["image"] in labels:
            groups[labels[record["image"]]].append(record)
    result = {
        "status": "computed",
        "unlabeled_images": len(records) - sum(map(len, groups.values())),
    }
    for layout, items in groups.items():
        result[layout] = {
            "samples": len(items),
            "exact_accuracy": sum(item["reference"] == item["prediction"] for item in items)
            / max(len(items), 1),
            "mean_confidence": sum(item["confidence"] for item in items) / max(len(items), 1),
        }
    return result


def main() -> None:  # noqa: PLR0912, PLR0915
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dataset = PlateOCRDataset(args.manifest)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=plate_ocr_collate,
    )
    records = []
    cpu_batches = []
    with torch.inference_mode():
        for batch_images, _, _, references in loader:
            cpu_batches.append(batch_images)
            predictions, confidences = greedy_with_confidence(model, batch_images.to(device))
            start = len(records)
            for offset, (reference, prediction, confidence) in enumerate(
                zip(references, predictions, confidences, strict=True)
            ):
                image = dataset.samples[start + offset][0].name
                records.append(
                    {
                        "image": image,
                        "reference": reference,
                        "prediction": prediction,
                        "confidence": confidence,
                        "exact": reference == prediction,
                    }
                )

    alphabet = list(ALPHABET)
    reference_labels = [*alphabet, INSERTION]
    prediction_labels = [*alphabet, DELETION]
    confusion = Counter()
    substitutions = insertions = deletions = 0
    for record in records:
        for expected, actual in align_characters(record["reference"], record["prediction"]):
            confusion[(expected, actual)] += 1
            if expected == INSERTION:
                insertions += 1
            elif actual == DELETION:
                deletions += 1
            elif expected != actual:
                substitutions += 1

    threshold_metrics = []
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = [record for record in records if record["confidence"] >= threshold]
        threshold_metrics.append(
            {
                "threshold": threshold,
                "accepted_plates": len(accepted),
                "coverage": len(accepted) / max(len(records), 1),
                "exact_plates": sum(record["exact"] for record in accepted),
                "exact_accuracy": sum(record["exact"] for record in accepted)
                / max(len(accepted), 1),
            }
        )

    synchronize(device)
    warmup_count = min(8, len(cpu_batches))
    with torch.inference_mode():
        for batch in cpu_batches[:warmup_count]:
            model.generate(batch[:1].to(device))
    synchronize(device)

    single_images = [
        batch[index : index + 1] for batch in cpu_batches for index in range(batch.shape[0])
    ]
    latency_samples = single_images[: min(100, len(single_images))]
    latencies_ms = []
    with torch.inference_mode():
        for batch in latency_samples:
            synchronize(device)
            started = time.perf_counter()
            model.generate(batch.to(device))
            synchronize(device)
            latencies_ms.append((time.perf_counter() - started) * 1000)

    synchronize(device)
    throughput_started = time.perf_counter()
    with torch.inference_mode():
        for batch in cpu_batches:
            model.generate(batch.to(device))
    synchronize(device)
    throughput_seconds = time.perf_counter() - throughput_started

    sorted_latencies = sorted(latencies_ms)
    speed = {
        "device": str(device),
        "batch_1_samples": len(latency_samples),
        "batch_1_latency_ms_mean": statistics.mean(latencies_ms),
        "batch_1_latency_ms_p50": statistics.median(latencies_ms),
        "batch_1_latency_ms_p95": sorted_latencies[max(0, int(0.95 * len(sorted_latencies)) - 1)],
        "batch_size": args.batch_size,
        "batched_samples": len(records),
        "batched_elapsed_seconds": throughput_seconds,
        "batched_throughput_plates_per_second": len(records) / max(throughput_seconds, 1e-12),
        "timing_scope": (
            "Includes tensor transfer and autoregressive model decoding; excludes image file "
            "decode/resize."
        ),
    }
    labels = load_layout_labels(args.layout_labels)
    report = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "test_samples": len(records),
        "exact_plate_accuracy": sum(record["exact"] for record in records) / max(len(records), 1),
        "mean_confidence_definition": (
            "Geometric mean probability of selected output character tokens; EOS excluded."
        ),
        "confidence_thresholds": threshold_metrics,
        "character_alignment": (
            "Levenshtein minimum-edit alignment; ties prefer diagonal, then deletion, then "
            "insertion."
        ),
        "character_substitutions": substitutions,
        "character_insertions": insertions,
        "character_deletions": deletions,
        "layout_metrics": layout_metrics(records, labels),
        "speed": speed,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "extended-evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "character-confusion-matrix.csv").open(
        "w", encoding="utf-8", newline=""
    ) as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["reference\\prediction", *prediction_labels])
        for expected in reference_labels:
            writer.writerow(
                [expected, *(confusion[(expected, actual)] for actual in prediction_labels)]
            )
    with (args.output_dir / "confidence-predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as prediction_file:
        writer = csv.writer(prediction_file)
        writer.writerow(("image", "reference", "prediction", "confidence", "exact_match"))
        writer.writerows(
            (
                item["image"],
                item["reference"],
                item["prediction"],
                item["confidence"],
                item["exact"],
            )
            for item in records
        )
    if args.layout_labels is None:
        with (args.output_dir / "layout-label-template.csv").open(
            "w", encoding="utf-8", newline=""
        ) as template_file:
            writer = csv.writer(template_file)
            writer.writerow(("image", "layout"))
            writer.writerows((item["image"], "") for item in records)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
