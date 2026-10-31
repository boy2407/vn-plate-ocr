"""Compare single-crop, string, and per-character voting on labeled tracks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_alpr.consensus import decide, read_rows
from custom_alpr.custom_ocr import build_model, layout_decode
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate
from custom_alpr.plate_tracks import normalized_text, per_character_vote, valid_plate
from custom_alpr.reject_eval import E, camera_index, wilson

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "custom_alpr/data/collected/queue.csv"
VLM = ROOT / "custom_alpr/data/collected/vlm-labels.csv"


def group_tracks(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("track_id"):
            groups[(row.get("camera", ""), row["track_id"])].append(row)
    usable, conflicts = [], []
    for (camera, track_id), observations in sorted(groups.items()):
        truths = {normalized_text(row["reference"]) for row in observations}
        if len(truths) != 1:
            conflicts.append(
                {
                    "camera": camera,
                    "track_id": track_id,
                    "labels": sorted(truths),
                    "crops": len(observations),
                }
            )
            continue
        truth = next(iter(truths))
        best = max(
            observations,
            key=lambda row: (
                0.65 * row["quality"] + 0.35 * row["geometric_mean"],
                row["geometric_mean"],
                row["height"],
            ),
        )
        weighted: dict[str, float] = defaultdict(float)
        counts: Counter = Counter()
        for row in observations:
            text = normalized_text(row["prediction"])
            weighted[text] += row["quality"] * row["geometric_mean"]
            counts[text] += 1
        string_text = max(
            weighted, key=lambda text: (valid_plate(text), weighted[text], counts[text])
        )
        char_observations = [
            {
                "text": row["prediction"],
                "quality": row["quality"],
                "char_probabilities": row["char_probabilities"],
            }
            for row in observations
        ]
        char = per_character_vote(char_observations)
        usable.append(
            {
                "camera": camera,
                "track_id": track_id,
                "truth": truth,
                "crops": len(observations),
                "single": best["prediction"],
                "string": string_text,
                "character": char["candidate_text"],
                "character_confidence": char["confidence"],
            }
        )
    return usable, conflicts


def metric(rows: list[dict], field: str, threshold: float | None = None) -> dict:
    accepted = [
        row for row in rows if threshold is None or row["character_confidence"] >= threshold
    ]
    correct = sum(row[field] == row["truth"] for row in accepted)
    return {
        "tracks": len(rows),
        "accepted": len(accepted),
        "correct": correct,
        "exact_accuracy_all_tracks": correct / len(rows) if rows else None,
        "exact_accuracy_ci_all_tracks": wilson(correct, len(rows)),
        "accepted_precision": correct / len(accepted) if accepted else None,
        "precision_ci": wilson(correct, len(accepted)),
        "coverage": len(accepted) / len(rows) if rows else 0,
        "coverage_ci": wilson(len(accepted), len(rows)),
    }


def consensus_manifest(output: Path) -> tuple[Path, dict[str, dict]]:
    collected = ROOT / "custom_alpr/data/collected"
    vlm_by_id = {row["crop_id"]: row for row in read_rows(VLM)}
    metadata = camera_index()
    agreed, joined = [], {}
    for row in read_rows(QUEUE):
        if decide(row, vlm_by_id.get(row["crop_id"])) != "agree":
            continue
        image = collected / row["image_path"]
        if not image.is_file():
            continue
        agreed.append(
            {
                "image_path": str(image.resolve()),
                "plate_text": normalized_text(row["plate_text"]),
                "source": "youtube",
            }
        )
        joined[image.stem] = metadata.get(row["crop_id"], {})
    path = output / "consensus-agreed-249.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_path", "plate_text", "source"))
        writer.writeheader()
        writer.writerows(agreed)
    return path, joined


@torch.inference_mode()
def infer_consensus(manifest: Path, joined: dict[str, dict], checkpoint_path: Path) -> list[dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = PlateOCRDataset(manifest)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=plate_ocr_collate)
    rows, index = [], 0
    for images, _, _, references in loader:
        free, confidence, probabilities = model.generate_with_confidence(
            images, return_positions=True
        )
        fitted, _ = layout_decode(probabilities)
        for local, (reference, prediction, geometric, layout) in enumerate(
            zip(references, free, confidence, fitted, strict=True)
        ):
            path, _ = dataset.samples[index]
            meta = joined.get(path.stem, {})
            p = probabilities[local].cpu()
            char_probs = [p[pos, 3:].tolist() for pos in range(len(prediction))]
            rows.append(
                {
                    "camera": meta.get("camera", "consensus"),
                    "track_id": meta.get("track_id", ""),
                    "reference": reference,
                    "prediction": prediction,
                    "geometric_mean": geometric,
                    "quality": float(meta.get("quality") or 1),
                    "height": 0,
                    "char_probabilities": char_probs,
                    "layout_prediction": layout,
                }
            )
            index += 1
    return rows


def evaluate(rows: list[dict], thresholds: dict | None = None) -> dict:
    tracks, found_conflicts = group_tracks(rows)
    result = {
        "input_crops": len(rows),
        "track_count": len(tracks),
        "conflicting_tracks": found_conflicts,
        "single_best_crop": metric(tracks, "single"),
        "current_string_vote": metric(tracks, "string"),
        "per_character_vote": metric(tracks, "character"),
    }
    if thresholds:
        result["per_character_vote_plus_rejection"] = {
            key: metric(tracks, "character", value)
            for key, value in thresholds.items()
            if value is not None
        }
        result["thresholds"] = thresholds
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-a", type=Path, default=ROOT / "custom_alpr/outputs/reject-eval-076/report.json"
    )
    parser.add_argument(
        "--primary-rows",
        type=Path,
        default=ROOT / "custom_alpr/outputs/reject-eval-076/E-confirmed.json",
    )
    parser.add_argument("--score", default="weakest_position_margin")
    parser.add_argument("--output", type=Path, default=ROOT / "custom_alpr/outputs/track-vote-076")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    primary_rows = json.loads(args.primary_rows.read_text(encoding="utf-8"))
    report_a = json.loads(args.phase_a.read_text(encoding="utf-8"))["E"]
    track_rows, conflicts = group_tracks(primary_rows)
    primary_result = {
        "input_crops": len(primary_rows),
        "track_count": len(track_rows),
        "conflicting_tracks": conflicts,
        "single_best_crop": metric(track_rows, "single"),
        "current_string_vote": metric(track_rows, "string"),
        "per_character_vote": metric(track_rows, "character"),
    }
    # Report both precision targets, with threshold selected on the opposite camera.
    primary_result["per_character_vote_plus_rejection"] = {}
    for target in ("0.98", "0.99"):
        for direction, details in report_a[args.score][target].items():
            heldout = direction.split("_to_")[1]
            threshold = details["calibration_threshold"]
            if threshold is not None:
                subset = [row for row in track_rows if row["camera"] == heldout]
                primary_result["per_character_vote_plus_rejection"][f"{direction}@{target}"] = (
                    metric(subset, "character", threshold) | {"threshold": threshold}
                )
    primary_result["model"] = "E"
    primary_result["selection_score"] = args.score
    output_report = {"human_confirmed": primary_result}

    manifest, joined = consensus_manifest(args.output)
    secondary_rows = infer_consensus(manifest, joined, E)
    secondary = evaluate(secondary_rows)
    secondary_tracks, _ = group_tracks(secondary_rows)
    secondary["per_character_vote_plus_rejection"] = {}
    for target in ("0.98", "0.99"):
        for direction, details in report_a[args.score][target].items():
            heldout = direction.split("_to_")[1]
            threshold = details["calibration_threshold"]
            if threshold is not None:
                subset = [row for row in secondary_tracks if row["camera"] == heldout]
                secondary["per_character_vote_plus_rejection"][f"{direction}@{target}"] = metric(
                    subset, "character", threshold
                ) | {"threshold": threshold}
    secondary["source"] = "Codex+VLM agreements; secondary evidence"
    output_report["codex_vlm_agreed"] = secondary
    (args.output / "report.json").write_text(
        json.dumps(output_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "primary-track-rows.json").write_text(
        json.dumps(track_rows, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "primary-conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
