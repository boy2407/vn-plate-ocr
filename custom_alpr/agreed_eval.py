"""Score OCR checkpoints on the Codex+VLM agreed video crops, split by what E-v1 trained on.

The 249 agreed crops share no image with any training manifest, but the same plates
appear in other frames. ``unseen`` keeps only plates whose text is absent from the E-v1
training manifest, so every model is read on plates it never saw.

    python -m custom_alpr.agreed_eval --output custom_alpr/outputs/agreed-eval-076
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from custom_alpr.phase_c_eval import bootstrap_metrics, load_recognizer, read_manifest, summarize
from custom_alpr.reject_eval import wilson

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "custom_alpr/outputs/train-custom-ocr"
AGREED = ROOT / "custom_alpr/outputs/track-vote-076/consensus-agreed-249.csv"
E_V1_TRAIN = ROOT / "custom_alpr/data/ocr/finetune-video-v1/all/train.csv"
MODELS = {
    "E": RUNS / "run-20261029-09-parseq-f-compress/best.pt",
    "E-loose": RUNS / "run-20261030-11-parseq-f-confusable-loose/best.pt",
    "d0": RUNS / "run-phased-d0/best.pt",
    "d05": RUNS / "run-phased-d05/best.pt",
    "d10": RUNS / "run-phased-d10/best.pt",
    "E-v1": RUNS / "run-20261030-10-parseq-e-v1-video/best.pt",
}


def seen_texts() -> set[str]:
    with E_V1_TRAIN.open(encoding="utf-8", newline="") as handle:
        return {row["plate_text"] for row in csv.DictReader(handle)}


def score(rows: list[dict]) -> dict:
    stats = summarize(rows)
    correct = sum(row["correct"] for row in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "exact_ci_wilson95": wilson(correct, len(rows)),
        "cer": stats["cer"],
        "cer_ci_bootstrap95": stats["cer_ci_bootstrap95"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    seen = seen_texts()
    rows_by_model = {}
    for name, checkpoint in MODELS.items():
        print("READ", name, flush=True)
        rows_by_model[name] = read_manifest(
            load_recognizer(checkpoint), AGREED, "agreed", args.batch_size
        )
    base = rows_by_model["E-loose"]
    subsets = {
        "all": lambda _row: True,
        "unseen_by_e_v1": lambda row: row["reference"] not in seen,
        "seen_by_e_v1": lambda row: row["reference"] in seen,
    }
    report = {"plates_seen_by_e_v1": len({r["reference"] for r in base if r["reference"] in seen})}
    report["plates_total"] = len({r["reference"] for r in base})
    report["models"], report["paired_vs_E_loose"] = {}, {}
    for name, rows in rows_by_model.items():
        report["models"][name] = {}
        for subset, keep in subsets.items():
            groups = {"all": [r for r in rows if keep(r)]}
            for camera in sorted({r["camera"] for r in rows}):
                groups[camera] = [r for r in rows if keep(r) and r["camera"] == camera]
            report["models"][name][subset] = {g: score(v) for g, v in groups.items() if v}
        if name != "E-loose":
            report["paired_vs_E_loose"][name] = {
                subset: bootstrap_metrics(
                    [r for r in base if keep(r)], [r for r in rows if keep(r)]
                )
                for subset, keep in subsets.items()
            }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "rows.json").write_text(
        json.dumps(rows_by_model, ensure_ascii=False), encoding="utf-8"
    )
    print("wrote", args.output / "report.json")


if __name__ == "__main__":
    main()
