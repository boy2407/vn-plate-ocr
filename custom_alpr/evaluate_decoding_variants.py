"""Compare decode-time variants of PlateOCR2D without retraining.

Variants: greedy, greedy with plate-format rejection, format-constrained greedy,
beam search, and format-constrained beam search. Allowed plate formats come from
train labels only (digit -> 9, letter -> A), so dev/test labels never leak in.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from custom_alpr.analyze_custom_ocr import (
    CONFIDENCE_THRESHOLDS,
    choose_device,
    edit_distance,
    greedy_with_confidence,
)
from custom_alpr.custom_ocr import (
    BOS_INDEX,
    DECODER_INDEX_TO_CHAR,
    DECODER_VOCAB_SIZE,
    EOS_INDEX,
    PAD_INDEX,
    PlateOCR2D,
    build_model,
    decode_targets,
)
from custom_alpr.custom_ocr_data import PlateOCRDataset, plate_ocr_collate

DEFAULT_DATA = Path("custom_alpr/data/ocr/kaggle-v9-prepared")
REJECTED = -1.0


def plate_format(text: str) -> str:
    """Map digits to 9 and letters to A, e.g. 51D61603 -> 99A99999."""
    return re.sub(r"[^9]", "A", re.sub(r"\d", "9", text))


def load_formats(train_manifest: Path, min_count: int) -> dict[str, int]:
    with train_manifest.open(encoding="utf-8", newline="") as manifest_file:
        counts = Counter(plate_format(row["plate_text"]) for row in csv.DictReader(manifest_file))
    return {fmt: count for fmt, count in counts.most_common() if count >= min_count}


def build_step_masks(formats: set[str], device: torch.device) -> dict[str, torch.Tensor]:
    """For every format prefix, the vocabulary IDs allowed as the next token."""
    digit_ids = [i for i, c in DECODER_INDEX_TO_CHAR.items() if c.isdigit()]
    letter_ids = [i for i, c in DECODER_INDEX_TO_CHAR.items() if not c.isdigit()]
    masks: dict[str, torch.Tensor] = {}
    for fmt in formats:
        for length in range(len(fmt) + 1):
            prefix = fmt[:length]
            mask = masks.setdefault(
                prefix, torch.zeros(DECODER_VOCAB_SIZE, dtype=torch.bool, device=device)
            )
            if length == len(fmt):
                mask[EOS_INDEX] = True
            else:
                mask[digit_ids if fmt[length] == "9" else letter_ids] = True
    return masks


@torch.inference_mode()
def beam_search(  # noqa: PLR0915
    model: PlateOCR2D,
    images: torch.Tensor,
    beam_width: int,
    step_masks: dict[str, torch.Tensor] | None,
) -> tuple[list[str], list[float]]:
    """Beam search ranked by mean log-probability per emitted token (characters + EOS).

    With ``step_masks`` only tokens that keep the prefix inside an allowed format
    are expanded. Scores use the model's unrenormalized probabilities, so the
    returned confidence is the same EOS-excluded geometric mean as greedy decoding.
    """
    device = images.device
    batch_size, width = images.shape[0], beam_width
    rows = batch_size * width
    memory = model.encode(images).repeat_interleave(width, dim=0)
    sequences = torch.full((rows, 1), BOS_INDEX, dtype=torch.long, device=device)
    log_sum = torch.zeros(rows, device=device)
    char_count = torch.zeros(rows, device=device)
    rank_sum = torch.zeros(rows, device=device)
    rank_count = torch.zeros(rows, device=device)
    finished = torch.zeros(rows, dtype=torch.bool, device=device)
    alive = torch.zeros(batch_size, width, dtype=torch.bool, device=device)
    alive[:, 0] = True  # all beams start identical; keep one until the first expansion
    alive = alive.reshape(-1)
    prefixes = [""] * rows
    no_char = torch.zeros(DECODER_VOCAB_SIZE, dtype=torch.bool, device=device)

    for _ in range(model.max_output_length - 1):
        log_probs = model.decode(memory, sequences)[:, -1].log_softmax(dim=-1)
        is_char = torch.arange(DECODER_VOCAB_SIZE, device=device) >= 3
        cand_sum = log_sum[:, None] + torch.where(is_char, log_probs, 0.0)
        cand_count = char_count[:, None] + is_char.float()
        is_emitted = is_char.clone()
        is_emitted[EOS_INDEX] = True
        cand_rank_sum = rank_sum[:, None] + torch.where(is_emitted, log_probs, 0.0)
        cand_rank_count = rank_count[:, None] + is_emitted.float()
        allowed = torch.zeros_like(log_probs, dtype=torch.bool)
        allowed[:, 3:] = True
        allowed[:, EOS_INDEX] = char_count > 0
        if step_masks is not None:
            allowed &= torch.stack([step_masks.get(p, no_char) for p in prefixes])
        allowed[finished] = False
        allowed[finished, PAD_INDEX] = True
        cand_sum = torch.where(finished[:, None], log_sum[:, None], cand_sum)
        cand_count = torch.where(finished[:, None], char_count[:, None], cand_count)
        cand_rank_sum = torch.where(finished[:, None], rank_sum[:, None], cand_rank_sum)
        cand_rank_count = torch.where(finished[:, None], rank_count[:, None], cand_rank_count)
        score = cand_rank_sum / cand_rank_count.clamp_min(1)
        score = score.masked_fill(~allowed | ~alive[:, None], float("-inf"))

        top_scores, top_flat = score.reshape(batch_size, -1).topk(width, dim=-1)
        source = (
            top_flat // DECODER_VOCAB_SIZE
            + torch.arange(batch_size, device=device)[:, None] * width
        ).reshape(-1)
        token = (top_flat % DECODER_VOCAB_SIZE).reshape(-1)
        alive = top_scores.reshape(-1).isfinite()
        log_sum = cand_sum[source, token]
        char_count = cand_count[source, token]
        rank_sum = cand_rank_sum[source, token]
        rank_count = cand_rank_count[source, token]
        was_finished = finished[source]
        finished = was_finished | token.eq(EOS_INDEX) | ~alive
        sequences = torch.cat((sequences[source], token[:, None]), dim=1)
        new_prefixes = []
        for row, (src, tok) in enumerate(zip(source.tolist(), token.tolist(), strict=True)):
            prefix = prefixes[src]
            if tok in DECODER_INDEX_TO_CHAR and not was_finished[row]:
                prefix += "9" if DECODER_INDEX_TO_CHAR[tok].isdigit() else "A"
            new_prefixes.append(prefix)
        prefixes = new_prefixes
        if finished.all():
            break

    final = (rank_sum / rank_count.clamp_min(1)).masked_fill(
        ~alive | char_count.eq(0), float("-inf")
    )
    if step_masks is not None:  # a beam cut off by max length is not a complete format
        complete = torch.tensor(
            [bool(step_masks.get(p, no_char)[EOS_INDEX]) for p in prefixes], device=device
        )
        final = final.masked_fill(~complete, float("-inf"))
    final = final.reshape(batch_size, width)
    best_score, best_beam = final.max(dim=-1)
    best_rows = best_beam + torch.arange(batch_size, device=device) * width
    texts = decode_targets(sequences[best_rows])
    char_mean = (log_sum / char_count.clamp_min(1))[best_rows].exp()
    confidences = [
        float(c) if s.isfinite() else REJECTED
        for s, c in zip(best_score.detach().cpu(), char_mean.detach().cpu(), strict=True)
    ]
    return [
        t if c != REJECTED else "" for t, c in zip(texts, confidences, strict=True)
    ], confidences


def summarize(references: list[str], predictions: list[str], confidences: list[float]) -> dict:
    total = len(references)
    exact = [r == p for r, p in zip(references, predictions, strict=True)]
    errors = sum(edit_distance(r, p) for r, p in zip(references, predictions, strict=True))
    thresholds = []
    for threshold in CONFIDENCE_THRESHOLDS:
        accepted = [ok for ok, c in zip(exact, confidences, strict=True) if c >= threshold]
        thresholds.append(
            {
                "threshold": threshold,
                "coverage": len(accepted) / total,
                "exact_accuracy": sum(accepted) / max(len(accepted), 1),
                "accepted_wrong": len(accepted) - sum(accepted),
            }
        )
    return {
        "exact_plate_accuracy": sum(exact) / total,
        "cer": errors / sum(len(r) for r in references),
        "rejected_by_format": sum(c == REJECTED for c in confidences),
        "confidence_thresholds": thresholds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_DATA / "train.csv")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-format-count", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    formats = load_formats(args.train_manifest, args.min_format_count)
    masks = build_step_masks(set(formats), device)
    dataset = PlateOCRDataset(args.manifest)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=plate_ocr_collate)

    names = ["greedy", "greedy_format_reject", "constrained_greedy", "beam", "constrained_beam"]
    outputs: dict[str, tuple[list[str], list[float]]] = {name: ([], []) for name in names}
    references: list[str] = []
    with torch.inference_mode():
        for batch_images, _, _, texts in loader:
            images = batch_images.to(device)
            references.extend(texts)
            greedy_texts, greedy_conf = greedy_with_confidence(model, images)
            reject_conf = [
                c if plate_format(t) in formats else REJECTED
                for t, c in zip(greedy_texts, greedy_conf, strict=True)
            ]
            results = {
                "greedy": (greedy_texts, greedy_conf),
                "greedy_format_reject": (greedy_texts, reject_conf),
                "constrained_greedy": beam_search(model, images, 1, masks),
                "beam": beam_search(model, images, args.beam_width, None),
                "constrained_beam": beam_search(model, images, args.beam_width, masks),
            }
            for name, (batch_texts, batch_conf) in results.items():
                outputs[name][0].extend(batch_texts)
                outputs[name][1].extend(batch_conf)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest": str(args.manifest),
        "samples": len(references),
        "beam_width": args.beam_width,
        "allowed_formats_from_train": formats,
        "rejected_confidence": (
            "Format-invalid or no complete beam -> confidence -1 (never accepted)."
        ),
        "variants": {name: summarize(references, *outputs[name]) for name in names},
    }
    (args.output_dir / "decoding-variants.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output_dir / "decoding-predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "reference", *(f"{n}_{k}" for n in names for k in ("pred", "conf"))]
        )
        for index, (path, reference) in enumerate(dataset.samples):
            row = [path.name, reference]
            for name in names:
                row += [outputs[name][0][index], f"{outputs[name][1][index]:.6f}"]
            writer.writerow(row)
    for name in names:
        v = report["variants"][name]
        t99 = next(t for t in v["confidence_thresholds"] if t["threshold"] == 0.99)
        print(
            f"{name:22s} exact={v['exact_plate_accuracy']:.4f} cer={v['cer']:.4f} "
            f"rejected={v['rejected_by_format']} @0.99 cov={t99['coverage']:.3f} "
            f"acc={t99['exact_accuracy']:.4f} wrong={t99['accepted_wrong']}"
        )


if __name__ == "__main__":
    main()
