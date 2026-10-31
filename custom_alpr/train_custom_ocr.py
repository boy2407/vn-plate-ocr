"""Train and evaluate the project-owned PlateOCR2D model."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, get_worker_info

from custom_alpr.custom_ocr import (
    DECODER_INDEX_TO_CHAR,
    EOS_INDEX,
    PAD_INDEX,
    PlateOCR2D,
    PlateOCRParseq,
    permutation_attention_mask,
    sequence_loss,
)
from custom_alpr.custom_ocr_data import (
    PlateOCRDataset,
    RepeatFactorSampler,
    plate_ocr_collate,
    plate_ocr_distill_collate,
    rare_letter_repeat_factors,
)

DEFAULT_DATA = Path("custom_alpr/data/ocr/kaggle-v9-prepared")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DEFAULT_DATA / "train.csv")
    parser.add_argument("--val", type=Path, default=DEFAULT_DATA / "val.csv")
    parser.add_argument("--test", type=Path, default=DEFAULT_DATA / "test.csv")
    parser.add_argument("--output", type=Path, default=Path("custom_alpr/outputs/train-custom-ocr"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps, or cuda[:index]")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-output-length", type=int, default=14)
    parser.add_argument("--decoder", choices=("ar", "parseq"), default="ar")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--noise-augment",
        action="store_true",
        help="also add Gaussian and salt-and-pepper noise to training crops",
    )
    parser.add_argument(
        "--blue-plate-augment",
        action="store_true",
        help="also recolor some crops as white text on a blue plate",
    )
    parser.add_argument(
        "--rotate-augment",
        action="store_true",
        help="also apply random rotation and perspective tilt",
    )
    parser.add_argument(
        "--view-tilt",
        action="store_true",
        help="rotate up to 12 deg and tilt the view 10-25%% (replaces --rotate-augment)",
    )
    parser.add_argument(
        "--strong-downscale",
        action="store_true",
        help="shrink half of the crops to 8-24 px high (default: 30%% to 12-32 px)",
    )
    parser.add_argument(
        "--compression-augment",
        action="store_true",
        help="also pass small copies through JPEG (p=0.3) or real H.264 with motion (p=0.15)",
    )
    parser.add_argument(
        "--balance-rare-letters",
        action="store_true",
        help="repeat plates with rare letters within each epoch (every plate still once)",
    )
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="fine-tune: start from this checkpoint's weights (same decoder and model config)",
    )
    parser.add_argument(
        "--select-by",
        choices=("val_loss", "greedy_exact"),
        default="val_loss",
        help="validation metric for checkpoint selection, LR plateau, and early stopping",
    )
    parser.add_argument(
        "--distill-teacher",
        type=Path,
        help="frozen teacher checkpoint reading the clean image; the student reads the augmented "
        "one and its encoder features are pulled toward the teacher's (parseq only)",
    )
    parser.add_argument(
        "--distill-weight", type=float, default=0.0, help="weight of the encoder-feature MSE"
    )
    parser.add_argument(
        "--distill-kl", type=float, default=0.0, help="weight of the output KL (left-to-right)"
    )
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--max-test-batches", type=int)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1:
        parser.error(
            "epochs, batch-size, and patience must be positive; workers cannot be negative"
        )
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning-rate must be positive and weight-decay cannot be negative")
    if not 0 <= args.label_smoothing < 1 or args.warmup_epochs < 0:
        parser.error("label-smoothing must be in [0, 1) and warmup-epochs cannot be negative")
    if args.distill_weight < 0 or args.distill_kl < 0 or args.distill_temperature <= 0:
        parser.error("distill weights must be non-negative and the temperature positive")
    if (args.distill_weight or args.distill_kl) and not args.distill_teacher:
        parser.error("--distill-weight/--distill-kl need --distill-teacher")
    if args.distill_teacher and args.decoder != "parseq":
        parser.error("distillation is implemented for the parseq decoder only")
    if args.max_output_length < 2:
        parser.error("max-output-length must allow at least BOS and EOS")
    if any(
        limit is not None and limit < 1
        for limit in (args.max_train_batches, args.max_val_batches, args.max_test_batches)
    ):
        parser.error("batch limits must be positive when provided")
    return args


def distillation_terms(
    student: PlateOCRParseq,
    teacher: PlateOCRParseq,
    student_memory: torch.Tensor,
    decoder_inputs: torch.Tensor,
    target_ids: torch.Tensor,
    clean: torch.Tensor,
    has_clean: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Feature MSE and output KL against a frozen teacher; rows without a clean view are skipped."""
    zero = student_memory.new_zeros(())
    if not has_clean.any():
        return zero, zero
    with torch.no_grad():
        teacher_memory = teacher.encode(clean)
    mse = torch.nn.functional.mse_loss(student_memory[has_clean], teacher_memory[has_clean])
    order = torch.arange(decoder_inputs.shape[1], device=decoder_inputs.device)
    mask = permutation_attention_mask(order)
    student_logits = student.decode_positions(student_memory, decoder_inputs, mask)
    with torch.no_grad():
        teacher_logits = teacher.decode_positions(teacher_memory, decoder_inputs, mask)
    valid = target_ids[:, 1:].ne(PAD_INDEX) & has_clean[:, None]
    kl = torch.nn.functional.kl_div(
        (student_logits[valid] / temperature).log_softmax(dim=-1),
        (teacher_logits[valid] / temperature).softmax(dim=-1),
        reduction="batchmean",
    ) * temperature**2
    return mse, kl


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


def initialize_worker(_: int) -> None:
    worker = get_worker_info()
    if worker is not None and hasattr(worker.dataset, "rng"):
        worker.dataset.rng = np.random.default_rng(torch.initial_seed() % (2**32))


def decode_prediction(token_ids: list[int]) -> str:
    characters = []
    for token_id in token_ids:
        if token_id == EOS_INDEX:
            break
        if token_id in DECODER_INDEX_TO_CHAR:
            characters.append(DECODER_INDEX_TO_CHAR[token_id])
    return "".join(characters)


def edit_distance(reference: str, prediction: str) -> int:
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def evaluate_teacher_forced(
    model: PlateOCR2D,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    token_count = 0
    correct_tokens = 0
    exact_plates = 0
    character_errors = 0
    character_count = 0
    sample_count = 0
    with torch.inference_mode():
        for batch_index, (batch_images, batch_inputs, batch_targets, references) in enumerate(
            loader
        ):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch_images.to(device)
            decoder_inputs = batch_inputs.to(device)
            target_ids = batch_targets.to(device)
            logits = model(images, decoder_inputs)
            loss = sequence_loss(logits, target_ids)
            expected = target_ids[:, 1:]
            valid = expected.ne(PAD_INDEX)
            predictions = logits.argmax(dim=-1)
            valid_count = int(valid.sum().item())
            loss_total += float(loss.item()) * valid_count
            token_count += valid_count
            correct_tokens += int(((predictions == expected) & valid).sum().item())
            for prediction_ids, reference in zip(predictions.tolist(), references, strict=True):
                prediction = decode_prediction(prediction_ids)
                exact_plates += prediction == reference
                character_errors += edit_distance(reference, prediction)
                character_count += len(reference)
                sample_count += 1
    return {
        "loss": loss_total / max(token_count, 1),
        "token_accuracy": correct_tokens / max(token_count, 1),
        "teacher_forced_exact_accuracy": exact_plates / max(sample_count, 1),
        "teacher_forced_cer": character_errors / max(character_count, 1),
    }


def evaluate_greedy(
    model: PlateOCR2D,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Measure inference behavior by feeding each generated prefix back to the decoder."""
    model.eval()
    exact_plates = 0
    character_errors = 0
    character_count = 0
    sample_count = 0
    with torch.inference_mode():
        for batch_index, (images, _, _, references) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            predictions = model.generate(images.to(device))
            for prediction, reference in zip(predictions, references, strict=True):
                exact_plates += prediction == reference
                character_errors += edit_distance(reference, prediction)
                character_count += len(reference)
                sample_count += 1
    return {
        "exact_accuracy": exact_plates / max(sample_count, 1),
        "cer": character_errors / max(character_count, 1),
    }


def main() -> None:  # noqa: PLR0912, PLR0915
    args = parse_args()
    for path in (args.train, args.val, args.test):
        if not path.is_file():
            raise SystemExit(f"Missing manifest: {path}")
    if args.output.exists():
        raise SystemExit(f"Output already exists; choose a new path: {args.output}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    print(f"Training PlateOCR2D ({args.decoder} decoder) on {device}")

    train_dataset = PlateOCRDataset(
        args.train,
        augment=True,
        noise=args.noise_augment,
        blue_plate=args.blue_plate_augment,
        rotate=args.rotate_augment,
        strong_downscale=args.strong_downscale,
        view_tilt=args.view_tilt,
        compression=args.compression_augment,
        return_clean=args.distill_teacher is not None,
        seed=args.seed,
    )
    val_dataset = PlateOCRDataset(args.val, seed=args.seed)
    test_dataset = PlateOCRDataset(args.test, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "collate_fn": plate_ocr_collate,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": initialize_worker,
    }
    train_options = dict(loader_options)
    if args.distill_teacher:
        train_options["collate_fn"] = plate_ocr_distill_collate
    if args.balance_rare_letters:
        factors, rare = rare_letter_repeat_factors([text for _, text in train_dataset.samples])
        sampler = RepeatFactorSampler(factors, generator=generator)
        print(
            f"Rare-letter repeats: letters {''.join(sorted(rare))}; "
            f"{len(sampler)} crops per epoch instead of {len(factors)}"
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_options)
    else:
        train_loader = DataLoader(
            train_dataset, shuffle=True, generator=generator, **train_options
        )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model_class = PlateOCRParseq if args.decoder == "parseq" else PlateOCR2D
    model = model_class(max_output_length=args.max_output_length)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        if initial.get("decoder", "ar") != args.decoder:
            raise SystemExit(f"{args.init_checkpoint} uses the {initial.get('decoder')} decoder")
        model.load_state_dict(initial["model_state_dict"])
        print(f"Initialized from {args.init_checkpoint} (epoch {initial.get('epoch')})")
    model = model.to(device)
    teacher = None
    if args.distill_teacher:
        teacher = PlateOCRParseq(max_output_length=args.max_output_length)
        teacher_state = torch.load(args.distill_teacher, map_location="cpu", weights_only=True)
        teacher.load_state_dict(teacher_state["model_state_dict"])
        teacher = teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(
            f"Distilling from {args.distill_teacher}: mse={args.distill_weight} "
            f"kl={args.distill_kl} T={args.distill_temperature}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if args.select_by == "val_loss" else "max", factor=0.5, patience=3
    )
    warmup_steps = round(args.warmup_epochs * len(train_loader))
    step = 0
    args.output.mkdir(parents=True)
    checkpoint_path = args.output / "best.pt"
    history_path = args.output / "history.csv"
    history_fields = (
        "epoch",
        "train_loss",
        "val_loss",
        "val_token_accuracy",
        "val_teacher_forced_exact_accuracy",
        "val_teacher_forced_cer",
        "val_greedy_exact_accuracy",
        "val_greedy_cer",
        "learning_rate",
    )
    best_val_loss = float("inf")
    best_score = float("-inf")
    best_state = None
    epochs_without_improvement = 0

    with history_path.open("w", encoding="utf-8", newline="") as history_file:
        history = csv.DictWriter(history_file, fieldnames=history_fields)
        history.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_total = 0.0
            token_count = 0
            for batch_index, batch in enumerate(train_loader):
                if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                    break
                batch_images, batch_inputs, batch_targets = batch[:3]
                images = batch_images.to(device)
                decoder_inputs = batch_inputs.to(device)
                target_ids = batch_targets.to(device)
                if step < warmup_steps:
                    for group in optimizer.param_groups:
                        group["lr"] = args.learning_rate * (step + 1) / warmup_steps
                step += 1
                optimizer.zero_grad(set_to_none=True)
                if teacher is not None:
                    memory = model.encode(images)
                    loss = model.permutation_loss(
                        images, decoder_inputs, target_ids, args.label_smoothing, memory=memory
                    )
                    mse, kl = distillation_terms(
                        model, teacher, memory, decoder_inputs, target_ids,
                        batch[4].to(device), batch[5].to(device), args.distill_temperature,
                    )  # fmt: skip
                    loss = loss + args.distill_weight * mse + args.distill_kl * kl
                elif isinstance(model, PlateOCRParseq):
                    loss = model.permutation_loss(
                        images, decoder_inputs, target_ids, args.label_smoothing
                    )
                else:
                    logits = model(images, decoder_inputs)
                    loss = sequence_loss(logits, target_ids, args.label_smoothing)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                valid_count = int(target_ids[:, 1:].ne(PAD_INDEX).sum().item())
                loss_total += float(loss.item()) * valid_count
                token_count += valid_count

            train_loss = loss_total / max(token_count, 1)
            validation = evaluate_teacher_forced(model, val_loader, device, args.max_val_batches)
            greedy = evaluate_greedy(model, val_loader, device, args.max_val_batches)
            # Exact accuracy stays at 0 for the first epochs; a tiny CER term breaks ties
            # so early stopping does not fire before the decoder starts reading plates.
            score = (
                -validation["loss"]
                if args.select_by == "val_loss"
                else greedy["exact_accuracy"] - 1e-3 * greedy["cer"]
            )
            scheduler.step(-score if args.select_by == "val_loss" else score)
            learning_rate = optimizer.param_groups[0]["lr"]
            history.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": validation["loss"],
                    "val_token_accuracy": validation["token_accuracy"],
                    "val_teacher_forced_exact_accuracy": validation[
                        "teacher_forced_exact_accuracy"
                    ],
                    "val_teacher_forced_cer": validation["teacher_forced_cer"],
                    "val_greedy_exact_accuracy": greedy["exact_accuracy"],
                    "val_greedy_cer": greedy["cer"],
                    "learning_rate": learning_rate,
                }
            )
            history_file.flush()
            print(
                f"epoch {epoch:03d}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={validation['loss']:.4f} "
                f"teacher_forced_exact={validation['teacher_forced_exact_accuracy']:.3f} "
                f"teacher_forced_CER={validation['teacher_forced_cer']:.3f} "
                f"greedy_exact={greedy['exact_accuracy']:.3f} greedy_CER={greedy['cer']:.3f}"
            )

            if score > best_score:
                best_score = score
                best_val_loss = validation["loss"]
                best_state = {
                    name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                }
                temporary_path = checkpoint_path.with_suffix(".tmp")
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "val_loss": best_val_loss,
                        "val_greedy_exact_accuracy": greedy["exact_accuracy"],
                        "select_by": args.select_by,
                        "decoder": args.decoder,
                        "label_smoothing": args.label_smoothing,
                        "noise_augment": args.noise_augment,
                        "blue_plate_augment": args.blue_plate_augment,
                        "rotate_augment": args.rotate_augment,
                        "strong_downscale": args.strong_downscale,
                        "view_tilt": args.view_tilt,
                        "compression_augment": args.compression_augment,
                        "balance_rare_letters": args.balance_rare_letters,
                        "distill_teacher": str(args.distill_teacher or ""),
                        "distill_weight": args.distill_weight,
                        "distill_kl": args.distill_kl,
                        "model_config": {
                            "max_output_length": args.max_output_length,
                            "image_height": 64,
                            "image_width": 128,
                        },
                        "seed": args.seed,
                    },
                    temporary_path,
                )
                temporary_path.replace(checkpoint_path)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(
                        f"Early stopping after {args.patience} epochs without validation "
                        "improvement"
                    )
                    break

    if best_state is None:
        raise RuntimeError("Training completed without a validation checkpoint")
    model.load_state_dict(best_state)
    test_metrics = evaluate_greedy(model, test_loader, device, args.max_test_batches)
    print(
        "Held-out test (greedy autoregressive): "
        f"exact={test_metrics['exact_accuracy']:.3f} CER={test_metrics['cer']:.3f}"
    )
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
