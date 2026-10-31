"""Export the PARSeq plate OCR to one ONNX graph and run it with ONNX Runtime on CPU."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from custom_alpr.custom_ocr import (
    BOS_INDEX,
    EOS_INDEX,
    PAD_INDEX,
    PlateOCRParseq,
    cloze_attention_mask,
    decode_targets,
    permutation_attention_mask,
)


def ended_after_eos(predicted: Tensor) -> Tensor:
    # ONNX CumSum rejects bool tensors, so count EOS as integers.
    return predicted.eq(EOS_INDEX).long().cumsum(dim=1) > 0


def context_from_prediction(predicted: Tensor) -> Tensor:
    """``custom_ocr.context_from_prediction`` with the ONNX-safe EOS mask."""
    ended = ended_after_eos(predicted)
    context = torch.cat((torch.full_like(predicted[:, :1], BOS_INDEX), predicted[:, :-1]), dim=1)
    after_eos = torch.cat((torch.zeros_like(ended[:, :2]), ended[:, :-2]), dim=1)
    return context.masked_fill(after_eos[:, : context.shape[1]], PAD_INDEX)


class UnrolledParseq(nn.Module):
    """``generate_with_confidence`` without the early exit, so tracing unrolls every step.

    The exit only skips steps once every plate has emitted EOS; the cloze refine pass
    pads everything after EOS, so the extra steps do not change the output.
    """

    def __init__(self, model: PlateOCRParseq):
        super().__init__()
        self.model = model

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        model = self.model
        steps = model.max_output_length - 1
        memory = model.encode(images)
        batch_size = images.shape[0]
        context = torch.full((batch_size, steps), PAD_INDEX, dtype=torch.long)
        context[:, 0] = BOS_INDEX
        causal = permutation_attention_mask(torch.arange(steps))
        step_logits = []
        for step in range(steps):
            current = model.decode_positions(memory, context, causal)[:, step]
            step_logits.append(current)
            if step + 1 < steps:
                context = context.clone()
                context[:, step + 1] = current.argmax(dim=-1)
        logits = torch.stack(step_logits, dim=1)
        cloze = cloze_attention_mask(steps, images.device)
        for _ in range(model.refine_iterations):
            context = context_from_prediction(logits.argmax(dim=-1))
            logits = model.decode_positions(memory, context, cloze)
        probabilities, predicted = logits.softmax(dim=-1).max(dim=-1)
        ended = ended_after_eos(predicted)
        characters = (~ended) & (predicted >= 3)
        log_sum = torch.where(characters, probabilities.clamp_min(1e-12).log(), 0.0).sum(dim=1)
        counts = characters.sum(dim=1)
        scores = torch.where(counts > 0, (log_sum / counts.clamp_min(1)).exp(), 0.0)
        return predicted, scores


def export(model: PlateOCRParseq, output: Path):
    # Tracing alters the module it is given (PyTorch encoder output moved by ~3 after an
    # export), so trace a copy and leave the caller's model usable.
    model = copy.deepcopy(model).eval()
    sample = torch.zeros(2, 3, model.image_height, model.image_width)
    torch.onnx.export(
        UnrolledParseq(model),
        (sample,),
        str(output),
        input_names=["images"],
        output_names=["predicted", "scores"],
        dynamic_axes={"images": {0: "batch"}, "predicted": {0: "batch"}, "scores": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


class OnnxPlateOCR:
    """Drop-in for ``PlateOCRParseq.generate_with_confidence`` backed by ONNX Runtime."""

    def __init__(self, path: Path, threads: int):
        import onnxruntime as ort  # noqa: PLC0415 -- only the ONNX path needs it

        options = ort.SessionOptions()
        # Explicit counts: in a CPU-limited container ORT otherwise pins one thread per
        # host core and fails to set affinity on cores it cannot use.
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    def generate_with_confidence(self, images: Tensor) -> tuple[list[str], list[float]]:
        predicted, scores = self.session.run(
            None, {"images": images.detach().cpu().numpy().astype(np.float32)}
        )
        return decode_targets(torch.from_numpy(predicted)), scores.tolist()


def main():
    from custom_alpr.video_demo import DEFAULT_OCR, load_ocr  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.checkpoint.with_suffix(".onnx")
    export(load_ocr(args.checkpoint, torch.device("cpu")), output)
    print(output)


if __name__ == "__main__":
    main()
