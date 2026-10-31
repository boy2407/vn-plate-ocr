"""Character confusion matrices from the per-crop rows written by ``confusion_report.py``.

Each figure has two panels for one model x manifest:
left, row-normalised share of each true character read as each character (diagonal = per-char accuracy);
right, error counts only (diagonal blanked), so rare confusions are visible.
Same-length strings are compared by position, others aligned with difflib; a true character the model dropped goes to the "∅" column,
an extra character the model inserted goes to the "∅" row.

    .venv-cpu/bin/python -m custom_alpr.confusion_matrix_plot \
        custom_alpr/outputs/confusion-report/E-kaggle-test.json \
        custom_alpr/outputs/confusion-report/E-video.json
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

GAP = "∅"
BLUES = LinearSegmentedColormap.from_list(
    "seq-blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def aligned_pairs(reference: str, prediction: str) -> list[tuple[str, str]]:
    """(true, read) pairs for every character, with GAP standing in for a dropped or inserted one.

    Same-length strings are compared position by position: difflib otherwise shifts repeated
    digits ("8810" vs "8100") into a delete plus an insert instead of plain substitutions.
    """
    if len(reference) == len(prediction):
        return list(zip(reference, prediction, strict=True))
    pairs = []
    matcher = difflib.SequenceMatcher(a=reference, b=prediction, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        true, read = reference[a0:a1], prediction[b0:b1]
        # An unequal replace pairs position by position; only the overhang counts as dropped/inserted.
        common = min(len(true), len(read))
        pairs += list(zip(true[:common], read[:common], strict=True))
        pairs += [(t, GAP) for t in true[common:]] + [(GAP, r) for r in read[common:]]
    return pairs


def char_order(chars: set[str]) -> list[str]:
    digits = sorted(c for c in chars if c.isdigit())
    letters = sorted(c for c in chars if not c.isdigit() and c != GAP)
    return digits + letters + [GAP]


def plot(rows: list[dict], decoding: str, title: str, output: Path) -> dict:
    counts = Counter(pair for row in rows for pair in aligned_pairs(row["reference"], row[decoding]))
    labels = char_order({c for pair in counts for c in pair})
    index = {c: i for i, c in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for (true, read), n in counts.items():
        matrix[index[true], index[read]] += n

    totals = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)

    size = 0.32 * len(labels) + 2
    fig, axes = plt.subplots(1, 2, figsize=(2 * size + 1.5, size), constrained_layout=True)
    panels = (
        (axes[0], share, "Tỉ lệ theo hàng (đường chéo = độ chính xác từng ký tự)", 1.0),
        (axes[1], errors, "Chỉ lỗi — số lần đọc nhầm", max(errors.max(), 1)),
    )
    for ax, data, subtitle, vmax in panels:
        image = ax.imshow(data, cmap=BLUES, vmin=0, vmax=vmax)
        ax.set_xticks(range(len(labels)), labels, fontsize=8)
        ax.set_yticks(
            range(len(labels)),
            [f"{c} ({totals[i, 0]})" if c != GAP else c for i, c in enumerate(labels)],
            fontsize=8,
        )
        ax.set_xlabel("Model đọc ra")
        ax.set_ylabel("Ký tự đúng (số lần xuất hiện)")
        ax.set_title(subtitle, fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#ececec", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        fig.colorbar(image, ax=ax, shrink=0.6)

    # Label cells sparingly: diagonal accuracy below 95% on the left, every error on the right.
    for i in range(len(labels)):
        if totals[i, 0] and labels[i] != GAP and share[i, i] < 0.95:
            axes[0].text(i, i, f"{share[i, i] * 100:.0f}", ha="center", va="center", fontsize=6,
                         color="white" if share[i, i] > 0.55 else "#222")
        for j in range(len(labels)):
            if errors[i, j]:
                axes[1].text(j, i, str(errors[i, j]), ha="center", va="center", fontsize=6,
                             color="white" if errors[i, j] > 0.55 * errors.max() else "#222")

    exact = sum(row["reference"] == row[decoding] for row in rows) / len(rows)
    fig.suptitle(f"{title} — {len(rows)} crop, đúng cả biển {exact:.1%}", fontsize=12)
    fig.savefig(output, dpi=150)
    plt.close(fig)

    top = sorted(((n, f"{labels[i]}→{labels[j]}") for (i, j), n in np.ndenumerate(errors) if n), reverse=True)
    return dict(crops=len(rows), exact=round(exact, 4), top_errors=[(pair, n) for n, pair in top[:10]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rows", nargs="+", type=Path, help="per-crop JSON from confusion_report.py")
    parser.add_argument("--decoding", choices=("free", "layout"), default="free")
    parser.add_argument("--output", type=Path, default=Path("custom_alpr/outputs/confusion-report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in args.rows:
        rows = json.loads(path.read_text(encoding="utf-8"))
        output = args.output / f"{path.stem}-matrix.png"
        summary = plot(rows, args.decoding, path.stem, output)
        print(f"{output}  exact={summary['exact']:.3f} n={summary['crops']}  top={summary['top_errors']}")


if __name__ == "__main__":
    main()
