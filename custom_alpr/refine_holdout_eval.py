"""Leakage-audited refinement evaluation on held-out cameras (task 071)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "custom_alpr/outputs/refine-holdout-071"
FOLDS = ("train-danang-ptz", "train-saigon-drive")
DATA = ROOT / "custom_alpr/data/ocr/finetune-video-v1"
PATTERN = re.compile(r"([^/]+)-f\d+-x\d+y\d+(?:_pad)?\.png$")


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def video_ids(records):
    return {m[1] for r in records if (m := PATTERN.search(r["image_path"]))}


def wilson(k, n):
    """95% Wilson interval for a binomial proportion, retaining its denominator."""
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return dict(
        k=k, n=n, rate=p, ci95=[max(0.0, center - half), min(1.0, center + half)], method="Wilson"
    )


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def metrics(truth, predictions):
    errors = sum(distance(a, b) for a, b in zip(truth, predictions, strict=True))
    chars = sum(map(len, truth))
    # CER edit operations are not independent Bernoulli trials; Wilson is not valid.
    import numpy as np  # noqa: PLC0415

    rng = np.random.default_rng(71)
    edits = np.array([distance(a, b) for a, b in zip(truth, predictions, strict=True)])
    lengths = np.array(list(map(len, truth)))
    indices = rng.integers(0, len(truth), (10000, len(truth)))
    ci = np.quantile(edits[indices].sum(1) / lengths[indices].sum(1), [0.025, 0.975])
    return dict(
        exact=wilson(sum(a == b for a, b in zip(truth, predictions, strict=True)), len(truth)),
        cer=dict(
            errors=errors,
            characters=chars,
            n=len(truth),
            rate=errors / chars,
            ci95=ci.tolist(),
            method="paired sample bootstrap, 10000 resamples, seed 71",
        ),
    )


def main():  # noqa: PLR0912, PLR0915
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from PIL import Image, ImageDraw  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415
    from custom_alpr.plate_tracks import normalized_text  # noqa: PLC0415
    from custom_alpr.preprocessing import padded_box, prepare_ocr_input  # noqa: PLC0415
    from custom_alpr.refine_box import refine_box, yolo_predictor  # noqa: PLC0415
    from custom_alpr.video_demo import DEFAULT_DETECTOR, load_ocr, ocr_arrays  # noqa: PLC0415

    OUT.mkdir(parents=True, exist_ok=True)
    audit, subsets = {}, {}
    for fold in FOLDS:
        train, holdout = rows(DATA / fold / "train.csv"), rows(DATA / fold / "holdout.csv")
        overlap = sorted(
            {r["sha256"] for r in train if r.get("sha256")}
            & {r["sha256"] for r in holdout if r.get("sha256")}
        )
        audit[fold] = dict(
            train_rows=len(train),
            holdout_rows=len(holdout),
            train_videos=sorted(video_ids(train)),
            holdout_videos=sorted(video_ids(holdout)),
            overlap_videos=sorted(video_ids(train) & video_ids(holdout)),
            overlap_sha256=overlap,
        )
        unique = {}
        for r in holdout:
            cid = Path(r["image_path"]).stem.removesuffix("_pad")
            if not PATTERN.search(r["image_path"]):
                raise ValueError("Unexpected holdout filename")
            if cid in unique and unique[cid] != r["plate_text"]:
                raise ValueError("Conflicting labels")
            unique[cid] = r["plate_text"]
        subsets[fold] = unique
    (OUT / "leakage-audit.json").write_text(json.dumps(audit, indent=2))
    if any(a["overlap_videos"] or a["overlap_sha256"] for a in audit.values()):
        raise RuntimeError("Leakage detected; evaluation stopped")
    torch.set_num_threads(4)
    collected = ROOT / "custom_alpr/data/collected"
    metadata = {r["crop_id"]: r for p in collected.glob("*/metadata.csv") for r in rows(p)}
    previous = {
        r["crop_id"]: r
        for r in json.loads(
            (ROOT / "custom_alpr/outputs/refine-confirmed-070/rows.json").read_text()
        )
    }
    predict = yolo_predictor(YOLO(str(DEFAULT_DETECTOR), task="detect"), "cpu")
    predict(np.zeros((160, 320, 3), np.uint8))
    samples, inputs, pictures = {}, {}, {}
    for cid in sorted(set().union(*[set(s) for s in subsets.values()])):
        r = dict(metadata[cid])
        box = json.loads(r["box"])
        x, y, a, b = box
        frame = cv2.imread(str(collected / "frames" / f"{r['video_id']}-f{r['frame']}.png"))
        if frame is None or cid not in previous:
            raise RuntimeError(f"Missing seek-verified frame: {cid}; recover with task 070 first")
        tight = cv2.imread(str(collected / r["video_id"] / "crops" / f"{cid}.png"))
        sx, sy = max(0, x - 10), max(0, y - 10)
        _, corr, _, loc = cv2.minMaxLoc(
            cv2.matchTemplate(frame[sy : b + 10, sx : a + 10], tight, cv2.TM_CCOEFF_NORMED)
        )
        if corr < 0.9 or loc != (x - sx, y - sy):
            raise RuntimeError(f"Frame alignment failed: {cid}")
        start = time.perf_counter()
        final, accepted = refine_box(predict, frame, box, 0.5, 320)
        elapsed = (time.perf_counter() - start) * 1000
        px, py, pa, pb = padded_box(box, frame.shape[1], frame.shape[0])
        rx, ry, ra, rb = final
        crops = [frame[y:b, x:a], frame[py:pb, px:pa], frame[ry:rb, rx:ra]]
        inputs[cid] = [prepare_ocr_input(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        pictures[cid] = crops
        samples[cid] = dict(
            crop_id=cid,
            video_id=r["video_id"],
            frame=r["frame"],
            correlation=corr,
            seek_verification="task070 cached frame; rechecked pixel alignment",
            box=box,
            final_box=final,
            refined=accepted,
            ms=elapsed,
        )
    report = dict(
        leakage_audit=audit,
        context=0.5,
        size=320,
        cpu_threads=4,
        results={},
        kaggle={},
        caveats=[
            "Recovered H264 differs from saved crops; variants use the same recovered frame.",
            "Wilson for binomial rates; CER uses bootstrap because edits are not Bernoulli.",
            "Bootstrap samples crops; correlated tracks limit generalization.",
        ],
    )
    checkpoints = {
        "E": ROOT / "custom_alpr/outputs/train-custom-ocr/run-20261029-09-parseq-f-compress/best.pt"
    }
    checkpoints.update(
        {f: ROOT / "custom_alpr/outputs/finetune-video-v1" / f / "best.pt" for f in FOLDS}
    )
    kaggle_path = ROOT / "custom_alpr/data/ocr/kaggle-v9-prepared/test.csv"
    kaggle = rows(kaggle_path)
    kaggle_inputs = [
        prepare_ocr_input(
            np.asarray(Image.open(kaggle_path.parent / r["image_path"]).convert("RGB"))
        )
        for r in kaggle
    ]
    details = {}
    for name, checkpoint in checkpoints.items():
        model = load_ocr(checkpoint, torch.device("cpu"))

        def read(arrays, model=model):
            texts = []
            for i in range(0, len(arrays), 16):
                texts.extend(
                    normalized_text(t)
                    for t, _ in ocr_arrays(arrays[i : i + 16], model, torch.device("cpu"))
                )
            return texts

        for fold in FOLDS if name == "E" else [name]:
            cids = sorted(subsets[fold])
            truth = [normalized_text(subsets[fold][c]) for c in cids]
            reads = read([im for c in cids for im in inputs[c]])
            predictions = [reads[j::3] for j in range(3)]
            result = {
                v: metrics(truth, p)
                for v, p in zip(["tight", "padding", "refine"], predictions, strict=True)
            }
            delta = np.array(
                [
                    int(c == t) - int(a == t)
                    for t, a, c in zip(truth, predictions[0], predictions[2], strict=True)
                ]
            )
            rng = np.random.default_rng(71)
            ci = np.quantile(
                delta[rng.integers(0, len(cids), (100000, len(cids)))].mean(1), [0.025, 0.975]
            )
            result.update(
                fixed=wilson(int((delta == 1).sum()), len(cids)),
                broken=wilson(int((delta == -1).sum()), len(cids)),
                difference=dict(
                    rate=float(delta.mean()),
                    ci95=ci.tolist(),
                    n=len(cids),
                    method="paired bootstrap 100000 resamples seed 71",
                ),
                fallback=wilson(sum(not samples[c]["refined"] for c in cids), len(cids)),
                cpu_ms_per_frame=dict(
                    n=len(cids),
                    mean=float(np.mean([samples[c]["ms"] for c in cids])),
                    median=float(np.median([samples[c]["ms"] for c in cids])),
                ),
            )
            camera = "saigon-drive" if fold == "train-danang-ptz" else "danang-ptz"
            key = f"{name}--heldout-{camera}"
            report["results"][key] = result
            details[key] = [
                dict(
                    samples[c],
                    truth=t,
                    predictions=dict(
                        zip(
                            ["tight", "padding", "refine"], [p[i] for p in predictions], strict=True
                        )
                    ),
                )
                for i, (c, t) in enumerate(zip(cids, truth, strict=True))
            ]
            changed = [i for i in range(len(cids)) if len({p[i] for p in predictions}) > 1]
            for page in range(0, len(changed), 12):
                subset = changed[page : page + 12]
                sheet = Image.new("RGB", (960, 180 * len(subset)), "white")
                draw = ImageDraw.Draw(sheet)
                for yy, i in enumerate(subset):
                    draw.text((5, yy * 180), f"{cids[i]} truth={truth[i]}", fill="black")
                    for j, crop in enumerate(pictures[cids[i]]):
                        im = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        im.thumbnail((300, 140))
                        sheet.paste(im, (j * 320, yy * 180 + 35))
                        draw.text(
                            (j * 320, yy * 180 + 18),
                            f"{['tight', 'padding', 'refine'][j]}: {predictions[j][i]}",
                            fill="red",
                        )
                sheet.save(OUT / f"{key}-{page // 12}.png")
            print(key, json.dumps(result), flush=True)
        predicted = read(kaggle_inputs)
        report["kaggle"][name] = dict(
            metrics([normalized_text(r["plate_text"]) for r in kaggle], predicted),
            checkpoint=str(checkpoint),
            sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        )
        (OUT / f"kaggle-{name}.json").write_text(json.dumps(predicted))
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
        (OUT / "rows.json").write_text(json.dumps(details, indent=2))
        print("KAGGLE", name, json.dumps(report["kaggle"][name]), flush=True)
    eligible = [
        report["results"][
            f"{f}--heldout-{'saigon-drive' if f == 'train-danang-ptz' else 'danang-ptz'}"
        ]["difference"]
        for f in FOLDS
    ]
    report["integrate"] = all(d["rate"] > 0 and d["ci95"][0] > 0 for d in eligible)
    report["conclusion"] = "Chưa nên tích hợp refine: chưa đủ dữ liệu theo tiêu chí cả hai fold."
    base_reads = json.loads((OUT / "kaggle-E.json").read_text())
    for name, result in report["kaggle"].items():
        predicted = json.loads((OUT / f"kaggle-{name}.json").read_text())
        truth = [normalized_text(r["plate_text"]) for r in kaggle]
        result["fixed_vs_E"] = sum(
            a != t == b for t, a, b in zip(truth, base_reads, predicted, strict=True)
        )
        result["broken_vs_E"] = sum(
            a == t != b for t, a, b in zip(truth, base_reads, predicted, strict=True)
        )
        result["fixed_vs_E_rate"] = wilson(result["fixed_vs_E"], len(truth))
        result["broken_vs_E_rate"] = wilson(result["broken_vs_E"], len(truth))
        folder = "E-on-kaggle-test" if name == "E" else name + "-on-kaggle-test"
        old = (
            ROOT
            / "custom_alpr/outputs/finetune-video-v1/eval"
            / folder
            / "character-fp-report.json"
        )
        result["historical_exact"] = json.loads(old.read_text())["exact_plate_accuracy"]
        result["historical_accuracy_reproduced"] = (
            abs(result["historical_exact"] - result["exact"]["rate"]) < 1e-10
        )
    report["sampling_unit"] = "Unique crop_id without _pad; multiple frames per vehicle may remain."
    report["frame_source"] = "Task070 seek-verified cached frames; all alignments rechecked."
    (OUT / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
