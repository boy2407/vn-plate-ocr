"""Validate frozen task072 proposals; never score Sài Gòn without original-set provenance."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from custom_alpr.box_context_sweep import iou, yolo_boxes
from custom_alpr.refine_fallback_report import E, FRAMES, OUT, ROOT, candidates, classify, cut, read
from custom_alpr.refine_holdout_eval import metrics, wilson
from custom_alpr.video_demo import DEFAULT_DETECTOR, load_ocr, select_boxes

PROPOSALS = ("remove_min_area", "remove_max_area")


def decide(first, found, proposal):
    best = found[0] if found else None
    reason = classify(first, best)
    if reason == "accepted":
        return best["box"], True
    if reason != "area" or proposal == "baseline":
        return first, False
    ratio = best["area_ratio"]
    if not (
        (proposal == "remove_min_area" and ratio < 0.2)
        or (proposal == "remove_max_area" and ratio > 0.95)
    ):
        return first, False
    # Recheck later outside_slack after bypassing only the specified area bound.
    x, y, a, b = best["box"]
    slack = 0.25 * max(first[2] - first[0], first[3] - first[1])
    accepted = (
        x >= first[0] - slack
        and y >= first[1] - slack
        and a <= first[2] + slack
        and b <= first[3] + slack
    )
    return (best["box"] if accepted else first), accepted


def summarize(records):
    truth = [r["truth"] for r in records]
    result = {}
    for variant in ["tight", "baseline", *PROPOSALS]:
        predicted = [r["texts"][variant] for r in records]
        m = metrics(truth, predicted)
        for reference in ["tight", "baseline"]:
            delta = np.array(
                [
                    int(p == r["truth"]) - int(r["texts"][reference] == r["truth"])
                    for p, r in zip(predicted, records, strict=True)
                ]
            )
            rng = np.random.default_rng(72)
            ci = np.quantile(
                delta[rng.integers(0, len(records), (100000, len(records)))].mean(1), [0.025, 0.975]
            )
            m[f"vs_{reference}"] = dict(
                fixed=wilson(int((delta == 1).sum()), len(records)),
                broken=wilson(int((delta == -1).sum()), len(records)),
                difference=dict(
                    rate=float(delta.mean()),
                    ci95=ci.tolist(),
                    n=len(records),
                    method="paired bootstrap 100000 seed72",
                ),
            )
        if variant != "tight":
            m["fallback"] = wilson(sum(not r["accepted"][variant] for r in records), len(records))
        result[variant] = m
    return result


def main():  # noqa: PLR0912, PLR0915
    torch.set_num_threads(4)
    proposals = dict(
        remove_min_area="Remove only MIN_AREA; retain MAX_AREA=.95 and all other checks.",
        remove_max_area="Remove only MAX_AREA; retain MIN_AREA=.2 and all other checks.",
    )
    frozen = OUT / "proposals.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == proposals
    else:
        frozen.write_text(json.dumps(proposals, indent=2))
    cache = OUT / "detector-val-candidates.json"
    if cache.exists():
        boxes = json.loads(cache.read_text())
    else:
        from custom_alpr.refine_box import yolo_predictor  # noqa: PLC0415

        detector = YOLO(str(DEFAULT_DETECTOR), task="detect")
        predict = yolo_predictor(detector, "cpu")
        boxes = []
        for index, path in enumerate(
            sorted((ROOT / "custom_alpr/data/detection/images/val").iterdir())
        ):
            label = ROOT / "custom_alpr/data/detection/labels/val" / f"{path.stem}.txt"
            if not label.exists():
                continue
            frame = cv2.imread(str(path))
            height, width = frame.shape[:2]
            truth = [b for b in yolo_boxes(label, width, height) if b[3] - b[1] >= 16]
            prediction = detector.predict(frame, conf=0.4, imgsz=640, device="cpu", verbose=False)[
                0
            ]
            first = [list(b[:4]) for b in select_boxes(prediction.boxes, width, height)]
            for gt in truth:
                match = max(first, key=lambda b: iou(gt, b), default=None)
                if match is None or iou(gt, match) < 0.5:
                    continue
                boxes.append(
                    dict(
                        image=str(path),
                        truth=gt,
                        first=match,
                        candidates=candidates(predict, frame, match),
                    )
                )
            if index % 100 == 0:
                print("DETECTOR", index, len(boxes), flush=True)
        cache.write_text(json.dumps(boxes, indent=2))
    # The user supplied original-set provenance after the supplemental run.
    # Verify its baseline before using this set to judge any proposal.
    original_dir = ROOT / "custom_alpr/outputs/detval-sample/images/val"
    if original_dir.exists():
        names = {p.name for p in original_dir.iterdir()}
        original = [r for r in boxes if Path(r["image"]).name in names]
        first_iou = [iou(r["truth"], r["first"]) for r in original]
        final_iou = [
            iou(r["truth"], decide(r["first"], r["candidates"], "baseline")[0]) for r in original
        ]
        baseline_check = dict(
            images=len(names),
            plates=len(original),
            median_first=statistics.median(first_iou),
            median_final=statistics.median(final_iou),
            worse_over_01=sum(b < a - 0.1 for a, b in zip(first_iou, final_iou, strict=True)),
        )
        baseline_check["gate"] = (
            baseline_check["images"] == 358
            and baseline_check["plates"] == 429
            and round(baseline_check["median_first"], 3) == 0.892
            and round(baseline_check["median_final"], 3) == 0.925
            and baseline_check["worse_over_01"] == 0
        )
        (OUT / "original-baseline-gate.json").write_text(json.dumps(baseline_check, indent=2))
        if not baseline_check["gate"]:
            raise RuntimeError(
                "Original 429 baseline differs from required numbers; stop before proposal scoring."
            )
    detector_report = {}
    for proposal in ["baseline", *PROPOSALS]:
        before = []
        after = []
        vs_baseline = []
        for r in boxes:
            final, _ = decide(r["first"], r["candidates"], proposal)
            baseline, _ = decide(r["first"], r["candidates"], "baseline")
            before.append(iou(r["truth"], r["first"]))
            after.append(iou(r["truth"], final))
            vs_baseline.append(iou(r["truth"], baseline))
        detector_report[proposal] = dict(
            n=len(boxes),
            median_iou_first=statistics.median(before),
            median_iou_final=statistics.median(after),
            worse_than_first_over_01=wilson(
                sum(b < a - 0.1 for a, b in zip(before, after, strict=True)), len(boxes)
            ),
            worse_than_baseline_over_01=wilson(
                sum(b < a - 0.1 for a, b in zip(vs_baseline, after, strict=True)), len(boxes)
            ),
        )
    records = [
        r for r in json.loads((OUT / "rows.json").read_text()) if r["camera"] == "danang-ptz"
    ]
    crops = []
    for r in records:
        frame = cv2.imread(str(FRAMES / f"{r['video_id']}-f{r['frame']}.png"))
        r["accepted"] = {}
        r["texts"] = {"tight": r["E_tight"]}
        for proposal in ["baseline", *PROPOSALS]:
            final, accepted = decide(r["box"], r["candidates"], proposal)
            r["accepted"][proposal] = accepted
            crops.append(cut(frame, final))
    model = load_ocr(E, torch.device("cpu"))
    texts = read(model, crops)
    for i, r in enumerate(records):
        r["texts"].update(
            dict(zip(["baseline", *PROPOSALS], texts[3 * i : 3 * i + 3], strict=True))
        )
        assert r["texts"]["baseline"] == r["predictions"]["refine"]
    danang = summarize(records)
    # Original 429 sample IDs/results are unavailable: current val is supplemental.
    gates = {}
    for proposal in PROPOSALS:
        d = detector_report[proposal]
        base = detector_report["baseline"]
        v = danang[proposal]
        gates[proposal] = dict(
            original_429_verified=False,
            supplemental_box_gate=d["median_iou_final"] >= base["median_iou_final"]
            and d["worse_than_first_over_01"]["k"] <= base["worse_than_first_over_01"]["k"]
            and d["worse_than_baseline_over_01"]["k"] == 0,
            danang_gate=v["vs_baseline"]["broken"]["k"] == 0
            and v["exact"]["k"] >= danang["baseline"]["exact"]["k"],
            saigon_run=False,
            reason="Original 429 identities unavailable; no Sài Gòn proposal evaluation permitted.",
        )
    report = dict(
        proposals=proposals,
        detector_set=dict(
            status="supplemental, not verified original 429",
            selection=(
                "all detection/images/val; truth height>=16; "
                "first prediction conf=.4 imgsz640; matched IoU>=.5"
            ),
            results=detector_report,
        ),
        danang=danang,
        gates=gates,
        saigon=dict(proposal_evaluation_runs=0, status="not run: original 429 gate unverified"),
        sampling="crop_id, not independent physical vehicles",
    )
    (OUT / "validation-report.json").write_text(json.dumps(report, indent=2))
    (OUT / "danang-proposal-rows.json").write_text(json.dumps(records, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
