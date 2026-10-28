"""Native FastALPR YOLOv9-t fine-tuning with the established corruption budget."""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from custom_alpr.yolo_vast.corruptions import apply_budget
from custom_alpr.yolo_vast.data import verify_inventory

UPSTREAM_COMMIT = "5b1ea9a8b3f0ffe4fe0e203ec6232d788bb3fcff"
INITIAL_SHA256 = "4d525da3ebe44fb6e6bfe8b05ca717b777db3feb2b44b472250f8b32f4754fdb"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NoExtraAugmentation:
    """Disable implicit optional Albumentations; preserve native geometry/HSV."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, image, labels, **_kwargs):
        return image, labels


class NativeQuality:
    """Apply the same BGR pixel corruptions to native RGB CHW training tensors."""

    def __init__(self, folder: Path):
        self.folder = folder
        self.rng = None
        self.seed = None
        self.counts = Counter()

    def __call__(self, image, labels, *, mosaic_closed=False):
        worker = torch.utils.data.get_worker_info()
        seed = (worker.seed if worker else torch.initial_seed()) % (2**32)
        if self.seed != seed:
            self.seed, self.rng = seed, np.random.default_rng(seed + 4852)
        bgr = image.numpy().transpose(1, 2, 0)[..., ::-1].copy()
        height, width = bgr.shape[:2]
        xywh = labels[:, 2:6].numpy().copy() * [width, height, width, height]
        boxes = np.concatenate(
            (xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), axis=1
        )
        result, effects = apply_budget(bgr, self.rng, boxes)
        self.counts["images"] += 1
        self.counts[f"effects_{len(effects)}"] += 1
        self.counts["mosaic_closed_images"] += int(mosaic_closed)
        self.counts.update(effects)
        if self.counts["images"] == 1 or self.counts["images"] % 128 == 0:
            self.folder.mkdir(parents=True, exist_ok=True)
            path = self.folder / f"worker-{os.getpid()}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(dict(self.counts), indent=2))
            tmp.replace(path)
        return torch.from_numpy(np.ascontiguousarray(result[..., ::-1].transpose(2, 0, 1)))


def install_quality(dataset_class, epoch, epochs, close_mosaic, folder):
    original = dataset_class.__getitem__

    def getitem(self, index):
        closed = bool(close_mosaic and epoch.value >= epochs - close_mosaic)
        if self.augment and closed:
            self.mosaic = False  # shared epoch is visible in persistent worker copies
        image, labels, path, shapes = original(self, index)
        if self.augment:
            if not hasattr(self, "_plate_quality"):
                self._plate_quality = NativeQuality(folder)
            image = self._plate_quality(image, labels, mosaic_closed=closed)
        return image, labels, path, shapes

    dataset_class.__getitem__ = getitem


def configure(root: Path, dataset: Path):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root / "upstream", text=True
    ).strip()
    if commit != UPSTREAM_COMMIT or digest(root / "weights/initial.pt") != INITIAL_SHA256:
        raise ValueError("Unexpected upstream commit or initial checkpoint")
    inventory = verify_inventory(dataset)
    selected = [
        str(dataset / "data" / row["image"])
        for row in inventory["records"]
        if row["split"] == "train" and row["image"] not in set(inventory["train_exclude"])
    ]
    if len(selected) != 9249:
        raise ValueError("Expected the same 9249-image training split")
    runtime = (dataset / "train.runtime.txt").read_text().splitlines()
    if set(runtime) != set(selected):
        raise ValueError("Runtime training list differs from the quarantined YOLOv8 split")
    train = root / "train.txt"
    train.write_text("\n".join(selected) + "\n")
    config = dict(
        path=str(dataset / "data"),
        train=str(train),
        val="images/val",
        test="images/test",
        nc=1,
        names={0: "license_plate"},
    )
    (root / "dataset.yaml").write_text(yaml.safe_dump(config))
    rng = random.Random(42)
    for split, count in (("train", 32), ("val", 16)):
        source = (
            selected
            if split == "train"
            else [
                str(dataset / "data" / r["image"])
                for r in inventory["records"]
                if r["split"] == "val"
            ]
        )
        (root / f"smoke-{split}.txt").write_text("\n".join(rng.sample(source, count)) + "\n")
    smoke = dict(config, train=str(root / "smoke-train.txt"), val=str(root / "smoke-val.txt"))
    (root / "smoke.yaml").write_text(yaml.safe_dump(smoke))
    hyp = yaml.safe_load((root / "upstream/data/hyps/hyp.scratch-high.yaml").read_text())
    previous = yaml.safe_load((dataset / "custom_alpr/yolo_vast/configs/train.yaml").read_text())
    for name in (
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "mosaic",
        "mixup",
        "copy_paste",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "fliplr",
        "flipud",
    ):
        hyp[name] = previous[name]
    hyp["momentum"] = 0.937
    # Native SGD bias warmup (0.1) is destructive for pretrained AdamW fine-tuning.
    hyp["warmup_bias_lr"] = 0.0
    (root / "hyp.yaml").write_text(yaml.safe_dump(hyp))
    return inventory


def native_imports(root):
    os.chdir(root / "upstream")
    sys.path.insert(0, str(root / "upstream"))
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # compatibility with the pinned trainer on NumPy 2.5+
    original = torch.load

    def legacy_load(*args, **kwargs):
        # Official/native checkpoints serialize model classes, not plain tensors.
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = legacy_load
    import train  # noqa: PLC0415 -- pinned external repository loaded after path setup

    return train


def preflight(args, native):
    inventory = configure(args.root, args.dataset)
    checkpoint = torch.load(args.root / "weights/initial.pt", map_location="cpu")
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    if type(model.model[-1]).__name__ != "DDetect" or model.model[-1].nc != 1:
        raise ValueError("Expected the converted, single-class FastALPR YOLOv9-t head")
    if sum(p.numel() for p in model.parameters()) != 1913443:
        raise ValueError("Unexpected model architecture")
    model.model[-1].export = False
    source = json.loads((args.dataset / "inventory.json").read_text())
    examples = [r for r in source["records"] if r["split"] == "val"][:8]
    from open_image_models.detection.core.yolo_v9.preprocess import preprocess  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415
    from open_image_models.detection.core.hub import download_model  # noqa: PLC0415

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(download_model("yolo-v9-t-384-license-plate-end2end")),
        options,
        providers=["CPUExecutionProvider"],
    )
    parity = []
    torch.set_num_threads(4)
    for row in examples:
        image = cv2.imread(str(args.dataset / "data" / row["image"]))
        pixels, _, _ = preprocess(image, 384)
        with torch.inference_mode():
            raw = model(torch.from_numpy(pixels))[0]
        predictions = native.validate.non_max_suppression(raw, 0.4, 0.45, max_det=100)[0].numpy()
        reference = session.run(None, {session.get_inputs()[0].name: pixels})[0].reshape(-1, 7)
        reference = reference[reference[:, 6] >= 0.4]
        assert len(reference) == len(predictions), (
            "Native PT and release ONNX detection count differs"
        )
        error = float(np.abs(predictions[:, :4] - reference[:, 1:5]).max()) if len(reference) else 0
        if error > 0.1:
            raise ValueError(f"PT/ONNX coordinate mismatch: {error}")
        parity.append(dict(image=row["image"], boxes=len(reference), max_error_px=error))
    checkpoint.update(model=model, ema=None, optimizer=None, epoch=-1, best_fitness=0.0, updates=0)
    torch.save(checkpoint, args.root / "weights/prepared-initial.pt")
    report = dict(
        initial_sha256=INITIAL_SHA256,
        upstream_commit=UPSTREAM_COMMIT,
        inventory_sha256=digest(args.dataset / "inventory.json"),
        train=inventory["effective_counts"]["train"],
        val=inventory["counts"]["val"],
        test=inventory["counts"]["test"],
        parity=parity,
        corruption_sha256=digest(Path(__file__).parent / "yolo_vast/corruptions.py"),
    )
    (args.root / "preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def train_model(args, native):  # noqa: PLR0915 -- staged training with integrity and resume checks
    smoke = args.stage == "smoke"
    epochs, close = (1, 0) if smoke else (30, 10)
    name = f"smoke-{args.device}" if smoke else "moderate-384-s42"
    save = args.root / "runs" / name
    if save.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {save}")
    if not (args.root / "preflight.json").exists():
        raise ValueError("Run preflight first")
    preflight_report = json.loads((args.root / "preflight.json").read_text())
    if digest(args.dataset / "inventory.json") != preflight_report["inventory_sha256"]:
        raise ValueError("Dataset manifest changed after preflight")
    if (
        digest(Path(__file__).parent / "yolo_vast/corruptions.py")
        != preflight_report["corruption_sha256"]
    ):
        raise ValueError("Corruption policy changed after preflight")
    verify_inventory(args.dataset)
    epoch = mp.Value("i", -1)
    from utils import dataloaders  # noqa: PLC0415

    dataloaders.Albumentations = NoExtraAugmentation
    install_quality(dataloaders.LoadImagesAndLabels, epoch, epochs, close, save / "quality_counts")
    native.Loggers = functools.partial(native.Loggers, include=("csv", "tb"))
    opt = native.parse_opt(known=True)
    values = dict(
        weights=str(args.resume or args.root / "weights/prepared-initial.pt"),
        cfg="",
        data=str(args.root / ("smoke.yaml" if smoke else "dataset.yaml")),
        hyp=str(args.root / "hyp.yaml"),
        epochs=epochs,
        batch_size=4 if smoke and args.device == "cpu" else 16,
        imgsz=384,
        workers=0 if args.device == "cpu" else 4,
        optimizer="AdamW",
        seed=42,
        cos_lr=True,
        patience=30,
        close_mosaic=close,
        save_period=1,
        noplots=True,
        project=str(args.root / "runs"),
        name=name,
        save_dir=str(save),
        exist_ok=False,
        noautoanchor=True,
        resume=bool(args.resume),
        nosave=False,
        noval=False,
    )
    for key, value in values.items():
        setattr(opt, key, value)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu")
        if state.get("optimizer") is None or state.get("epoch", -1) >= epochs - 1:
            raise ValueError("Checkpoint is stripped or already finished")
        epoch.value = state["epoch"]
    callbacks = native.Callbacks()

    def start_epoch():
        with epoch.get_lock():
            epoch.value += 1

    def finite_loss(_model, _step, _images, _targets, _paths, losses):
        if not np.isfinite([float(v) for v in losses]).all():
            raise ValueError("Nonfinite training loss")

    def progress(values, current_epoch, *_args):
        (save / "progress.json").write_text(
            json.dumps(
                dict(epoch=current_epoch + 1, epochs=epochs, values=[float(v) for v in values]),
                indent=2,
            )
        )

    callbacks.register_action("on_train_epoch_start", callback=start_epoch)
    callbacks.register_action("on_train_batch_end", callback=finite_loss)
    callbacks.register_action("on_fit_epoch_end", callback=progress)
    device = native.select_device(args.device, batch_size=opt.batch_size)
    torch.set_num_threads(8)
    save.mkdir(parents=True, exist_ok=bool(args.resume))
    plan = dict(
        values,
        preflight=json.loads((args.root / "preflight.json").read_text()),
        runner_sha256=digest(__file__),
        hyp_sha256=digest(args.root / "hyp.yaml"),
        train_list_sha256=digest(args.root / "train.txt"),
    )
    plan["resume"] = False
    metadata = save / "experiment.json"
    if not args.resume:
        metadata.write_text(json.dumps(plan, indent=2))
    else:
        recorded = json.loads(metadata.read_text())
        for key in (
            "runner_sha256",
            "hyp_sha256",
            "train_list_sha256",
            "preflight",
            "epochs",
            "imgsz",
            "batch_size",
        ):
            if recorded[key] != plan[key]:
                raise ValueError(f"Resume fingerprint differs: {key}")
    result = native.train(str(args.root / "hyp.yaml"), opt, device, callbacks)
    (save / "run_result.json").write_text(
        json.dumps(
            dict(status="complete", results=list(result), best=str(save / "weights/best.pt")),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage", choices=("preflight", "smoke", "train"), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.root, args.dataset = args.root.resolve(), args.dataset.resolve()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("WANDB_MODE", "disabled")
    cv2.setNumThreads(1)
    with (args.root / "training.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native = native_imports(args.root)
        if args.stage == "preflight":
            preflight(args, native)
        else:
            train_model(args, native)


if __name__ == "__main__":
    main()
