"""Run checked control/moderate training or resume the same experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torchvision
import ultralytics
import yaml
from ultralytics import YOLO

from custom_alpr.yolo_vast.data import code_fingerprint, runtime_yaml, sha256, verify_inventory
from custom_alpr.yolo_vast.trainer import ControlTrainer, ModerateTrainer


def gpu_probe(device="0"):
    if device == "cpu":
        return {"device": "cpu", "torch": torch.__version__}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Activate the PyTorch GPU environment first.")
    index = int(device)
    dev = torch.device(f"cuda:{index}")
    layer = torch.nn.Conv2d(3, 8, 3).to(dev)
    sample = torch.randn(2, 3, 32, 32, device=dev)
    layer(sample).square().mean().backward()
    torchvision.ops.nms(
        torch.tensor([[0.0, 0.0, 20.0, 20.0]], device=dev), torch.tensor([0.9], device=dev), 0.5
    )
    torch.cuda.synchronize(dev)
    return dict(
        device=str(dev),
        name=torch.cuda.get_device_name(index),
        capability=torch.cuda.get_device_capability(index),
        vram_bytes=torch.cuda.get_device_properties(index).total_memory,
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        cuda=torch.version.cuda,
    )


def main():  # noqa: PLR0912, PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "moderate"), default="moderate")
    parser.add_argument("--name")
    parser.add_argument(
        "--device", default="0", help="Single CUDA device index; cpu only for smoke"
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path, help="Existing experiment folder")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if ultralytics.__version__ != "8.4.183":
        parser.error("This trainer is validated with ultralytics==8.4.183")
    if args.device == "cpu" and not (args.smoke or args.preflight_only):
        parser.error("Full runs require CUDA; --device cpu is for smoke only")
    if args.batch is not None and args.batch < 1:
        parser.error("batch must be positive")
    if args.epochs is not None and args.epochs < 1:
        parser.error("epochs must be positive")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("Another YOLO job already owns this bundle") from exc
        inventory = verify_inventory(root)
        environment = gpu_probe(args.device)
        environment.update(ultralytics=ultralytics.__version__, python=platform.python_version())
        (root / "preflight.json").write_text(json.dumps(environment, indent=2))
        if args.preflight_only:
            print(json.dumps(environment, indent=2))
            return
        fingerprint = dict(data=sha256(root / "inventory.json"), code=code_fingerprint(root))
        config = yaml.safe_load((root / "custom_alpr/yolo_vast/configs/train.yaml").read_text())
        if args.resume:
            directory = args.resume.resolve()
            metadata = json.loads((directory / "experiment.json").read_text())
            if metadata["fingerprint"] != fingerprint:
                raise ValueError(
                    "Code or dataset changed: resume rejected; create a new experiment"
                )
            arm, smoke = metadata["arm"], metadata["smoke"]
            checkpoint = directory / "weights/last.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if state.get("optimizer") is None or state.get("epoch", -1) < 0:
                raise ValueError("Checkpoint is completed/stripped, not resumable")
        else:
            arm, smoke = args.arm, args.smoke
            name = args.name or (f"smoke-{arm}-{int(time.time())}" if smoke else f"{arm}-s42")
            if Path(name).name != name or name in {".", ".."}:
                parser.error("name must be a single directory name")
            directory = root / "runs" / name
            directory.mkdir(parents=True, exist_ok=False)
            checkpoint = root / "weights/initial.pt"
            if smoke:
                config.update(epochs=args.epochs or 1, workers=0, close_mosaic=0, plots=True)
            elif args.epochs:
                config["epochs"] = args.epochs
            if args.batch:
                config["batch"] = args.batch
            if args.workers is not None:
                config["workers"] = args.workers
            metadata = dict(
                arm=arm,
                smoke=smoke,
                config=config,
                fingerprint=fingerprint,
                initial_sha256=inventory["initial_sha256"],
                environment=environment,
                dataset_counts=inventory["counts"],
                effective_counts=inventory.get("effective_counts", inventory["counts"]),
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            (directory / "experiment.json").write_text(json.dumps(metadata, indent=2))
            frozen = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"], check=True, capture_output=True, text=True
            )
            (directory / "requirements.lock.txt").write_text(frozen.stdout)
        data = runtime_yaml(root, smoke=smoke)
        trainer = ModerateTrainer if arm == "moderate" else ControlTrainer
        if args.resume:
            overrides = dict(
                resume=str(checkpoint),
                data=str(data),
                device=args.device,
                project=str(directory.parent),
                name=directory.name,
                exist_ok=True,
            )
            if args.batch:
                overrides["batch"] = args.batch
            if args.workers is not None:
                overrides["workers"] = args.workers
        else:
            overrides = dict(
                config,
                data=str(data),
                device=args.device,
                project=str(directory.parent),
                name=directory.name,
                exist_ok=True,
            )
        started = time.perf_counter()
        model = YOLO(str(checkpoint))
        model.train(trainer=trainer, **overrides)
        result = dict(
            status="complete",
            arm=arm,
            smoke=smoke,
            seconds=time.perf_counter() - started,
            metrics={k: float(v) for k, v in model.trainer.metrics.items()},
            best=str(model.trainer.best),
            last=str(model.trainer.last),
        )
        (directory / "run_result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
