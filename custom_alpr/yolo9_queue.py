"""Wait for prior GPU work; run validated smoke, fine-tune and final evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from custom_alpr.yolo_vast.wait_for import gate_state, gpu_processes, process_alive


def main():  # noqa: PLR0915 -- explicit staged job with evidence checks
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--prior-pid", type=int, required=True)
    parser.add_argument("--prior-log", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    status_file = root / "queue_status.json"

    def status(state, **fields):
        data = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **fields)
        tmp = status_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(status_file)
        print(json.dumps(data), flush=True)

    def run(module, *extra):
        subprocess.run(
            [
                str(root / ".venv/bin/python"),
                "-u",
                "-m",
                module,
                "--root",
                str(root),
                "--dataset",
                str(args.dataset),
                *extra,
            ],
            check=True,
            cwd=root,
        )

    try:
        cpu = json.loads((root / "runs/smoke-cpu/run_result.json").read_text())
        if cpu["status"] != "complete" or cpu["results"][2] < 0.1:
            raise ValueError("CPU smoke incomplete or detector collapsed; refusing full train")
        previous, idle_since = None, None
        while True:
            alive = process_alive(args.prior_pid, "run-20261028-08-parseq-e-all")
            log = args.prior_log.read_text(errors="replace") if args.prior_log.exists() else ""
            pids = gpu_processes()
            if alive or pids:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            idle = 0 if idle_since is None else time.monotonic() - idle_since
            state = gate_state(alive, log, pids, idle, required_idle=30)
            if state != previous:
                status(state, gpu_pids=pids)
                previous = state
            if state == "prior_run_incomplete":
                raise ValueError("Prior OCR process ended without completed test markers")
            if state == "ready":
                break
            time.sleep(10)
        status("gpu_smoke")
        run("custom_alpr.yolo9_vast", "--stage", "smoke", "--device", "0")
        smoke = json.loads((root / "runs/smoke-0/run_result.json").read_text())
        if smoke["status"] != "complete" or smoke["results"][2] < 0.1:
            raise ValueError("GPU smoke incomplete or collapsed")
        if gpu_processes():
            raise ValueError("Another GPU job started before fine-tune; stopped safely")
        status("training", epochs=30, imgsz=384)
        run("custom_alpr.yolo9_vast", "--stage", "train", "--device", "0")
        status("exporting_and_testing")
        run("custom_alpr.yolo9_finish")
        status("finished")
    except Exception as exc:
        status("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
