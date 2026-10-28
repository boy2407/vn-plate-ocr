"""Wait for the prior OCR run AND its held-out test before touching the GPU."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_MARKERS = ("Held-out test (greedy autoregressive):", "Best checkpoint:")


def gate_state(prior_alive, log_text, gpu_pids, idle_seconds, required_idle=60):
    if prior_alive:
        return "waiting_prior_run"
    if not all(marker in log_text for marker in REQUIRED_MARKERS):
        return "prior_run_incomplete"
    if gpu_pids:
        return "waiting_other_gpu_processes"
    if idle_seconds < required_idle:
        return "waiting_idle_window"
    return "ready"


def process_alive(pid, expected):
    try:
        return expected.encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
    except FileNotFoundError:
        return False


def gpu_processes():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def main():  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior-pid", type=int, required=True)
    parser.add_argument("--prior-match", required=True)
    parser.add_argument("--prior-log", type=Path, required=True)
    parser.add_argument("--quiet-seconds", type=int, default=60)
    parser.add_argument("--stage", choices=("pair", "moderate", "control"), default="pair")
    args = parser.parse_args()
    if args.quiet_seconds < 10:
        parser.error("quiet window must be at least 10 seconds")
    root = args.root.resolve()
    status_path = root / "queue_status.json"

    def status(state, **extra):
        data = dict(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2))
        temporary.replace(status_path)
        print(json.dumps(data), flush=True)

    idle_since, previous = None, None
    try:
        while True:
            alive = process_alive(args.prior_pid, args.prior_match)
            text = args.prior_log.read_text(errors="replace") if args.prior_log.exists() else ""
            pids = gpu_processes()
            if alive or pids:
                idle_since = None
            elif idle_since is None:
                idle_since = time.monotonic()
            idle = 0 if idle_since is None else time.monotonic() - idle_since
            state = gate_state(alive, text, pids, idle, args.quiet_seconds)
            if state != previous:
                status(state, gpu_pids=pids, prior_pid=args.prior_pid)
                previous = state
            if state == "prior_run_incomplete":
                raise RuntimeError(
                    "Prior process ended without completed test markers; YOLO not started"
                )
            if state == "ready":
                break
            time.sleep(10)
        scripts = root / "custom_alpr/yolo_vast/scripts"
        status("setting_up_isolated_environment")
        subprocess.run(["bash", str(scripts / "setup.sh")], check=True, cwd=root)
        if gpu_processes():
            raise RuntimeError("A new GPU process started during setup; recheck before launching")
        status("gpu_smoke")
        subprocess.run(["bash", str(scripts / "run_stage.sh"), "smoke"], check=True, cwd=root)
        if gpu_processes():
            raise RuntimeError("A new GPU process started after smoke; recheck before launching")
        status("training", stage=args.stage)
        subprocess.run(["bash", str(scripts / "run_stage.sh"), args.stage], check=True, cwd=root)
        status("finished", stage=args.stage)
    except Exception as exc:
        status("blocked", error=str(exc))
        raise


if __name__ == "__main__":
    main()
