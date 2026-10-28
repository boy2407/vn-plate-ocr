"""Resumable rsync transport; credentials stay in SSH agent/config."""

import argparse
import re
import subprocess
from pathlib import Path

from custom_alpr.yolo_vast.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("push", "pull"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", default="/workspace/vn-plate-yolo")
    parser.add_argument("--archive", type=Path, help="dataset.tar for push")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9.-]+", args.host) or not re.fullmatch(
        r"[A-Za-z0-9_-]+", args.user
    ):
        parser.error("Invalid SSH host/user")
    if not 1 <= args.port <= 65535 or not re.fullmatch(r"/[A-Za-z0-9/_-]+", args.remote_root):
        parser.error("Invalid port or remote-root")
    bundle = args.bundle.resolve()
    ssh = [
        "ssh",
        "-p",
        str(args.port),
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "StrictHostKeyChecking=accept-new",
    ]
    target = f"{args.user}@{args.host}"
    transport = " ".join(ssh)
    common = ["rsync", "-a", "--partial", "--progress", "-e", transport]
    if args.action == "pull":
        destination = bundle / "remote-results"
        destination.mkdir(exist_ok=True)
        subprocess.run(
            [*common, f"{target}:{args.remote_root}/runs/", str(destination) + "/"], check=True
        )
        return
    if args.archive is None or not args.archive.is_file():
        parser.error("push requires --archive dataset.tar")
    subprocess.run([*ssh, target, f"mkdir -p {args.remote_root}/data"], check=True)
    excluded = [
        "data/",
        "runs/",
        "remote-results/",
        "logs/",
        "previews/",
        "__pycache__/",
        "*.runtime.yaml",
        "*.runtime.txt",
        ".venv/",
        "training.lock",
        "preflight.json",
        "smoke-*.txt",
    ]
    flags = [value for pattern in excluded for value in ("--exclude", pattern)]
    subprocess.run(
        [*common, *flags, str(bundle) + "/", f"{target}:{args.remote_root}/"], check=True
    )
    subprocess.run(
        [*common, str(args.archive), f"{target}:{args.remote_root}/dataset.tar"], check=True
    )
    digest = sha256(args.archive)
    command = (
        f"cd {args.remote_root} && "
        f"printf '%s  dataset.tar\\n' {digest} | sha256sum -c - && "
        "tar --warning=no-unknown-keyword --no-same-owner "
        "--no-same-permissions -xf dataset.tar -C data"
    )
    subprocess.run([*ssh, target, command], check=True)
    print("Transfer and archive SHA-256 check complete")


if __name__ == "__main__":
    main()
