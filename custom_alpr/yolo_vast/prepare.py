"""Build an immutable training bundle from the existing detector split."""

import argparse
import json
import os
import shutil
from pathlib import Path

from custom_alpr.yolo_vast.data import audit_dataset, sha256, training_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-empty-labels",
        action="store_true",
        help="Use only after reviewing that empty-label frames are truly plate-free",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory")
    inventory = training_policy(audit_dataset(args.source), args.include_empty_labels)
    args.output.mkdir(parents=True)
    for row in inventory["records"]:
        for kind in ("image", "label"):
            source, target = args.source / row[kind], args.output / "data" / row[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
    for name in ("manifest.json", "report.json"):
        shutil.copy2(args.source / name, args.output / "data" / name)
    (args.output / "weights").mkdir()
    shutil.copy2(args.checkpoint, args.output / "weights/initial.pt")
    (args.output / "negative-quarantine.json").write_text(
        json.dumps(
            {"images": inventory["train_exclude"], "reason": inventory["exclusion_reason"]},
            indent=2,
        )
    )
    inventory["initial_sha256"] = sha256(args.output / "weights/initial.pt")
    inventory["note"] = "Existing dataset only. Optional clip-derived negatives are kept separate."
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    target = args.output / "custom_alpr/yolo_vast"
    shutil.copytree(Path(__file__).parent, target, ignore=shutil.ignore_patterns("__pycache__"))
    (args.output / "custom_alpr/__init__.py").write_text('"""Portable training bundle."""\n')
    tests = args.output / "tests"
    tests.mkdir()
    shutil.copy2(Path(__file__).resolve().parents[1] / "tests/test_yolo_vast.py", tests)
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        shutil.copy2(readme, args.output / "README.md")
    print(json.dumps(inventory["effective_counts"], indent=2), flush=True)
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
