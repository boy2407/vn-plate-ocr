"""Merge Codex and VLM labels; a person decides every plate they disagree on.

Rules, per crop that Codex has labelled:
- both read a plate and the texts match: accepted;
- either reads a plate and they differ: goes to ``disagreements.csv`` for a person;
- neither reads a plate (unreadable / partial / not_plate): Codex's verdict stands.
VLM readings of the template ``...1234`` are hallucinations and count as unreadable.

Re-running keeps what the person already filled in ``disagreements.csv`` and
adds new rows as Codex progresses. ``queue-final.csv`` holds the merged labels
in the queue format for ``ingest_labels --queue``.

    python -m custom_alpr.consensus
    python -m custom_alpr.consensus --confirmed-only   # plates only from the person
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path

from custom_alpr.plate_tracks import normalized_text

ROOT = Path(__file__).resolve().parents[1]
COLLECTED = ROOT / "custom_alpr/data/collected"
TEMPLATE = re.compile(r"12345?$")  # VLM filler for plates it cannot read: 78A1234, 76AB1234...
DISAGREEMENT_FIELDS = [
    "crop_id", "view_path", "codex_verdict", "codex_text", "vlm_verdict", "vlm_text",
    "verdict", "plate_text", "lines", "note",
]  # fmt: skip


def vlm_reading(row: dict) -> tuple[str, str]:
    verdict, text = row.get("verdict", ""), normalized_text(row.get("plate_text", ""))
    if verdict == "label" and TEMPLATE.search(text):
        return "unreadable", ""
    return verdict, text if verdict == "label" else ""


def decide(codex: dict, vlm: dict | None) -> str:
    """``pending``, ``agree``, ``person`` or ``codex`` for one queue row."""
    codex_verdict = codex["verdict"].strip()
    if not codex_verdict:
        return "pending"
    vlm_verdict, vlm_text = vlm_reading(vlm or {})
    codex_text = normalized_text(codex["plate_text"]) if codex_verdict == "label" else ""
    if codex_verdict == "label" and vlm_verdict == "label" and codex_text == vlm_text:
        return "agree"
    if "label" in (codex_verdict, vlm_verdict):
        return "person"
    return "codex"


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--collected", type=Path, default=COLLECTED)
    parser.add_argument(
        "--confirmed-only", action="store_true", help="Leave Codex/VLM agreements out; keep only plates a person labelled"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = read_rows(args.collected / "queue.csv")
    vlm = {row["crop_id"]: row for row in read_rows(args.collected / "vlm-labels.csv")}
    disagreements_path = args.collected / "disagreements.csv"
    resolved = {row["crop_id"]: row for row in read_rows(disagreements_path)}

    statuses, disagreements, final = Counter(), [], []
    for row in queue:
        status = decide(row, vlm.get(row["crop_id"]))
        statuses[status] += 1
        merged = dict(row)
        if status == "person":
            vlm_verdict, vlm_text = vlm_reading(vlm.get(row["crop_id"], {}))
            previous = resolved.get(row["crop_id"], {})
            disagreements.append(
                dict(
                    crop_id=row["crop_id"],
                    view_path=row["image_path"].replace("/crops/", "/view/"),
                    codex_verdict=row["verdict"],
                    codex_text=row["plate_text"],
                    vlm_verdict=vlm_verdict,
                    vlm_text=vlm_text,
                    **{key: previous.get(key, "") for key in ("verdict", "plate_text", "lines", "note")},
                )
            )
            # Only the person's decision is imported; undecided rows stay out.
            merged.update({key: previous.get(key, "") for key in ("verdict", "plate_text", "lines", "note")})
        elif status == "pending" or (status == "agree" and args.confirmed_only):
            continue
        final.append(merged)

    write_rows(disagreements_path, DISAGREEMENT_FIELDS, disagreements)
    write_rows(args.collected / "queue-final.csv", list(queue[0]), final)
    cells = "\n".join(
        f'<figure><img src="{html.escape(row["view_path"])}"><figcaption>'
        f'<b>{html.escape(row["crop_id"])}</b><br>Codex: {html.escape(row["codex_verdict"])} '
        f'{html.escape(row["codex_text"])}<br>VLM: {html.escape(row["vlm_verdict"])} '
        f'{html.escape(row["vlm_text"])}<br>Bạn: {html.escape(row["verdict"])} {html.escape(row["plate_text"])}'
        f"</figcaption></figure>"
        for row in disagreements
    )
    (args.collected / "disagreements.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Disagreements</title>'
        "<style>body{font:14px sans-serif;display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;width:300px}img{height:150px;image-rendering:pixelated}</style>\n" + cells,
        encoding="utf-8",
    )
    decided = sum(1 for row in disagreements if row["verdict"].strip())
    print(
        f"labelled by Codex: {len(queue) - statuses['pending']}  agree: {statuses['agree']}  "
        f"codex-only (no plate text): {statuses['codex']}  for a person: {statuses['person']} "
        f"({decided} decided)  pending: {statuses['pending']}"
    )
    print(f"wrote {disagreements_path}, disagreements.html, queue-final.csv")


if __name__ == "__main__":
    main()
