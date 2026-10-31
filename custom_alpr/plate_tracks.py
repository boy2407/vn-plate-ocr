"""Associate plate observations in time, then summarize observed OCR strings."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from math import exp, hypot, log
import re

# Civil (incl. 2-letter series, motorbike "29B1", "MĐ1") and 2-letter military plates.
PLATE_FORMAT = re.compile(r"^(?:\d{2}[A-ZĐ]{1,2}\d?\d{4,5}|[A-Z]{2}\d{4,5})$")
VOTE_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZĐ"


def normalized_text(text: str) -> str:
    return "".join(char for char in text.upper() if char.isalnum())


def valid_plate(text: str) -> bool:
    return bool(PLATE_FORMAT.match(text))


def vote_weight(item: dict) -> float:
    """A sharp, confidently read crop counts more than a blurred one."""
    quality = item.get("crop_quality", {}).get("score")
    return item["ocr_confidence"] * (1.0 if quality is None else quality)


def representative_key(item: dict) -> tuple:
    confidence = item["ocr_confidence"]
    quality = item.get("crop_quality", {}).get("score")
    score = confidence if quality is None else 0.65 * quality + 0.35 * confidence
    return (score, confidence, item["width"] * item["height"], item["detection_confidence"])


def predicted_layout(text: str) -> str:
    return "".join("D" if char.isdigit() else "L" for char in normalized_text(text))


def per_character_vote(observations: list[dict], rejection_threshold: float | None = None) -> dict:
    """Vote per character within the dominant predicted layout using crop quality only.

    Each observation may carry ``char_probabilities`` as one vocabulary-probability
    vector per predicted position. The confidence is the minimum top-two posterior
    margin over positions, so one weak character can reject the whole track.
    """
    if not observations:
        raise ValueError("A character vote needs at least one observation")
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in observations:
        text = normalized_text(item["text"])
        layout = item.get("layout") or predicted_layout(text)
        if text and len(layout) == len(text):
            groups[layout].append(item)
    if not groups:
        return {"text": "unreadable", "confidence": 0.0, "layout": "", "accepted": False}
    layout = max(groups, key=lambda value: (len(groups[value]), value))
    selected = groups[layout]
    output, margins = [], []
    for position, kind in enumerate(layout):
        totals: dict[str, float] = defaultdict(float)
        for item in selected:
            probabilities = item.get("char_probabilities")
            if probabilities is None or position >= len(probabilities):
                continue
            weight = item.get("quality")
            if weight is None:
                weight = item.get("crop_quality", {}).get("score", 1.0)
            weight = max(float(weight or 0.0), 0.0)
            vector = probabilities[position]
            for index, probability in enumerate(vector[: len(VOTE_ALPHABET)]):
                char = VOTE_ALPHABET[index]
                if (kind == "D") != char.isdigit():
                    continue
                totals[char] += weight * log(max(float(probability), 1e-12))
        if not totals:
            # Old observations remain usable, but have no calibrated character evidence.
            text = max(selected, key=vote_weight)["text"]
            candidate = normalized_text(text)
            accepted = rejection_threshold is None
            return {
                "text": candidate if accepted else "unreadable",
                "candidate_text": candidate,
                "confidence": 0.0,
                "layout": layout,
                "accepted": accepted,
            }
        ranked = sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))
        output.append(ranked[0][0])
        normalization = sum(exp(score - ranked[0][1]) for _, score in ranked)
        margins.append(
            (exp(ranked[0][1] - ranked[0][1]) - exp(ranked[1][1] - ranked[0][1])) / normalization
            if len(ranked) > 1
            else 1.0
        )
    confidence = min(margins, default=0.0)
    text = "".join(output)
    accepted = rejection_threshold is None or confidence >= rejection_threshold
    return {
        "text": text if accepted else "unreadable",
        "candidate_text": text,
        "confidence": confidence,
        "layout": layout,
        "accepted": accepted,
        "observations": len(selected),
    }


def association_score(a: list, b: list) -> float:
    aw, ah = a[2] - a[0], a[3] - a[1]
    bw, bh = b[2] - b[0], b[3] - b[1]
    if min(aw, ah, bw, bh) <= 0:
        return 0.0
    size_ratio = max(aw / bw, bw / aw, ah / bh, bh / ah)
    if size_ratio > 2.5:
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    iou = intersection / (aw * ah + bw * bh - intersection)
    distance = hypot(
        (a[0] + a[2] - b[0] - b[2]) / (aw + bw), (a[1] + a[3] - b[1] - b[3]) / (ah + bh)
    )
    if iou >= 0.2 or distance <= 0.5:
        return iou + 0.25 / (1 + distance)
    return 0.0


@dataclass
class PlateTrack:
    track_id: int
    box: list
    first_time: float
    last_time: float
    observations: int = 0
    counts: dict = field(default_factory=lambda: defaultdict(int))
    weights: dict = field(default_factory=lambda: defaultdict(float))
    best: dict = field(default_factory=dict)
    items: list[dict] = field(default_factory=list)

    def add(self, item: dict):
        text = normalized_text(item["text"])
        item["track_id"] = self.track_id
        self.observations += 1
        self.items.append(item.copy())
        self.counts[text] += 1
        self.weights[text] += vote_weight(item)
        if text not in self.best or representative_key(item) > representative_key(self.best[text]):
            self.best[text] = item
        self.box = item["box"]
        self.last_time = item["time_seconds"]

    def summary(
        self, *, vote_strategy: str = "string", rejection_threshold: float | None = None
    ) -> dict:
        """Vote over the whole track: well-formed plates first, then quality-weighted votes."""
        if vote_strategy == "per_character":
            voted = per_character_vote(self.items, rejection_threshold)
            item = max(self.items, key=representative_key).copy()
            item.update(
                text=voted["text"],
                per_character_candidate_text=voted.get("candidate_text", voted["text"]),
                per_character_confidence=voted["confidence"],
                vote_strategy=vote_strategy,
                plate_format_valid=valid_plate(voted.get("candidate_text", voted["text"])),
                observations=self.observations,
                first_time=self.first_time,
                last_time=self.last_time,
                track_ids=[self.track_id],
                track_count=1,
            )
            return item
        if vote_strategy != "string":
            raise ValueError("vote_strategy must be 'string' or 'per_character'")
        text = max(
            self.counts,
            key=lambda value: (valid_plate(value), self.weights[value], self.counts[value]),
        )
        item = self.best[text].copy()
        total = sum(self.weights.values())
        item.update(
            plate_format_valid=valid_plate(text),
            text_weight_share=self.weights[text] / total if total else 0.0,
            observations=self.observations,
            first_time=self.first_time,
            last_time=self.last_time,
            text_votes=self.counts[text],
            text_agreement=self.counts[text] / self.observations,
            track_ids=[self.track_id],
            track_count=1,
        )
        return item


class PlateTracker:
    """Light spatial association of plate boxes; each track takes one box per frame.

    A gap over one second starts another track. This is an OCR review aid, not
    vehicle identity tracking. Exact consensus texts are merged for the gallery;
    individual track IDs and all frame observations remain available separately.
    """

    def __init__(self, max_gap: float = 1.0):
        self.max_gap = max_gap
        self.tracks: list[PlateTrack] = []
        self.active: list[PlateTrack] = []
        self.by_yolo_id: dict[int, PlateTrack] = {}

    def observe(self, item: dict):
        return self.observe_frame([item])[0]

    def observe_frame(self, items: list[dict]) -> list[dict]:
        """Associate all boxes of one frame; a track takes at most one box per frame."""
        if not items:
            return []
        now = items[0]["time_seconds"]
        self.active = [track for track in self.active if 0 <= now - track.last_time <= self.max_gap]
        assigned, used = {}, set()
        # ByteTrack IDs from YOLO follow motion across gaps; trust them first.
        for index, item in enumerate(items):
            track = self.by_yolo_id.get(item.get("yolo_track_id"))
            if track is not None and track.track_id not in used:
                assigned[index] = track
                used.add(track.track_id)
                if track not in self.active:
                    self.active.append(track)
        pairs = sorted(
            (
                (association_score(track.box, item["box"]), index, track)
                for index, item in enumerate(items)
                if index not in assigned
                for track in self.active
                if track.track_id not in used
            ),
            key=lambda pair: (-pair[0], pair[1], pair[2].track_id),
        )
        for score, index, track in pairs:
            if score <= 0:
                break
            if index not in assigned and track.track_id not in used:
                assigned[index] = track
                used.add(track.track_id)
        summaries = []
        for index, item in enumerate(items):
            track = assigned.get(index)
            if track is None:
                track = PlateTrack(len(self.tracks) + 1, item["box"], now, now)
                self.tracks.append(track)
                self.active.append(track)
            if item.get("yolo_track_id") is not None:
                self.by_yolo_id[item["yolo_track_id"]] = track
            track.add(item)
            summaries.append(track.summary())
        return summaries

    def summaries(
        self, *, vote_strategy: str = "string", rejection_threshold: float | None = None
    ) -> list[dict]:
        if vote_strategy == "per_character":
            return [
                track.summary(vote_strategy=vote_strategy, rejection_threshold=rejection_threshold)
                for track in self.tracks
            ]
        grouped = {}
        for track in self.tracks:
            item = track.summary(vote_strategy=vote_strategy)
            text = normalized_text(item["text"])
            key = text or f"empty-track-{track.track_id}"
            if key not in grouped:
                grouped[key] = item
                continue
            previous = grouped[key]
            clearer = representative_key(item) > representative_key(previous)
            best = (item if clearer else previous).copy()
            best.update(
                observations=previous["observations"] + item["observations"],
                first_time=min(previous["first_time"], item["first_time"]),
                last_time=max(previous["last_time"], item["last_time"]),
                track_ids=previous["track_ids"] + item["track_ids"],
                track_count=previous["track_count"] + item["track_count"],
                text_votes=previous["text_votes"] + item["text_votes"],
            )
            best["text_agreement"] = best["text_votes"] / best["observations"]
            grouped[key] = best
        return sorted(grouped.values(), key=lambda item: item["first_time"])


def group_observations(
    observations: list[dict],
    *,
    vote_strategy: str = "string",
    rejection_threshold: float | None = None,
) -> list[dict]:
    tracker = PlateTracker()
    frames = defaultdict(list)
    for item in observations:
        frames[(item["time_seconds"], item["frame"])].append(item)
    for key in sorted(frames):
        tracker.observe_frame(frames[key])
    return tracker.summaries(vote_strategy=vote_strategy, rejection_threshold=rejection_threshold)
