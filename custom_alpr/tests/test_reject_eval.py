import torch
import pytest

from custom_alpr.reject_eval import (
    choose_threshold,
    risk_coverage,
    score_positions,
    summarize_threshold,
    wilson,
)


def test_score_positions_uses_selected_char_probabilities_and_agreement():
    probabilities = torch.full((1, 3, 40), 0.001)
    probabilities[0, 0, 3] = 0.8  # 0
    probabilities[0, 1, 4] = 0.7  # 1
    probabilities[0, 2, 5] = 0.6  # 2
    scores = score_positions(probabilities, "012", "012")
    assert scores["minimum_probability"] == pytest.approx(0.6)
    assert scores["geometric_mean"] > scores["minimum_probability"]
    assert scores["free_layout_agreement"] == 1.0
    assert scores["weakest_position_margin"] > 0


def test_threshold_selection_and_wilson_bounds():
    rows = [
        {"score": 0.9, "correct": True},
        {"score": 0.8, "correct": True},
        {"score": 0.2, "correct": False},
    ]
    assert choose_threshold(rows, "score", 0.98) == 0.8
    summary = summarize_threshold(rows, "score", 0.8)
    assert summary["accepted"] == 2
    assert summary["precision"] == 1.0
    assert summary["precision_ci"][0] < 1.0
    assert wilson(0, 0) == [0.0, 1.0]


def test_risk_coverage_has_one_point_per_distinct_threshold():
    rows = [
        {"score": 0.9, "correct": True},
        {"score": 0.8, "correct": False},
        {"score": 0.8, "correct": True},
    ]
    curve = risk_coverage(rows, "score")
    assert len(curve) == 2
    assert curve[0]["coverage"] == pytest.approx(1 / 3)
    assert curve[-1]["accuracy"] == pytest.approx(2 / 3)
