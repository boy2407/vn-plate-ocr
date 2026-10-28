"""Check shared detector evaluation matching and counts."""

import numpy as np

from custom_alpr.compare_detectors import match_boxes, summarize


def test_detector_comparison_matches_duplicates_once_and_keeps_missing_targets():
    truth = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    predictions = np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float)
    matched = match_boxes(predictions, truth)
    assert matched[0].all() and not matched[1].any()
    result = summarize([dict(boxes=predictions.tolist(), truth=truth.tolist(), scores=[0.9, 0.8])])
    counts = result["operating_points"]["0.4"]
    assert (counts["tp"], counts["fp"], counts["fn"]) == (1, 1, 1)
    assert counts["precision"] == counts["recall"] == 0.5


def test_detector_comparison_counts_no_predictions_as_misses():
    result = summarize([dict(boxes=[], scores=[], truth=[[0, 0, 10, 10]])])
    counts = result["operating_points"]["0.4"]
    assert (counts["tp"], counts["fp"], counts["fn"]) == (0, 0, 1)
    assert result["mAP50"] == 0
