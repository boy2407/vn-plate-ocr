"""Model replies must become valid queue labels or fall back to unreadable."""

from custom_alpr.vlm_label import parse_answer


def test_clean_json_label_is_normalized():
    raw = '{"verdict": "label", "plate_text": "29-B1 123.45", "lines": 2}'
    assert parse_answer(raw) == dict(verdict="label", plate_text="29B112345", lines="2")


def test_json_inside_prose_or_code_fence_is_found():
    raw = 'Here you go:\n```json\n{"verdict": "not_plate", "plate_text": "", "lines": null}\n```'
    assert parse_answer(raw) == dict(verdict="not_plate", plate_text="", lines="")


def test_off_format_plate_text_is_not_a_label():
    raw = '{"verdict": "label", "plate_text": "TUYENDUNG", "lines": 1}'
    assert parse_answer(raw)["verdict"] == "unreadable"


def test_garbage_and_unknown_verdicts_are_unreadable():
    assert parse_answer("I cannot tell")["verdict"] == "unreadable"
    assert parse_answer('{"verdict": "maybe"}')["verdict"] == "unreadable"


def test_text_is_dropped_for_non_label_verdicts():
    raw = '{"verdict": "partial", "plate_text": "51G12", "lines": 2}'
    assert parse_answer(raw) == dict(verdict="partial", plate_text="", lines="2")
