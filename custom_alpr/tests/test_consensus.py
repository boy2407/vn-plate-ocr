"""A plate label is accepted only when Codex and the VLM read the same text."""

import pytest

from custom_alpr.consensus import decide


def codex(verdict="label", text="59L287956"):
    return dict(verdict=verdict, plate_text=text)


def vlm(verdict="label", text="59L287956"):
    return dict(verdict=verdict, plate_text=text)


@pytest.mark.parametrize(
    ("codex_row", "vlm_row", "status"),
    [
        (codex(), vlm(), "agree"),
        (codex(text="59L287456"), vlm(), "person"),  # the digit Codex misread
        (codex(), vlm("unreadable", ""), "person"),
        (codex("unreadable", ""), vlm(), "person"),  # VLM may see what Codex missed
        (codex("not_plate", ""), vlm("unreadable", ""), "codex"),
        (codex("unreadable", ""), vlm("label", "78A1234"), "codex"),  # template hallucination
        (codex("label", "78A1234"), vlm("label", "78A1234"), "person"),  # template never auto-accepted
        (codex("", ""), vlm(), "pending"),
        (codex(), None, "person"),
    ],
)
def test_decide(codex_row, vlm_row, status):
    assert decide(codex_row, vlm_row) == status
