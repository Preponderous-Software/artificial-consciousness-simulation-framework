"""Tests for experiments/report.py — markdown table rendering.

`_fmt_table` stringifies every cell, so callers legitimately pass rows whose
elements are not all `str` (vocabulary rows lead with an int rank). These
tests pin that mixed-element contract, which the parameter annotation now
states explicitly (#11).
"""

from __future__ import annotations

from experiments.report import _fmt_mood, _fmt_table


def test_fmt_table_renders_header_separator_and_rows() -> None:
    rendered = _fmt_table([("a", "b")], ["Left", "Right"])
    assert rendered.splitlines() == [
        "| Left | Right |",
        "|---|---|",
        "| a | b |",
    ]


def test_fmt_table_stringifies_mixed_element_rows() -> None:
    """Vocabulary rows are `(int, str, int)`; ranks must render as text."""
    rendered = _fmt_table([(1, "memory", 12), (2, "self", 9)], ["#", "Word", "Count"])
    assert "| 1 | memory | 12 |" in rendered
    assert "| 2 | self | 9 |" in rendered


def test_fmt_mood_reports_missing_data() -> None:
    assert _fmt_mood({}) == "(no mood data)"
    assert _fmt_mood({"curiosity": 0.712}) == "`curiosity=0.71`"
