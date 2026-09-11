"""Tests for scripts/inspect.py — read-only journal tail for a named instance.

Covers:
- output format (``<timestamp> [<type>] <content>``, one event per line) and
  chronological ordering.
- the default ``--limit`` of 20 and an explicit ``--limit``.
- a missing journal (never-run or never-persisted instance) printing nothing
  and exiting 0 rather than failing.
- corrupted / non-object journal lines being skipped so the readable tail is
  still shown (Journal.recent's contract, exercised through the CLI).

The command only reads ``journal.jsonl`` from ``CONSCIOUSNESS_HOME``; no run
loop, provider, or network is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.inspect import main as inspect_main  # noqa: E402


def _event(i: int, kind: str = "thought") -> dict[str, str]:
    return {
        "timestamp": f"2026-09-01T00:00:{i:02d}+00:00",
        "type": kind,
        "content": f"event {i}",
    }


def _write_journal(home: Path, name: str, lines: list[str]) -> Path:
    agent_dir = home / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    path = agent_dir / "journal.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def _output_lines(stdout: str) -> list[str]:
    # stdout only: Journal.recent logs a WARNING per skipped line, and click's
    # CliRunner folds stderr into ``result.output`` — keep the two apart.
    return [line for line in stdout.splitlines() if line]


# --- missing journal --------------------------------------------------------


def test_inspect_prints_nothing_for_missing_journal(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    result = CliRunner().invoke(inspect_main, ["--name", "NeverRan"])

    assert result.exit_code == 0
    assert result.exception is None
    assert result.stdout == ""


# --- formatting and ordering ------------------------------------------------


def test_inspect_prints_one_formatted_line_per_event_in_order(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(
        tmp_path,
        "Aria",
        [json.dumps(_event(0)), json.dumps(_event(1, "reflection")), json.dumps(_event(2, "perception"))],
    )

    result = CliRunner().invoke(inspect_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert _output_lines(result.stdout) == [
        "2026-09-01T00:00:00+00:00 [thought] event 0",
        "2026-09-01T00:00:01+00:00 [reflection] event 1",
        "2026-09-01T00:00:02+00:00 [perception] event 2",
    ]


def test_inspect_resolves_journal_through_sanitized_name(monkeypatch, tmp_path) -> None:
    """The journal path goes through consciousness_dir(), so the same
    sanitization spawn.py applied at write time is applied at read time."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(tmp_path, "Aria_1", [json.dumps(_event(0))])

    result = CliRunner().invoke(inspect_main, ["--name", "Aria 1"])

    assert result.exit_code == 0
    assert _output_lines(result.stdout) == ["2026-09-01T00:00:00+00:00 [thought] event 0"]


# --- --limit ----------------------------------------------------------------


def test_inspect_default_limit_is_last_20_events(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(tmp_path, "Aria", [json.dumps(_event(i)) for i in range(25)])

    result = CliRunner().invoke(inspect_main, ["--name", "Aria"])

    lines = _output_lines(result.stdout)
    assert len(lines) == 20
    assert lines[0].endswith("event 5")
    assert lines[-1].endswith("event 24")


def test_inspect_explicit_limit_keeps_most_recent_events(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(tmp_path, "Aria", [json.dumps(_event(i)) for i in range(10)])

    result = CliRunner().invoke(inspect_main, ["--name", "Aria", "--limit", "3"])

    assert result.exit_code == 0
    assert _output_lines(result.stdout) == [
        "2026-09-01T00:00:07+00:00 [thought] event 7",
        "2026-09-01T00:00:08+00:00 [thought] event 8",
        "2026-09-01T00:00:09+00:00 [thought] event 9",
    ]


def test_inspect_limit_zero_prints_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(tmp_path, "Aria", [json.dumps(_event(i)) for i in range(3)])

    result = CliRunner().invoke(inspect_main, ["--name", "Aria", "--limit", "0"])

    assert result.exit_code == 0
    assert result.stdout == ""


# --- corrupted journal lines -------------------------------------------------


def test_inspect_skips_corrupted_and_non_object_lines(monkeypatch, tmp_path) -> None:
    """Inspection is most useful on an instance that has misbehaved, so a bad
    line in the middle of the journal must not hide the readable events
    around it (#108 / #175)."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    _write_journal(
        tmp_path,
        "Aria",
        [
            json.dumps(_event(0)),
            "NOT JSON",
            "",
            "[1, 2]",
            "123",
            "null",
            json.dumps(_event(1)),
        ],
    )

    result = CliRunner().invoke(inspect_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert _output_lines(result.stdout) == [
        "2026-09-01T00:00:00+00:00 [thought] event 0",
        "2026-09-01T00:00:01+00:00 [thought] event 1",
    ]
