"""Tests for scripts/_logging.py — the per-instance rotating-file log setup
shared by spawn.py and resume.py.

Covers:
- the returned path (``<CONSCIOUSNESS_HOME>/<sanitized name>/run.log``) and
  creation of the instance directory.
- the handler installed on the root logger: a RotatingFileHandler with the
  2 MiB / 3-backup / utf-8 settings, and the ``asctime level name: message``
  line format.
- the root level: case-insensitive level names, and a fallback to WARNING for
  an unrecognised name rather than an exception.

``configure_logging`` mutates the process-wide root logger, so every test
snapshots and restores the root logger's handlers and level.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

import pytest

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._logging import configure_logging  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_root_logger():
    """configure_logging() adds a handler to the root logger and changes its
    level; undo both so the rotating file handler does not leak into (or hold
    a file open across) other tests."""
    root = logging.root
    handlers_before = list(root.handlers)
    level_before = root.level
    yield
    # Only the rotating file handlers configure_logging() added are removed;
    # pytest attaches its own per-phase capture handlers to the root logger
    # and tears those down itself.
    for handler in _new_file_handlers(handlers_before):
        root.removeHandler(handler)
        handler.close()
    root.setLevel(level_before)


def _new_file_handlers(before: list[logging.Handler]) -> list[logging.handlers.RotatingFileHandler]:
    return [
        h for h in logging.root.handlers
        if h not in before and isinstance(h, logging.handlers.RotatingFileHandler)
    ]


# --- returned path ----------------------------------------------------------


def test_configure_logging_returns_run_log_under_instance_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    log_path = configure_logging("Aria", "INFO")

    assert log_path == tmp_path / "Aria" / "run.log"
    assert log_path.parent.is_dir(), "the instance directory is created on demand"


def test_configure_logging_sanitizes_instance_name(monkeypatch, tmp_path) -> None:
    """The log lands in the same directory spawn.py persists to, so the name
    is sanitized the same way (spaces collapse to underscores)."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    log_path = configure_logging("Aria 1", "INFO")

    assert log_path == tmp_path / "Aria_1" / "run.log"


# --- installed handler -------------------------------------------------------


def test_configure_logging_installs_rotating_file_handler_on_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    before = list(logging.root.handlers)

    log_path = configure_logging("Aria", "INFO")

    added = _new_file_handlers(before)
    assert len(added) == 1, f"expected exactly one new RotatingFileHandler, got {added}"
    handler = added[0]
    assert Path(handler.baseFilename) == log_path
    assert handler.maxBytes == 2 * 1024 * 1024
    assert handler.backupCount == 3
    assert handler.encoding == "utf-8"


def test_configure_logging_writes_formatted_records_to_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    log_path = configure_logging("Aria", "INFO")
    logging.getLogger("tests.logging_config").info("cycle %d complete", 7)
    for handler in logging.root.handlers:
        handler.flush()

    line = log_path.read_text(encoding="utf-8").strip()
    # "%(asctime)s %(levelname)s %(name)s: %(message)s"
    assert line.endswith(" INFO tests.logging_config: cycle 7 complete")


def test_configure_logging_called_twice_stacks_handlers(monkeypatch, tmp_path) -> None:
    """Characterizes current behaviour: each call appends another handler
    rather than replacing the previous one, so a second call for the same
    instance would write every record twice."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    before = list(logging.root.handlers)

    configure_logging("Aria", "INFO")
    configure_logging("Aria", "INFO")

    assert len(_new_file_handlers(before)) == 2


# --- root level -------------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("Info", logging.INFO),
        ("WARNING", logging.WARNING),
        ("error", logging.ERROR),
    ],
)
def test_configure_logging_sets_root_level_case_insensitively(monkeypatch, tmp_path, level, expected) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    configure_logging("Aria", level)

    assert logging.root.level == expected


def test_configure_logging_unknown_level_falls_back_to_warning(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    configure_logging("Aria", "VERBOSE")

    assert logging.root.level == logging.WARNING


def test_configure_logging_level_filters_records_below_threshold(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))

    log_path = configure_logging("Aria", "WARNING")
    logging.getLogger("tests.logging_config").info("filtered out")
    logging.getLogger("tests.logging_config").warning("kept")
    for handler in logging.root.handlers:
        handler.flush()

    text = log_path.read_text(encoding="utf-8")
    assert "filtered out" not in text
    assert "kept" in text
