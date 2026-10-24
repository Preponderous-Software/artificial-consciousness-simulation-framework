"""Tests for persistence/paths.py — persistence root and name sanitization.

Covers:
- sanitize_consciousness_name() collapses unsafe runs, strips separators,
  and rejects names with nothing safe left
- consciousness_root() honours CONSCIOUSNESS_HOME and expands ``~``
- consciousness_dir() joins the root with the sanitized name
"""

from __future__ import annotations

from pathlib import Path

import pytest

from persistence.paths import consciousness_dir, consciousness_root, sanitize_consciousness_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Aria", "Aria"),
        ("  Aria  ", "Aria"),
        ("Aria!", "Aria"),
        ("a b/c", "a_b_c"),
        ("a..//b", "a_b"),
        ("-Aria-", "Aria"),
        ("_Aria_", "Aria"),
        (".hidden", "hidden"),
        ("a/../b", "a_b"),
    ],
)
def test_sanitize_collapses_unsafe_runs_and_strips_separators(raw: str, expected: str) -> None:
    assert sanitize_consciousness_name(raw) == expected


def test_sanitize_replaces_non_ascii_letters() -> None:
    """The safe set is ASCII-only, so an accented letter is unsafe like any other."""
    assert sanitize_consciousness_name("Ária") == "ria"
    assert sanitize_consciousness_name("Zoë") == "Zo"


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "...", "--", "/", "Ω"])
def test_sanitize_raises_when_nothing_safe_remains(raw: str) -> None:
    with pytest.raises(ValueError, match="at least one alphanumeric"):
        sanitize_consciousness_name(raw)


def test_sanitize_result_is_a_single_path_component() -> None:
    name = sanitize_consciousness_name("../../etc/passwd")
    assert name == "etc_passwd"
    assert Path(name).name == name


def test_consciousness_root_defaults_to_home_dot_consciousness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONSCIOUSNESS_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert consciousness_root() == tmp_path / ".consciousness"


def test_consciousness_root_honours_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "custom"))
    assert consciousness_root() == tmp_path / "custom"


def test_consciousness_root_expands_tilde_in_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CONSCIOUSNESS_HOME", "~/minds")
    assert consciousness_root() == tmp_path / "minds"


def test_consciousness_root_is_resolved_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var is read on every call, not captured at import time."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "one"))
    assert consciousness_root() == tmp_path / "one"
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "two"))
    assert consciousness_root() == tmp_path / "two"


def test_consciousness_dir_joins_root_with_sanitized_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    assert consciousness_dir("Aria") == tmp_path / "Aria"
    assert consciousness_dir("../escape") == tmp_path / "escape"
    assert consciousness_dir("my mind") == tmp_path / "my_mind"


def test_consciousness_dir_does_not_create_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    path = consciousness_dir("Aria")
    assert not path.exists()


def test_consciousness_dir_rejects_unsanitizable_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    with pytest.raises(ValueError):
        consciousness_dir("???")
