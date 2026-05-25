"""Tests for scripts/spawn.py helpers.

Currently focused on _build_config_path's tempfile-leak regression (#105) —
overrides must land in the consciousness dir, not a /tmp file that nobody
ever cleans up.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import yaml

# Make scripts/ importable the same way spawn.py does at runtime.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.spawn import _build_config_path  # noqa: E402


def test_build_config_path_returns_default_when_no_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider=None, model=None)
    # Default config — same path returned without overrides
    assert result.name == "default_consciousness.yaml"
    assert result.exists()


def test_build_config_path_with_overrides_writes_to_consciousness_dir(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider="mock", model="mock")

    expected = tmp_path / "Aria" / "_spawn_config.yaml"
    assert result == expected, f"override config should land in {expected}, got {result}"
    assert result.exists()

    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "mock"
    assert cfg["llm"]["model"] == "mock"


def test_build_config_path_does_not_leak_tmp_files(monkeypatch, tmp_path) -> None:
    """Regression for #105 — the prior implementation left a /tmp file per spawn."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    before = set(glob.glob("/tmp/consciousness_override_*.yaml"))
    for _ in range(3):
        _build_config_path("Aria", provider="mock", model="mock")
    after = set(glob.glob("/tmp/consciousness_override_*.yaml"))
    new_files = after - before
    assert not new_files, f"_build_config_path leaked tmp files: {new_files}"


def test_build_config_path_overwrites_on_repeat_spawn(monkeypatch, tmp_path) -> None:
    """Two spawns of the same name must reuse the same file, not accumulate."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    p1 = _build_config_path("Aria", provider="mock", model="mock")
    p2 = _build_config_path("Aria", provider="anthropic", model="claude-opus-4-7")
    assert p1 == p2
    cfg = yaml.safe_load(p2.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "anthropic"
    assert cfg["llm"]["model"] == "claude-opus-4-7"


def test_build_config_path_provider_only_keeps_default_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider="anthropic", model=None)
    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["provider"] == "anthropic"
    # Default model from default_consciousness.yaml preserved (not None)
    assert cfg["llm"]["model"]


def test_build_config_path_model_only_keeps_default_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    result = _build_config_path("Aria", provider=None, model="llama3.1:8b")
    cfg = yaml.safe_load(result.read_text(encoding="utf-8"))
    assert cfg["llm"]["model"] == "llama3.1:8b"
    assert cfg["llm"]["provider"]
