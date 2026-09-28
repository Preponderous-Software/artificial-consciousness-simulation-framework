"""Tests for scripts/_config.py — per-instance config resolution (#187).

Covers:
- an uncustomized instance runs against the shipped default, and one with a
  persisted ``config.yaml`` runs against that file, both without rewriting it.
- ``config`` / ``provider`` / ``model`` overrides are layered in precedence
  order and persisted to ``<CONSCIOUSNESS_HOME>/<name>/config.yaml``.
- the persisted copy keeps ``${VAR}`` references unexpanded.
- an invalid override raises ConfigError and leaves the existing file intact.
- the #105 regression: overrides never leak files outside the instance dir.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    ConfigError,
    resolve_config_path,
)


def _default() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


def _write(path: Path, cfg: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    return tmp_path


def test_uncustomized_instance_uses_default_and_persists_nothing(home) -> None:
    assert resolve_config_path("Aria") == DEFAULT_CONFIG_PATH
    assert not (home / "Aria" / "config.yaml").exists()


def test_persisted_config_is_used_without_rewriting(home) -> None:
    cfg = _default()
    cfg["thought_loop"]["rpt_critique"] = True
    persisted = _write(home / "Aria" / "config.yaml", cfg)
    before = persisted.read_text(encoding="utf-8")

    assert resolve_config_path("Aria") == persisted
    assert persisted.read_text(encoding="utf-8") == before


def test_config_override_is_copied_into_instance_dir(home, tmp_path) -> None:
    cfg = _default()
    cfg["mood"]["semantic"]["enabled"] = True
    source = _write(tmp_path / "src" / "custom.yaml", cfg)

    result = resolve_config_path("Aria", config=source)

    assert result == home / "Aria" / "config.yaml"
    assert _load(result)["mood"]["semantic"]["enabled"] is True


def test_provider_and_model_apply_on_top_of_config_override(home, tmp_path) -> None:
    cfg = _default()
    cfg["thought_loop"]["rpt_critique"] = True
    source = _write(tmp_path / "src" / "custom.yaml", cfg)

    result = _load(resolve_config_path("Aria", config=source, provider="mock", model="mock"))

    assert result["llm"]["provider"] == "mock"
    assert result["llm"]["model"] == "mock"
    assert result["thought_loop"]["rpt_critique"] is True


def test_provider_override_layers_on_persisted_config(home) -> None:
    cfg = _default()
    cfg["thought_loop"]["rpt_critique"] = True
    _write(home / "Aria" / "config.yaml", cfg)

    result = _load(resolve_config_path("Aria", provider="anthropic"))

    assert result["llm"]["provider"] == "anthropic"
    assert result["llm"]["model"] == _default()["llm"]["model"]
    assert result["thought_loop"]["rpt_critique"] is True


def test_model_only_override_keeps_default_provider(home) -> None:
    result = _load(resolve_config_path("Aria", model="llama3.1:8b"))

    assert result["llm"]["model"] == "llama3.1:8b"
    assert result["llm"]["provider"] == _default()["llm"]["provider"]


def test_repeat_overrides_reuse_one_file(home) -> None:
    p1 = resolve_config_path("Aria", provider="mock", model="mock")
    p2 = resolve_config_path("Aria", provider="anthropic", model="claude-opus-4-7")

    assert p1 == p2
    assert _load(p2)["llm"]["provider"] == "anthropic"
    assert sorted(p.name for p in (home / "Aria").iterdir()) == ["config.yaml"]


def test_overrides_do_not_leak_tmp_files(home) -> None:
    """Regression for #105 — an earlier implementation left a /tmp file per spawn."""
    before = set(glob.glob("/tmp/consciousness_override_*.yaml"))
    for _ in range(3):
        resolve_config_path("Aria", provider="mock", model="mock")
    assert set(glob.glob("/tmp/consciousness_override_*.yaml")) == before


def test_env_var_references_are_persisted_unexpanded(home, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_DISCORD_WEBHOOK", "https://example.invalid/secret")
    source = _write(tmp_path / "src" / "custom.yaml", _default())

    result = _load(resolve_config_path("Aria", config=source))

    assert result["discord"]["webhook_url"] == "${CONSCIOUSNESS_DISCORD_WEBHOOK}"


def test_invalid_override_raises_and_keeps_existing_config(home, tmp_path) -> None:
    cfg = _default()
    cfg["thought_loop"]["rpt_critique"] = True
    persisted = _write(home / "Aria" / "config.yaml", cfg)
    before = persisted.read_text(encoding="utf-8")
    bad = _default()
    del bad["memory"]
    source = _write(tmp_path / "src" / "bad.yaml", bad)

    with pytest.raises(ConfigError, match="memory"):
        resolve_config_path("Aria", config=source)

    assert persisted.read_text(encoding="utf-8") == before


def test_non_mapping_config_raises(home, tmp_path) -> None:
    source = tmp_path / "list.yaml"
    source.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        resolve_config_path("Aria", config=source)
