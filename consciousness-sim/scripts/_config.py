"""Per-instance config resolution shared by spawn.py and resume.py (#187).

No direct theory mapping — entry-point helper.
An instance runs against the shipped default unless it has been customized
(``--config``, ``--provider``, ``--model``); a customized instance keeps its
resolved config at ``<CONSCIOUSNESS_HOME>/<name>/config.yaml`` so a later
resume or dashboard respawn does not silently revert to the defaults.
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPTS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from core.consciousness import _expand_env_vars, _validate_config  # noqa: E402
from persistence.paths import consciousness_dir  # noqa: E402

DEFAULT_CONFIG_PATH: Path = _ROOT_DIR / "config" / "default_consciousness.yaml"
INSTANCE_CONFIG_NAME: str = "config.yaml"


class ConfigError(Exception):
    """A config file that cannot be loaded or fails ``_validate_config``."""


def instance_config_path(name: str) -> Path:
    return consciousness_dir(name) / INSTANCE_CONFIG_NAME


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"Config {path} must be a YAML mapping, got {type(parsed).__name__}")
    return parsed


def resolve_config_path(
    name: str,
    config: Path | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Return the config file the instance ``name`` should run against.

    Precedence: ``config`` > the instance's persisted ``config.yaml`` > the
    shipped default; ``provider`` / ``model`` are applied on top. Without any
    override the existing file is returned untouched, so an uncustomized
    instance keeps tracking the shipped default across upgrades.
    """
    persisted = instance_config_path(name)
    if config is None and provider is None and model is None:
        return persisted if persisted.is_file() else DEFAULT_CONFIG_PATH

    base_path = config or (persisted if persisted.is_file() else DEFAULT_CONFIG_PATH)
    raw = _load_mapping(base_path)
    if provider or model:
        llm_cfg = raw.get("llm")
        if not isinstance(llm_cfg, dict):
            raise ConfigError(f"Config {base_path} has no 'llm' mapping to apply --provider/--model to")
        if provider:
            llm_cfg["provider"] = provider
        if model:
            llm_cfg["model"] = model

    # Validate before persisting so a bad --config cannot replace a working
    # instance config. The persisted copy keeps ${VAR} references unexpanded
    # so secrets such as the Discord webhook URL never land on disk.
    try:
        _validate_config(_expand_env_vars(copy.deepcopy(raw)))
    except (KeyError, ValueError, TypeError) as exc:
        raise ConfigError(f"Invalid config {base_path}: {exc}") from exc

    persisted.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = persisted.with_name(persisted.name + ".tmp")
    tmp_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    os.replace(tmp_path, persisted)
    return persisted
