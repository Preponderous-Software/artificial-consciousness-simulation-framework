"""Shared path helpers for safe consciousness persistence locations."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SAFE_NAME_PATTERN = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_consciousness_name(name: str) -> str:
    """Return a filesystem-safe consciousness identifier."""
    cleaned = _SAFE_NAME_PATTERN.sub("_", name.strip()).strip("._-")
    if not cleaned:
        raise ValueError("Consciousness name must include at least one alphanumeric character.")
    return cleaned


def consciousness_root() -> Path:
    """Return persistence root honoring CONSCIOUSNESS_HOME."""
    return Path(os.path.expanduser(os.getenv("CONSCIOUSNESS_HOME", "~/.consciousness")))


def consciousness_dir(name: str) -> Path:
    """Return per-agent persistence directory using a sanitized name."""
    return consciousness_root() / sanitize_consciousness_name(name)
