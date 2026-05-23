"""Shared logging configuration for consciousness entry-point scripts."""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import sys
import os

# Ensure the package root is importable when this module is loaded directly.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPTS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from persistence.paths import consciousness_dir


def configure_logging(name: str, level: str) -> Path:
    """Configure a rotating file handler for the named consciousness instance.

    Logs go to <CONSCIOUSNESS_HOME>/<name>/run.log so they persist across
    Rich Live sessions (which swallow stderr). Returns the log path.
    """
    log_path = consciousness_dir(name) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.root.setLevel(getattr(logging, level.upper(), logging.WARNING))
    logging.root.addHandler(handler)
    return log_path
