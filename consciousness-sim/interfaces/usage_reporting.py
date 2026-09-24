"""Usage reporting to the trace service: that the framework was run, and nothing else.

No direct theory mapping — infrastructure module.

What is sent: the program name (``artificial-consciousness-simulation-framework``)
and its version with a ``startup`` event when ``scripts/spawn.py``,
``scripts/resume.py`` or ``scripts/web.py`` starts (tag ``command``; the web
dashboard's is also tagged ``service=true``), and an ``experiment-started``
event when ``scripts/experiment.py run`` begins a manifest. Nothing about the
user, the machine, instance names, configuration, providers, models, thoughts,
journals or any other content.

Reporting is on by default. The first launch writes a ``usage_reporting``
block to ``settings.json`` in the persistence root (``~/.consciousness/`` or
``$CONSCIOUSNESS_HOME``) and prints a one-line notice saying so and how to
turn it off; setting ``enabled`` to ``false`` there does. So do the
``TRACE_USAGE_REPORTING=off`` and ``DO_NOT_TRACK=1`` environment variables,
which every trace client honours and which win over the settings file because
the vendored client checks them first. Every call returns immediately and
never raises: the network happens on a daemon thread owned by the client in
``interfaces/trace_client.py``.
Details: https://github.com/Stephenson-Software/trace#usage-reporting
"""

from __future__ import annotations

import atexit
import json
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from interfaces.trace_client import TraceClient, environment_opts_out
from persistence.paths import consciousness_root

APPLICATION = "artificial-consciousness-simulation-framework"
SETTINGS_FILE_NAME = "settings.json"
SETTINGS_SECTION = "usage_reporting"
DEFAULT_ENDPOINT = "https://trace.danielstephenson.dev"
# The program key this framework ships with. Keys identify a program rather
# than guard anything (trace's ADR 0001), so it is kept here in the open.
DEFAULT_KEY = "0sw-zEhUfxPw_Z5DcYuf3stJ3cGZ-GHuSgjq56tAjZY"

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

DETAILS_URL = "https://github.com/Stephenson-Software/trace#usage-reporting"

FIRST_RUN_NOTICE = (
    "Usage reporting is on: artificial-consciousness-simulation-framework sends its name "
    "and version when spawn, resume or the web dashboard starts and when an experiment "
    "starts, to https://trace.danielstephenson.dev - nothing about you, your machine, "
    "your instances or their content. Turn it off with "
    '"usage_reporting": {"enabled": false} in settings.json in the persistence root '
    "(~/.consciousness/ or $CONSCIOUSNESS_HOME), or for every trace-reporting program "
    "with the environment variable TRACE_USAGE_REPORTING=off. Details: " + DETAILS_URL
)

# Shown on the first launch instead when TRACE_USAGE_REPORTING=off or
# DO_NOT_TRACK=1 is already set: the block is still written, but saying
# reporting is on would mislead.
FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT = "Usage reporting is off (environment). Details: " + DETAILS_URL


def settings_file() -> Path:
    """Where the usage_reporting block lives: the persistence root, resolved per call so
    ``CONSCIOUSNESS_HOME`` set by a test or a wrapper is honoured."""
    return consciousness_root() / SETTINGS_FILE_NAME


def read_version(pyproject: Path = _PYPROJECT) -> str | None:
    """The project version: pyproject.toml of this checkout, else the installed
    distribution's metadata, else None."""
    try:
        with pyproject.open("rb") as f:
            version = tomllib.load(f)["project"]["version"]
        if isinstance(version, str) and version.strip():
            return version.strip()
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        pass
    try:
        return metadata.version("consciousness-sim")
    except metadata.PackageNotFoundError:
        return None


def first_run_notice() -> str:
    """FIRST_RUN_NOTICE, unless the environment has already opted out."""
    if environment_opts_out():
        return FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT
    return FIRST_RUN_NOTICE


def default_settings() -> dict[str, Any]:
    """The usage_reporting block written to the settings file on the first launch."""
    return {"enabled": True, "endpoint": DEFAULT_ENDPOINT, "key": DEFAULT_KEY}


def load_settings(path: Path | None = None, log: Callable[[str], None] = print) -> dict[str, Any] | None:
    """Read the usage_reporting block, writing the default block (and printing the
    one-time notice) when the settings file does not have one yet.

    Returns the block, or None if the settings file exists but cannot be read, in
    which case nothing is reported and the file is left alone.
    """
    target = settings_file() if path is None else path
    settings: dict[str, Any] = {}
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("settings file is not a JSON object")
            settings = loaded
        except (OSError, ValueError) as e:
            log(f"Could not read {target} ({e}); usage reporting is off until it is fixed.")
            return None

    section = settings.get(SETTINGS_SECTION)
    if isinstance(section, dict):
        return section

    block = default_settings()
    settings[SETTINGS_SECTION] = block
    log(first_run_notice())
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        log(f"Could not write {target} ({e}); the notice above will be shown again next time.")
    return block


def build_client(section: dict[str, Any] | None) -> TraceClient:
    """A TraceClient for the given block; disabled when it is None.

    Always built through the client's constructor otherwise, which puts
    TRACE_USAGE_REPORTING / DO_NOT_TRACK ahead of ``enabled`` and records why it
    is off in ``disabled_reason``. A missing endpoint or key falls back to the
    shipped default.
    """
    if section is None:
        return TraceClient.disabled()
    try:
        return TraceClient(
            str(section.get("endpoint") or DEFAULT_ENDPOINT),
            APPLICATION,
            key=str(section.get("key") or DEFAULT_KEY),
            enabled=bool(section.get("enabled", True)),
        )
    except Exception:
        return TraceClient.disabled()


def start_usage_reporting(path: Path | None = None, log: Callable[[str], None] = print) -> TraceClient:
    """Read (or on the first launch write) the settings and build the client.

    Reports nothing by itself, so a process that only hands off to another one
    (``spawn.py --bg``'s parent) can show the notice without counting twice.
    Never raises; the client is closed (sending what is queued, bounded by its
    timeout) when the interpreter exits.
    """
    try:
        client = build_client(load_settings(path, log))
    except Exception:
        return TraceClient.disabled()
    atexit.register(client.close)
    return client


def report_startup(client: TraceClient, command: str, service: bool = False) -> None:
    """Report ``startup`` for the entry point ``command`` (spawn / resume / web)."""
    tags = {"command": command}
    version = read_version()
    if version:
        tags["version"] = version
    if service:
        tags["service"] = "true"
    client.report("startup", tags=tags)


def report_experiment_started(client: TraceClient) -> None:
    """Report that ``scripts/experiment.py run`` began running a manifest."""
    client.report("experiment-started")
