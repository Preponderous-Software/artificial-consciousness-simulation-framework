"""Usage reporting to the trace service: that the framework was run, and nothing else.

No direct theory mapping — infrastructure module.

What is sent: the program name (``artificial-consciousness-simulation-framework``)
and its version (tag ``version``, on every event) with a ``startup`` event when ``scripts/spawn.py``,
``scripts/resume.py`` or ``scripts/web.py`` starts (tag ``command``; the web
dashboard's is also tagged ``service=true``), and an ``experiment-started``
event when ``scripts/experiment.py run`` begins a manifest. Every event also
carries a random installation ID (tag ``install``) so installations can be
counted rather than events: a UUID the client keeps in ``trace-install-id`` in
the persistence root, or the ``TRACE_INSTALL_ID`` environment variable when set.
Nothing about the user, instance names, configuration, providers, models, thoughts,
journals or any other content.

Reporting is on by default. The first launch writes a ``usage_reporting``
block to ``settings.json`` in the persistence root (``~/.consciousness/`` or
``$CONSCIOUSNESS_HOME``) and prints a one-line notice saying so and how to
turn it off; setting ``enabled`` to ``false`` there does. So do the
``TRACE_USAGE_REPORTING=off`` and ``DO_NOT_TRACK=1`` environment variables,
which every trace client honours and which win over the settings file because
the vendored client checks them first. Every switch fails closed: only
``true`` (or an "on" string) reports, ``"usage_reporting": false`` is read as
the switch and never overwritten, and a first launch under an environment
opt-out writes ``"enabled": false`` so the opt-out outlives the variable. An
opt-out written under another spelling (``usage-reporting``, ``usageReporting``,
an ``enable``/``Enabled`` switch, ``"disabled": true``) also counts as off, and
the first launch then writes ``"enabled": false`` next to it (see
:func:`has_near_miss_opt_out`). Every call returns immediately and
never raises: the network happens on a daemon thread owned by the client in
``interfaces/trace_client.py``.
Details: https://danielstephenson.dev/usage-reporting
"""

from __future__ import annotations

import atexit
import json
import os
import re
import tomllib
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from interfaces.trace_client import TraceClient, allows_reporting, environment_opts_out
from persistence.paths import consciousness_root

APPLICATION = "artificial-consciousness-simulation-framework"
SETTINGS_FILE_NAME = "settings.json"
INSTALL_ID_FILE_NAME = "trace-install-id"
SETTINGS_SECTION = "usage_reporting"
DEFAULT_ENDPOINT = "https://trace.danielstephenson.dev"
# The program key this framework ships with. Keys identify a program rather
# than guard anything (trace's ADR 0001), so it is kept here in the open.
DEFAULT_KEY = "0sw-zEhUfxPw_Z5DcYuf3stJ3cGZ-GHuSgjq56tAjZY"

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
# Sent as the version when neither pyproject.toml nor the installed metadata
# has one: the client requires a version, and a missing one must never stop
# an entry point from starting.
UNKNOWN_VERSION = "unknown"

DETAILS_URL = "https://danielstephenson.dev/usage-reporting"

FIRST_RUN_NOTICE = (
    "Usage reporting is on: artificial-consciousness-simulation-framework sends its name "
    "and version (with a random installation ID) when spawn, resume or the web dashboard "
    "starts and when an experiment starts, to https://trace.danielstephenson.dev - nothing "
    "about you, your instances or their content. Turn it off with "
    '"usage_reporting": {"enabled": false} in settings.json in the persistence root '
    "(~/.consciousness/ or $CONSCIOUSNESS_HOME), or for every trace-reporting program "
    "with the environment variable TRACE_USAGE_REPORTING=off. Details: " + DETAILS_URL
)

# Shown on the first launch instead when TRACE_USAGE_REPORTING=off or
# DO_NOT_TRACK=1 is already set: the block is written with "enabled": false,
# so a later launch without the variable does not report either.
FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT = (
    'Usage reporting is off (environment), and "usage_reporting": {"enabled": false} was '
    "written to settings.json so it stays off. Details: " + DETAILS_URL
)


def settings_file() -> Path:
    """Where the usage_reporting block lives: the persistence root, resolved per call so
    ``CONSCIOUSNESS_HOME`` set by a test or a wrapper is honoured."""
    return consciousness_root() / SETTINGS_FILE_NAME


def install_id_file() -> Path:
    """Where the client keeps this installation's random ID: next to the settings file in
    the persistence root, resolved per call like :func:`settings_file`. The client only
    creates it when reporting is on; deleting it resets the ID."""
    return consciousness_root() / INSTALL_ID_FILE_NAME


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


# Shown on the first launch instead when settings.json already turns reporting
# off under another spelling (see has_near_miss_opt_out): the real block is
# written with "enabled": false next to it.
FIRST_RUN_NOTICE_OFF_BY_SETTINGS = (
    'Usage reporting is off (settings.json), and "usage_reporting": {"enabled": false} was '
    "written to settings.json so it stays off. Details: " + DETAILS_URL
)


def first_run_notice(near_miss_opt_out: bool = False) -> str:
    """FIRST_RUN_NOTICE, unless the environment or a near-miss opt-out in the
    settings file has already turned reporting off."""
    if environment_opts_out():
        return FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT
    if near_miss_opt_out:
        return FIRST_RUN_NOTICE_OFF_BY_SETTINGS
    return FIRST_RUN_NOTICE


def _normalise_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def has_near_miss_opt_out(settings: dict[str, Any], settings_key: str = SETTINGS_SECTION,
                          switch_key: str = "enabled") -> bool:
    """Whether ``settings`` holds an attempt to turn usage reporting off that is
    not written exactly as ``usage_reporting`` -> ``enabled``. Such a setting
    would not be recognised, the default block (``enabled: true``) would be
    written next to it, and reporting would stay on. Whoever wrote any of these
    did not ask for reporting, so each counts as off:

    - a top-level key that spells "usage reporting" another way
      (``usage-reporting``, ``usageReporting``, ``UsageReporting``, ...) holding
      a value, or an ``enabled``-like key, that is not clearly on;
    - ``"usage_reporting": "off"`` -- a value instead of a block;
    - inside any such block, a key that spells "enabled" another way
      (``enable``, ``Enabled``, ...) that is not clearly on, or one that spells
      "disabled" whose value is on.
    """
    for key, value in settings.items():
        if "usagereport" not in _normalise_key(key):
            continue
        if not isinstance(value, dict):
            if not allows_reporting(value):
                return True
            continue
        for child, child_value in value.items():
            if key == settings_key and child == switch_key:
                continue  # the switch itself, read by the caller
            name = _normalise_key(child)
            if "disabl" in name and allows_reporting(child_value):
                return True
            if "enabl" in name and not allows_reporting(child_value):
                return True
    return False


def default_settings(enabled: bool = True) -> dict[str, Any]:
    """The usage_reporting block written to the settings file on the first launch."""
    return {"enabled": enabled, "endpoint": DEFAULT_ENDPOINT, "key": DEFAULT_KEY}


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

    # An opt-out written under another spelling (usage-reporting, an "enable"
    # switch, "disabled": true, ...) is an opt-out.
    near_miss_opt_out = has_near_miss_opt_out(settings)
    section = settings.get(SETTINGS_SECTION)
    if isinstance(section, dict):
        return dict(section, enabled=False) if near_miss_opt_out else section
    if SETTINGS_SECTION in settings:
        # "usage_reporting": false (or "off", or anything else that is not a
        # block) is the switch itself: read it as such, and never write the
        # default block (on) over it.
        return {"enabled": False if near_miss_opt_out else section}

    # Under an environment or near-miss opt-out the first launch records
    # "off", so a later launch does not quietly report.
    block = default_settings(enabled=not (environment_opts_out() or near_miss_opt_out))
    settings[SETTINGS_SECTION] = block
    log(first_run_notice(near_miss_opt_out))
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
    shipped default. Every event carries :func:`read_version` (or
    ``UNKNOWN_VERSION``) as the tag ``version`` and the installation ID
    (``TRACE_INSTALL_ID``, else :func:`install_id_file`) as ``install``, both
    added by the client.
    """
    if section is None:
        return TraceClient.disabled()
    try:
        return TraceClient(
            str(section.get("endpoint") or DEFAULT_ENDPOINT),
            APPLICATION,
            read_version() or UNKNOWN_VERSION,
            key=str(section.get("key") or DEFAULT_KEY),
            # Not bool(): bool("false") is True. Only true or an "on" string
            # reports; "false", "off", 0, null or a typo do not.
            enabled=allows_reporting(section.get("enabled", True)),
            # Resolved by the client only after its opt-out checks: a disabled
            # client never creates the file.
            install_id=os.environ.get("TRACE_INSTALL_ID"),
            install_id_file=install_id_file(),
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
    """Report ``startup`` for the entry point ``command`` (spawn / resume / web).
    The client adds the ``version`` tag itself."""
    tags = {"command": command}
    if service:
        tags["service"] = "true"
    client.report("startup", tags=tags)


def report_experiment_started(client: TraceClient) -> None:
    """Report that ``scripts/experiment.py run`` began running a manifest."""
    client.report("experiment-started")
