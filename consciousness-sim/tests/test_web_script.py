"""Tests for scripts/web.py — the standalone dashboard launcher (#55).

Covers:
- the defaults (127.0.0.1:8080, remote spawn off) reaching
  ``interfaces.web.server.start`` and the printed dashboard URL, which shows
  ``localhost`` for the loopback bind.
- the two exposure notices on stderr: the NOTE for a non-loopback bind
  without ``--allow-remote-spawn``, and the WARNING whenever
  ``--allow-remote-spawn`` is passed (which replaces the NOTE).
- ``--log-level`` mapping onto ``logging.basicConfig``, including the INFO
  fallback for an unknown level name.
- the startup usage event being tagged ``web`` and ``service=True``.

``start`` is replaced with an async recorder, so no uvicorn server is bound;
usage reporting and ``load_dotenv`` are stubbed so nothing leaves the process.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import interfaces.web.server as server  # noqa: E402
import scripts.web as web  # noqa: E402


class _Recorder:
    """Collects every call the patched collaborators receive."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.log_levels: list[int] = []
        self.startups: list[tuple[str, bool]] = []


def _install(monkeypatch) -> _Recorder:
    rec = _Recorder()

    async def _start(port: int, host: str = "127.0.0.1", allow_remote_spawn: bool = False) -> None:
        rec.starts.append({"port": port, "host": host, "allow_remote_spawn": allow_remote_spawn})

    def _basic_config(**kwargs: Any) -> None:
        rec.log_levels.append(kwargs["level"])

    def _report_startup(client: Any, command: str, service: bool = False) -> None:
        rec.startups.append((command, service))

    # main() imports start lazily from the module, so patch it at the source.
    monkeypatch.setattr(server, "start", _start)
    monkeypatch.setattr(logging, "basicConfig", _basic_config)
    monkeypatch.setattr(web, "start_usage_reporting", lambda log: object())
    monkeypatch.setattr(web, "report_startup", _report_startup)
    # A developer's consciousness-sim/.env must not leak into the test run.
    monkeypatch.setattr(web, "load_dotenv", lambda: None)
    return rec


# --- bind defaults ----------------------------------------------------------


def test_web_defaults_bind_loopback_8080_without_remote_spawn(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, [])

    assert result.exit_code == 0, result.output
    assert rec.starts == [{"port": 8080, "host": "127.0.0.1", "allow_remote_spawn": False}]
    assert "Dashboard: http://localhost:8080" in result.stdout
    assert result.stderr == ""


def test_web_passes_explicit_port_through(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--port", "9123"])

    assert result.exit_code == 0, result.output
    assert rec.starts[0]["port"] == 9123
    assert "Dashboard: http://localhost:9123" in result.stdout


# --- exposure notices -------------------------------------------------------


def test_web_non_loopback_host_notes_spawn_stays_localhost_only(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--host", "0.0.0.0"])

    assert result.exit_code == 0, result.output
    assert rec.starts == [{"port": 8080, "host": "0.0.0.0", "allow_remote_spawn": False}]
    assert "NOTE: bound to 0.0.0.0" in result.stderr
    assert "WARNING" not in result.stderr
    # The URL shows the real bind host once it is not loopback.
    assert "Dashboard: http://0.0.0.0:8080" in result.stdout


def test_web_allow_remote_spawn_warns_and_replaces_note(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--host", "0.0.0.0", "--allow-remote-spawn"])

    assert result.exit_code == 0, result.output
    assert rec.starts == [{"port": 8080, "host": "0.0.0.0", "allow_remote_spawn": True}]
    assert "WARNING: --allow-remote-spawn enabled" in result.stderr
    assert "http://0.0.0.0:8080/" in result.stderr
    assert "NOTE:" not in result.stderr


def test_web_allow_remote_spawn_warns_on_loopback_too(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--allow-remote-spawn"])

    assert result.exit_code == 0, result.output
    assert rec.starts[0]["allow_remote_spawn"] is True
    assert "WARNING: --allow-remote-spawn enabled" in result.stderr
    assert "NOTE:" not in result.stderr


# --- logging ----------------------------------------------------------------


def test_web_default_log_level_is_info(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, [])

    assert result.exit_code == 0, result.output
    assert rec.log_levels == [logging.INFO]


def test_web_log_level_is_case_insensitive(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--log-level", "debug"])

    assert result.exit_code == 0, result.output
    assert rec.log_levels == [logging.DEBUG]


def test_web_unknown_log_level_falls_back_to_info(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, ["--log-level", "NOPE"])

    assert result.exit_code == 0, result.output
    assert rec.log_levels == [logging.INFO]


# --- usage reporting --------------------------------------------------------


def test_web_reports_startup_as_service(monkeypatch) -> None:
    rec = _install(monkeypatch)

    result = CliRunner().invoke(web.main, [])

    assert result.exit_code == 0, result.output
    assert rec.startups == [("web", True)]
