"""Tests for interfaces/usage_reporting.py and its wiring into the entry points.

No direct theory mapping — infrastructure tests. Nothing here contacts the real
trace service: tests that need reporting on point it at a loopback stub, and
every other test runs with TRACE_USAGE_REPORTING=off (tests/conftest.py).
"""

from __future__ import annotations

import json
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest
from click.testing import CliRunner

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from interfaces import usage_reporting  # noqa: E402
from interfaces.usage_reporting import (  # noqa: E402
    APPLICATION,
    DEFAULT_ENDPOINT,
    DEFAULT_KEY,
    DETAILS_URL,
    FIRST_RUN_NOTICE,
    FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT,
    build_client,
    load_settings,
    read_version,
    report_experiment_started,
    report_startup,
    settings_file,
    start_usage_reporting,
)


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An isolated persistence root with reporting allowed by the environment."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    monkeypatch.delenv("TRACE_USAGE_REPORTING", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    return tmp_path


@pytest.fixture
def stub() -> Iterator[tuple[str, list[dict[str, Any]], threading.Event]]:
    """A loopback trace stand-in that records every POST and answers 201."""
    requests: list[dict[str, Any]] = []
    arrived = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            requests.append({
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": json.loads(self.rfile.read(length).decode("utf-8")),
            })
            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()
            arrived.set()

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1], requests, arrived
    finally:
        server.shutdown()
        server.server_close()


def _write_settings(home: Path, block: dict[str, Any]) -> None:
    (home / "settings.json").write_text(json.dumps({"usage_reporting": block}), encoding="utf-8")


def test_application_name_endpoint_and_shipped_key() -> None:
    assert APPLICATION == "artificial-consciousness-simulation-framework"
    assert DEFAULT_ENDPOINT == "https://trace.danielstephenson.dev"
    assert len(DEFAULT_KEY) == 43


def test_settings_file_lives_in_the_persistence_root(home: Path) -> None:
    assert settings_file() == home / "settings.json"


def test_version_comes_from_pyproject() -> None:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as f:
        expected = tomllib.load(f)["project"]["version"]
    assert read_version() == expected


def test_first_run_writes_the_block_and_shows_the_notice_once(home: Path) -> None:
    logged: list[str] = []

    section = load_settings(log=logged.append)

    assert section == {"enabled": True, "endpoint": DEFAULT_ENDPOINT, "key": DEFAULT_KEY}
    assert logged == [FIRST_RUN_NOTICE]
    assert json.loads((home / "settings.json").read_text(encoding="utf-8")) == {"usage_reporting": section}

    logged.clear()
    assert load_settings(log=logged.append) == section
    assert logged == [], "the notice must not be shown on the second run"


def test_first_run_creates_a_missing_persistence_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path / "not-yet"))
    load_settings(log=lambda message: None)
    assert (tmp_path / "not-yet" / "settings.json").is_file()


def test_notice_says_reporting_is_on_and_names_every_opt_out() -> None:
    assert FIRST_RUN_NOTICE.startswith("Usage reporting is on: artificial-consciousness-simulation-framework sends")
    assert "https://trace.danielstephenson.dev" in FIRST_RUN_NOTICE
    assert '"enabled": false' in FIRST_RUN_NOTICE
    assert "TRACE_USAGE_REPORTING=off" in FIRST_RUN_NOTICE
    assert DETAILS_URL in FIRST_RUN_NOTICE
    assert DETAILS_URL == "https://github.com/Stephenson-Software/trace#usage-reporting"
    assert "\n" not in FIRST_RUN_NOTICE


def test_first_run_under_an_environment_opt_out_says_reporting_is_off(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    logged: list[str] = []
    load_settings(log=logged.append)
    assert logged == [FIRST_RUN_NOTICE_OFF_BY_ENVIRONMENT]


def test_existing_settings_are_preserved_when_the_block_is_added(home: Path) -> None:
    (home / "settings.json").write_text(json.dumps({"other": {"kept": 1}}), encoding="utf-8")
    load_settings(log=lambda message: None)
    written = json.loads((home / "settings.json").read_text(encoding="utf-8"))
    assert written["other"] == {"kept": 1}
    assert written["usage_reporting"]["enabled"] is True


def test_opt_out_is_respected_and_not_rewritten(home: Path) -> None:
    _write_settings(home, {"enabled": False})
    logged: list[str] = []
    client = build_client(load_settings(log=logged.append))
    assert not client.enabled
    assert client.disabled_reason == "config"
    assert logged == []
    assert json.loads((home / "settings.json").read_text(encoding="utf-8")) == {"usage_reporting": {"enabled": False}}


def test_unreadable_settings_disable_reporting_and_are_left_alone(home: Path) -> None:
    (home / "settings.json").write_text("{not json", encoding="utf-8")
    logged: list[str] = []
    section = load_settings(log=logged.append)
    assert section is None
    assert not build_client(section).enabled
    assert "usage reporting is off" in logged[0]
    assert (home / "settings.json").read_text(encoding="utf-8") == "{not json"


def test_missing_endpoint_and_key_fall_back_to_the_shipped_defaults(home: Path) -> None:
    client = build_client({"enabled": True})
    assert client.disabled_reason is None
    assert client._endpoint == DEFAULT_ENDPOINT + "/api/metrics"
    assert client._key == DEFAULT_KEY
    client.close()  # nothing was reported, so nothing is sent


def test_the_suite_runs_with_reporting_off() -> None:
    """tests/conftest.py's guard: without the `home` fixture the environment wins."""
    assert build_client({"enabled": True, "key": "k"}).disabled_reason == "environment"


@pytest.mark.parametrize("variable,value", [("TRACE_USAGE_REPORTING", "off"), ("DO_NOT_TRACK", "1")])
def test_environment_opt_out_wins_over_enabled_settings(
    home: Path, monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    _write_settings(home, {"enabled": True, "key": "k"})
    monkeypatch.setenv(variable, value)
    client = build_client(load_settings(log=lambda message: None))
    assert not client.enabled
    assert client.disabled_reason == "environment"


def test_startup_and_experiment_started_reach_the_configured_endpoint(
    home: Path, monkeypatch: pytest.MonkeyPatch, stub: tuple[str, list[dict[str, Any]], threading.Event]
) -> None:
    endpoint, requests, _ = stub
    _write_settings(home, {"enabled": True, "endpoint": endpoint, "key": "test-key"})
    monkeypatch.setattr(usage_reporting.atexit, "register", lambda f: None)

    client = start_usage_reporting(log=lambda message: None)
    report_startup(client, "spawn")
    report_startup(client, "web", service=True)
    report_experiment_started(client)
    client.close()

    assert [r["path"] for r in requests] == ["/api/metrics"] * 3
    assert {r["authorization"] for r in requests} == {"Bearer test-key"}
    version = read_version()
    assert [r["body"] for r in requests] == [
        {"application": APPLICATION, "name": "startup", "tags": {"command": "spawn", "version": version}},
        {"application": APPLICATION, "name": "startup",
         "tags": {"command": "web", "version": version, "service": "true"}},
        {"application": APPLICATION, "name": "experiment-started"},
    ]


def test_start_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(usage_reporting, "load_settings", boom)
    assert not start_usage_reporting(log=lambda message: None).enabled


def test_spawn_bg_parent_shows_the_notice_but_leaves_startup_to_the_child(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.spawn as spawn

    reported: list[str] = []
    monkeypatch.setattr(spawn, "report_startup", lambda client, command, service=False: reported.append(command))

    class FakeProcess:
        pid = 424242

    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setenv("TRACE_USAGE_REPORTING", "off")  # the real client stays silent regardless

    result = CliRunner().invoke(spawn.main, ["--name", "Aria", "--bg"])

    assert result.exit_code == 0, result.output
    assert reported == [], "the --bg parent must not count the run; its child does"
    assert "Usage reporting is off (environment)" in result.output
    assert (home / "settings.json").is_file()
