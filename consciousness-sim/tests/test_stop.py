"""Tests for scripts/stop.py — the SIGTERM/SIGKILL sender for --bg instances.

Covers:
- the three pid-file states the command distinguishes: missing, stale
  (process gone), and live.
- the default SIGTERM path, including the 5 s grace poll and its escalation
  to SIGKILL when the process ignores SIGTERM.
- the --force path, which sends SIGKILL immediately and skips the poll.
- pid-file cleanup on every exit path that reaches the process.

No real signals are sent: ``os.kill`` is replaced with a recorder that
simulates the target process, and ``time.sleep`` is a no-op so the grace
window completes instantly.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from click.testing import CliRunner

# Make scripts/ importable the same way spawn.py's tests do.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.stop import main as stop_main  # noqa: E402


class _FakeProcess:
    """Stand-in for ``os.kill`` that records every signal sent to ``pid``.

    Signal 0 (the liveness probe) raises ``ProcessLookupError`` once the
    process is dead. Any other signal is recorded; the process dies after
    ``dies_after`` real signals, or never when ``dies_after`` is ``None``.
    """

    def __init__(self, pid: int, *, alive: bool = True, dies_after: int | None = 1) -> None:
        self.pid = pid
        self.alive = alive
        self.dies_after = dies_after
        self.sent: list[int] = []

    def __call__(self, pid: int, sig: int) -> None:
        assert pid == self.pid, f"signal sent to unexpected pid {pid}"
        if sig == 0:
            if not self.alive:
                raise ProcessLookupError
            return
        if not self.alive:
            raise ProcessLookupError
        self.sent.append(sig)
        if self.dies_after is not None and len(self.sent) >= self.dies_after:
            self.alive = False


def _write_pid(home: Path, name: str, pid: int | str) -> Path:
    agent_dir = home / name
    agent_dir.mkdir(parents=True, exist_ok=True)
    pid_path = agent_dir / "pid"
    pid_path.write_text(str(pid), encoding="utf-8")
    return pid_path


def _install_fake(monkeypatch, fake: _FakeProcess) -> list[float]:
    """Route os.kill to ``fake`` and make time.sleep instant, returning the
    list of requested sleep durations so the grace window can be asserted."""
    sleeps: list[float] = []
    monkeypatch.setattr(os, "kill", fake)
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    return sleeps


# --- pid file missing -------------------------------------------------------


def test_stop_exits_1_when_no_pid_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242)
    _install_fake(monkeypatch, fake)

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code == 1
    assert "No PID file found for 'Aria'" in result.output
    assert fake.sent == [], "nothing should be signalled without a pid file"


def test_stop_resolves_pid_file_through_sanitized_name(monkeypatch, tmp_path) -> None:
    """The pid path goes through consciousness_dir(), so a name spawn.py would
    have sanitized on the way in must be sanitized identically on the way out."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, alive=False)
    _install_fake(monkeypatch, fake)
    _write_pid(tmp_path, "Aria_1", 4242)

    result = CliRunner().invoke(stop_main, ["--name", "Aria 1"])

    assert result.exit_code == 0
    assert "not running" in result.output


# --- pid file stale ---------------------------------------------------------


def test_stop_removes_stale_pid_file_and_exits_0(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, alive=False)
    _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", 4242)

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert "Process 4242 is not running. Removing stale PID file." in result.output
    assert not pid_path.exists()
    assert fake.sent == []


# --- live process, default SIGTERM path --------------------------------------


def test_stop_sends_sigterm_and_removes_pid_file_when_process_exits(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, dies_after=1)
    sleeps = _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", 4242)

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert fake.sent == [signal.SIGTERM]
    assert "Sent SIGTERM to 'Aria' (PID 4242)" in result.output
    assert "Process stopped." in result.output
    assert not pid_path.exists()
    # The process was gone on the first poll, so exactly one 0.25 s tick elapsed.
    assert sleeps == [0.25]


def test_stop_escalates_to_sigkill_after_grace_window(monkeypatch, tmp_path) -> None:
    """A process that ignores SIGTERM is polled 20 x 0.25 s and then SIGKILLed."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, dies_after=None)
    sleeps = _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", 4242)

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert fake.sent == [signal.SIGTERM, signal.SIGKILL]
    assert "Process still alive after 5 s — escalating to SIGKILL." in result.output
    assert sleeps == [0.25] * 20
    assert not pid_path.exists()


def test_stop_tolerates_process_exiting_between_poll_and_sigkill(monkeypatch, tmp_path) -> None:
    """If the process dies after the last poll but before the escalation
    SIGKILL, the ProcessLookupError is swallowed and cleanup still happens."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, dies_after=None)
    _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", 4242)

    original_call = fake.__call__

    def _kill(pid: int, sig: int) -> None:
        if sig == signal.SIGKILL:
            fake.alive = False  # died just before the escalation signal
        original_call(pid, sig)

    monkeypatch.setattr(os, "kill", _kill)

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code == 0
    assert result.exception is None
    assert fake.sent == [signal.SIGTERM]
    assert not pid_path.exists()


# --- live process, --force --------------------------------------------------


def test_stop_force_sends_sigkill_without_polling(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242, dies_after=None)
    sleeps = _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", 4242)

    result = CliRunner().invoke(stop_main, ["--name", "Aria", "--force"])

    assert result.exit_code == 0
    assert fake.sent == [signal.SIGKILL]
    assert "Sent SIGKILL to 'Aria' (PID 4242)" in result.output
    assert "Process stopped." not in result.output
    assert sleeps == [], "--force must not enter the grace-window poll"
    assert not pid_path.exists()


# --- malformed pid file (characterization) ----------------------------------


def test_stop_malformed_pid_file_raises_value_error(monkeypatch, tmp_path) -> None:
    """Characterizes current behaviour: unlike spawn.py's _read_pid, stop.py
    parses the pid with a bare int() and a non-numeric file escapes as an
    uncaught ValueError rather than a clean error message. The file is left
    in place. Tracked as #179; when that is fixed, this test should flip to
    assert the friendly exit path instead."""
    monkeypatch.setenv("CONSCIOUSNESS_HOME", str(tmp_path))
    fake = _FakeProcess(pid=4242)
    _install_fake(monkeypatch, fake)
    pid_path = _write_pid(tmp_path, "Aria", "not-a-pid")

    result = CliRunner().invoke(stop_main, ["--name", "Aria"])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert pid_path.exists()
    assert fake.sent == []
