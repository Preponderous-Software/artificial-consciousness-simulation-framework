"""Unit tests for experiments/prune.py.

Uses a tempdir mimicking the `experiments/<manifest>/<UTC-ts>/` layout. No
subprocesses, no network. Fast.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from experiments.prune import find_prunable_runs, prune_runs


def _make_run(root: Path, manifest: str, name: str, *, mtime: float | None = None,
              with_started: bool = False) -> Path:
    """Create a fake run dir with a deterministic mtime."""
    d = root / manifest / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.yaml").write_text("placeholder\n", encoding="utf-8")
    if with_started:
        (d / ".STARTED").touch()
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def _seconds_ago(days: float) -> float:
    return time.time() - days * 86400


def test_keep_last_returns_dirs_beyond_n(tmp_path: Path) -> None:
    # Newest first: A is oldest, E is newest
    a = _make_run(tmp_path, "m1", "A", mtime=_seconds_ago(5))
    b = _make_run(tmp_path, "m1", "B", mtime=_seconds_ago(4))
    c = _make_run(tmp_path, "m1", "C", mtime=_seconds_ago(3))
    d = _make_run(tmp_path, "m1", "D", mtime=_seconds_ago(2))
    e = _make_run(tmp_path, "m1", "E", mtime=_seconds_ago(1))

    result = find_prunable_runs(tmp_path, keep_last=2)
    # Keeps E, D (2 newest); prunes C, B, A
    assert set(result) == {a, b, c}


def test_older_than_filters_by_age(tmp_path: Path) -> None:
    young = _make_run(tmp_path, "m1", "young", mtime=_seconds_ago(2))
    old = _make_run(tmp_path, "m1", "old", mtime=_seconds_ago(10))

    result = find_prunable_runs(tmp_path, older_than_days=7)
    assert result == [old]
    assert young not in result


def test_keep_last_and_older_than_intersect(tmp_path: Path) -> None:
    """A run is prunable only if BOTH policies say it is."""
    # Two old runs: only one of them is beyond keep_last=1
    old1 = _make_run(tmp_path, "m1", "old1", mtime=_seconds_ago(20))
    old2 = _make_run(tmp_path, "m1", "old2", mtime=_seconds_ago(10))
    young = _make_run(tmp_path, "m1", "young", mtime=_seconds_ago(2))

    # keep_last=1 → keep young (newest); old1, old2 are beyond
    # older_than=7 → old1, old2 are old enough; young is not
    # Intersection: old1, old2
    result = find_prunable_runs(tmp_path, keep_last=1, older_than_days=7)
    assert set(result) == {old1, old2}


def test_dirs_with_started_marker_are_skipped(tmp_path: Path) -> None:
    running = _make_run(tmp_path, "m1", "running", mtime=_seconds_ago(10), with_started=True)
    done = _make_run(tmp_path, "m1", "done", mtime=_seconds_ago(10))

    result = find_prunable_runs(tmp_path, older_than_days=7)
    assert result == [done]
    assert running not in result


def test_future_mtime_is_skipped_for_skew_protection(tmp_path: Path) -> None:
    future = _make_run(tmp_path, "m1", "future", mtime=time.time() + 3600)
    past = _make_run(tmp_path, "m1", "past", mtime=_seconds_ago(10))

    result = find_prunable_runs(tmp_path, older_than_days=7)
    assert result == [past]
    assert future not in result


def test_golden_and_manifests_subdirs_are_protected(tmp_path: Path) -> None:
    real_run = _make_run(tmp_path, "m1", "run-a", mtime=_seconds_ago(100))
    golden_run = _make_run(tmp_path, "golden", "Echo", mtime=_seconds_ago(100))
    manifest_file_dir = _make_run(tmp_path, "manifests", "foo.yaml.d", mtime=_seconds_ago(100))

    result = find_prunable_runs(tmp_path, older_than_days=7)
    assert real_run in result
    assert golden_run not in result
    assert manifest_file_dir not in result


def test_manifest_filter_scopes_to_one_subdir(tmp_path: Path) -> None:
    in_m1 = _make_run(tmp_path, "m1", "run-a", mtime=_seconds_ago(10))
    in_m2 = _make_run(tmp_path, "m2", "run-b", mtime=_seconds_ago(10))

    result = find_prunable_runs(tmp_path, older_than_days=7, manifest="m1")
    assert result == [in_m1]
    assert in_m2 not in result


def test_no_policy_returns_empty(tmp_path: Path) -> None:
    _make_run(tmp_path, "m1", "run-a", mtime=_seconds_ago(10))
    assert find_prunable_runs(tmp_path) == []


def test_missing_experiments_root_returns_empty(tmp_path: Path) -> None:
    assert find_prunable_runs(tmp_path / "does-not-exist", keep_last=1) == []


def test_prune_runs_dry_run_does_not_delete(tmp_path: Path) -> None:
    target = _make_run(tmp_path, "m1", "old", mtime=_seconds_ago(10))
    returned = prune_runs(tmp_path, older_than_days=7, dry_run=True)
    assert returned == [target]
    assert target.exists(), "dry-run must not delete"


def test_prune_runs_actually_deletes_when_not_dry_run(tmp_path: Path) -> None:
    target = _make_run(tmp_path, "m1", "old", mtime=_seconds_ago(10))
    other = _make_run(tmp_path, "m1", "young", mtime=_seconds_ago(1))
    returned = prune_runs(tmp_path, older_than_days=7, dry_run=False)
    assert returned == [target]
    assert not target.exists(), "non-dry-run must delete the target"
    assert other.exists(), "non-targets must remain"


def test_now_parameter_is_honored_for_age_computation(tmp_path: Path) -> None:
    """`now` injection lets tests pin a deterministic reference time."""
    fixed_now = datetime(2026, 5, 25, tzinfo=timezone.utc)
    # mtime 10 days before fixed_now
    ten_days_ago = (fixed_now - timedelta(days=10)).timestamp()
    five_days_ago = (fixed_now - timedelta(days=5)).timestamp()

    old = _make_run(tmp_path, "m1", "old", mtime=ten_days_ago)
    young = _make_run(tmp_path, "m1", "young", mtime=five_days_ago)

    result = find_prunable_runs(tmp_path, older_than_days=7, now=fixed_now)
    assert result == [old]
    assert young not in result
