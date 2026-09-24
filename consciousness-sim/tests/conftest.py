"""Suite-wide fixtures.

Usage reporting is switched off for every test through the same environment
variable a user would set, so no test — nor any subprocess a test starts —
can report to the real trace service. tests/test_usage_reporting.py removes
it where a test needs reporting on, and points that test at a loopback stub.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _usage_reporting_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACE_USAGE_REPORTING", "off")
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
