"""Tests for the web dashboard's optional-dependency packaging (issue #169).

`interfaces.web.server` is a packaged module that imports FastAPI at module
scope and uvicorn inside `start()`. Neither is a core dependency, so both are
declared in the `web` extra. These tests pin two halves of that contract: the
extra actually lists the packages the dashboard imports, and a missing extra
surfaces as a message naming what to install rather than a bare
`ModuleNotFoundError: No module named 'fastapi'`.
"""

from __future__ import annotations

import builtins
import importlib
import re
import sys
import tomllib
from pathlib import Path

import pytest

from interfaces.web._deps import WEB_EXTRA_HINT, reraise_if_web_extra_missing

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _requirement_names(requirements: list[str]) -> set[str]:
    """Reduce PEP 508 requirement strings to their bare distribution names."""
    return {re.split(r"[<>=!~\[;\s]", req, maxsplit=1)[0].strip().lower() for req in requirements}


# ---------------------------------------------------------------------------
# Declared metadata
# ---------------------------------------------------------------------------

def test_pyproject_web_extra_declares_the_dashboard_imports():
    """Regression: the packaged dashboard's imports must be installable."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    extras = metadata["project"]["optional-dependencies"]
    assert "web" in extras, "the 'web' extra named by the import guard is not declared"
    assert {"fastapi", "uvicorn"} <= _requirement_names(extras["web"])


def test_pyproject_web_extra_matches_the_name_in_the_import_guard():
    """The hint tells the reader to install an extra that exists."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for extra in metadata["project"]["optional-dependencies"]:
        if f"[{extra}]" in WEB_EXTRA_HINT:
            return
    pytest.fail(f"no declared extra is named in the guidance message: {WEB_EXTRA_HINT}")


def test_web_extra_packages_are_not_core_dependencies():
    """Keeping them out of [project].dependencies is the point of the extra."""
    metadata = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    core = _requirement_names(metadata["project"]["dependencies"])
    assert not ({"fastapi", "uvicorn"} & core)


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def test_reraise_if_web_extra_missing_names_the_extra():
    exc = ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
    with pytest.raises(ModuleNotFoundError) as excinfo:
        reraise_if_web_extra_missing(exc)
    assert "consciousness-sim[web]" in str(excinfo.value)
    assert excinfo.value.name == "fastapi"
    assert excinfo.value.__cause__ is exc


def test_reraise_if_web_extra_missing_matches_submodules():
    """`from fastapi.responses import ...` reports the submodule, not the root."""
    exc = ModuleNotFoundError("No module named 'fastapi.responses'", name="fastapi.responses")
    with pytest.raises(ModuleNotFoundError, match=r"consciousness-sim\[web\]"):
        reraise_if_web_extra_missing(exc)


def test_reraise_if_web_extra_missing_covers_uvicorn_and_starlette():
    for name in ("uvicorn", "starlette"):
        with pytest.raises(ModuleNotFoundError, match=r"consciousness-sim\[web\]"):
            reraise_if_web_extra_missing(ModuleNotFoundError(f"No module named '{name}'", name=name))


def test_reraise_if_web_extra_missing_ignores_unrelated_module():
    """A real bug inside FastAPI must not be relabelled as a missing extra."""
    assert reraise_if_web_extra_missing(
        ModuleNotFoundError("No module named 'numpy'", name="numpy")
    ) is None


def test_reraise_if_web_extra_missing_ignores_unnamed_error():
    assert reraise_if_web_extra_missing(ModuleNotFoundError("no name attribute")) is None


def test_importing_server_without_fastapi_names_the_extra(monkeypatch):
    """End-to-end: the failure a pip-installed copy hits without the extra."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] == "fastapi":
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "interfaces.web.server", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        importlib.import_module("interfaces.web.server")
    assert "consciousness-sim[web]" in str(excinfo.value)
