"""Optional-dependency guard for the web dashboard.

No direct theory mapping — packaging support code.

`interfaces.web.server` imports FastAPI at module scope and uvicorn inside
`start()`, but the distribution declares neither as a required dependency:
both live in the ``web`` extra (issue #169). This module holds the names and
the guidance message so the check itself stays importable without FastAPI
installed — the guard has to run *before* the failing import is resolved.
"""

from __future__ import annotations

# Top-level module names whose absence means the ``web`` extra is missing.
# starlette is included because FastAPI re-exports from it, so a partial
# install surfaces as a missing starlette rather than a missing fastapi.
WEB_EXTRA_MODULES = frozenset({"fastapi", "starlette", "uvicorn"})

WEB_EXTRA_HINT = (
    "The web dashboard needs the optional 'web' extra: "
    "install it with \"pip install 'consciousness-sim[web]'\", or install the "
    'development requirements with "pip install -r requirements.txt".'
)


def reraise_if_web_extra_missing(exc: ModuleNotFoundError) -> None:
    """Raise an actionable error when *exc* names a package from the ``web`` extra.

    Returns normally when the missing module is unrelated, leaving the caller
    to re-raise the original error untouched — a genuine ``ModuleNotFoundError``
    from inside FastAPI must not be mislabelled as an uninstalled extra.
    """
    if (exc.name or "").split(".")[0] not in WEB_EXTRA_MODULES:
        return
    raise ModuleNotFoundError(f"{exc}. {WEB_EXTRA_HINT}", name=exc.name) from exc
