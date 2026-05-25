"""Discord webhook sink — posts consciousness events to a Discord channel.

Theory mapping — GWT-3 (global broadcast): extends the workspace broadcast to a
Discord channel, making the agent's state observable by remote humans without a
terminal or browser. Read-only — observers cannot write back to the workspace.

Design notes:
- Subscribes to the same `on_*` event channels as the web dashboard. No coupling
  between the two — either, both, or neither can be active per instance.
- `memory_stored` is excluded by default (one ping per cycle is noise, not signal).
- The webhook URL is treated as a secret. It never appears in logs, exception
  messages, or repr — replaced with a masked form everywhere.
- Token-bucket limits outbound rate to stay under Discord's ~30/min sustained
  threshold. On overflow: drop (default) or queue.
- All HTTP failures are swallowed and logged as WARNING — a Discord outage must
  never break a thought cycle.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Bubble colors from interfaces/web/static/index.html (PR #52) — same observer
# mental model across the dashboard and Discord.
_EVENT_COLOR: dict[str, int] = {
    "thought":        0x7c83ff,
    "reflection":     0xb48eff,
    "perception":     0xfbbf24,
    "identity_shift": 0x4ade80,
    "memory":         0x8b8fa8,   # muted — opt-in event type
}

# Discord allows up to 4096 chars in embed.description; keep headroom for the
# truncation suffix. The user-facing config knob is truncate_chars.
_DEFAULT_TRUNCATE = 1800
_TRUNCATE_SUFFIX = "… [truncated]"

_ALLOWED_HOSTS = frozenset({"discord.com", "discordapp.com"})


def _mask_url(url: str) -> str:
    """Replace webhook id + token with `***` so the URL is safe to log."""
    if not url:
        return ""
    # Discord webhook URL shape: https://discord.com/api/webhooks/<id>/<token>
    return re.sub(
        r"(/api/webhooks/)[^/]+/[^/?#]+",
        r"\1***/***",
        url,
    )


class _MaskingFilter(logging.Filter):
    """Logging filter that scrubs any webhook URL substring from emitted records.

    Belt-and-braces: even if a code path forgets to call _mask_url, this filter
    ensures the raw secret never reaches a handler.
    """

    def __init__(self, secret: str) -> None:
        super().__init__()
        self._secret = secret
        self._masked = _mask_url(secret)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str) and self._secret in record.msg:
                record.msg = record.msg.replace(self._secret, self._masked)
            if record.args:
                new_args = tuple(
                    a.replace(self._secret, self._masked) if isinstance(a, str) and self._secret in a else a
                    for a in record.args
                )
                record.args = new_args
        except Exception:
            pass
        return True


class _TokenBucket:
    """Tiny token-bucket rate limiter — refills `rate_per_min` tokens per minute."""

    def __init__(self, rate_per_min: int) -> None:
        self._capacity = float(rate_per_min)
        self._tokens = float(rate_per_min)
        self._refill_per_sec = float(rate_per_min) / 60.0
        self._updated_at = time.monotonic()

    def try_consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self._updated_at
        self._updated_at = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


class DiscordWebhookSink:
    """Posts consciousness events to a Discord channel via webhook.

    Construct, then call `register(mind)` to subscribe to the events listed in
    `events`. Each event triggers an async POST; failures are logged but never
    raised to the caller.
    """

    def __init__(
        self,
        webhook_url: str,
        *,
        events: set[str] | list[str] | tuple[str, ...],
        rate_limit_per_min: int = 25,
        username: str | None = None,
        avatar_url: str | None = None,
        truncate_chars: int = _DEFAULT_TRUNCATE,
        include_perception_url: bool = True,
    ) -> None:
        self._validate_url(webhook_url)
        self.webhook_url = webhook_url
        self.events = set(events)
        self.username = username
        self.avatar_url = avatar_url
        self.truncate_chars = max(64, int(truncate_chars))
        self.include_perception_url = bool(include_perception_url)
        self._bucket = _TokenBucket(rate_limit_per_min)
        self._dropped_since_last_warn = 0
        # Sentinel: -inf guarantees the first drop's `now - _last >= 60.0` check fires
        # regardless of `time.monotonic()`'s origin (which is undefined — on Linux
        # it's seconds-since-boot, so 0.0 fails the comparison on freshly-booted CI runners).
        self._last_drop_warning_at: float = -math.inf

        # Install masking filter on the module logger so the URL is never leaked
        # even if a future code path or third-party traceback formats it raw.
        logger.addFilter(_MaskingFilter(webhook_url))

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _validate_url(url: str) -> None:
        if not url or "${" in url:
            raise ValueError(
                "Discord webhook_url is empty or contains an unresolved ${VAR} "
                "reference — set the env var before launching."
            )
        host = (urlparse(url).hostname or "").lower()
        if host not in _ALLOWED_HOSTS:
            raise ValueError(
                f"Discord webhook host {host!r} not in allowlist {sorted(_ALLOWED_HOSTS)}. "
                "Refusing to send — check the URL for typos."
            )

    @property
    def masked_url(self) -> str:
        return _mask_url(self.webhook_url)

    def __repr__(self) -> str:
        return f"DiscordWebhookSink(url={self.masked_url!r}, events={sorted(self.events)})"

    # ------------------------------------------------------------------ wiring

    def register(self, mind: Any) -> None:
        """Hook this sink's `_post` into the requested event channels on `mind`."""
        for event_type in self.events:
            channel = getattr(mind, f"on_{event_type}", None)
            if channel is None:
                logger.warning("Discord sink: no on_%s channel on mind — skipping", event_type)
                continue

            async def _handler(payload: dict[str, Any], _self=self) -> None:
                await _self._post(payload)

            channel.append(_handler)
        # Bind the agent's name as the webhook username if not explicitly set
        if self.username is None and getattr(mind, "name", None):
            self.username = str(mind.name)
        logger.info(
            "Discord sink registered for events=%s (url=%s, rate=%d/min)",
            sorted(self.events), self.masked_url,
            int(self._bucket._capacity),
        )

    # ------------------------------------------------------------------ post

    async def _post(self, payload: dict[str, Any]) -> None:
        if not self._bucket.try_consume():
            self._record_drop()
            return

        body = self._build_body(payload)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.webhook_url, json=body)
                # 2xx + 204 are success; 429 is rate-limit (Discord-side)
                if response.status_code == 429:
                    logger.warning(
                        "Discord sink: rate-limited by Discord (429) for %s",
                        self.masked_url,
                    )
                elif response.status_code >= 400:
                    logger.warning(
                        "Discord sink: POST returned HTTP %d for %s",
                        response.status_code, self.masked_url,
                    )
        except Exception as exc:
            # Mask the URL out of the exception text too — some httpx errors
            # include it in the message.
            msg = str(exc).replace(self.webhook_url, self.masked_url)
            logger.warning("Discord sink: POST failed (%s)", msg)

    def _record_drop(self) -> None:
        """Throttle the drop-warning log so a sustained burst emits at most one WARNING per minute."""
        self._dropped_since_last_warn += 1
        now = time.monotonic()
        if now - self._last_drop_warning_at >= 60.0:
            logger.warning(
                "Discord sink: rate-limit reached — dropped %d event(s) in the last minute",
                self._dropped_since_last_warn,
            )
            self._dropped_since_last_warn = 0
            self._last_drop_warning_at = now

    # --------------------------------------------------------------- format

    def _build_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = str(payload.get("type", "thought"))
        content = str(payload.get("content", ""))
        truncated = self._truncate(content)

        # Per-type footer text
        if event_type == "perception":
            src = payload.get("source") or "external"
            title = payload.get("title") or ""
            footer_text = f"perception · {src}: {title}".strip(": ").strip()
        elif event_type == "identity_shift":
            footer_text = "identity shift · self-concept updated"
        else:
            footer_text = event_type

        embed: dict[str, Any] = {
            "description": truncated,
            "color": _EVENT_COLOR.get(event_type, _EVENT_COLOR["thought"]),
            "footer": {"text": footer_text},
        }

        # Perceptions can carry a source URL — let Discord auto-unfurl it
        if event_type == "perception" and self.include_perception_url:
            url = payload.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                embed["url"] = url
                if payload.get("title"):
                    embed["title"] = str(payload["title"])

        body: dict[str, Any] = {"embeds": [embed]}
        if self.username:
            body["username"] = self.username
        if self.avatar_url:
            body["avatar_url"] = self.avatar_url
        return body

    def _truncate(self, text: str) -> str:
        if len(text) <= self.truncate_chars:
            return text
        keep = max(0, self.truncate_chars - len(_TRUNCATE_SUFFIX))
        return text[:keep].rstrip() + _TRUNCATE_SUFFIX


# ---------------------------------------------------------------------- factory

def build_sink_from_config(discord_cfg: dict[str, Any]) -> DiscordWebhookSink | None:
    """Construct a sink from a config dict, returning None when disabled or unset.

    Raises ValueError if enabled with an invalid URL / unresolved env var /
    non-allowlisted host — fail loudly on misconfiguration at startup.
    """
    if not discord_cfg or not bool(discord_cfg.get("enabled", False)):
        return None

    rate_cfg = dict(discord_cfg.get("rate_limit", {}) or {})
    return DiscordWebhookSink(
        webhook_url=str(discord_cfg["webhook_url"]),
        events=list(discord_cfg.get("events", ["thought", "reflection", "perception", "identity_shift"])),
        rate_limit_per_min=int(rate_cfg.get("max_per_minute", 25)),
        username=discord_cfg.get("username") or None,
        avatar_url=discord_cfg.get("avatar_url") or None,
        truncate_chars=int(discord_cfg.get("truncate_chars", _DEFAULT_TRUNCATE)),
        include_perception_url=bool(discord_cfg.get("include_perception_url", True)),
    )
