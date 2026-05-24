"""Discord webhook integration — stream consciousness events to a Discord channel."""

from interfaces.discord.webhook import DiscordWebhookSink, build_sink_from_config

__all__ = ["DiscordWebhookSink", "build_sink_from_config"]
