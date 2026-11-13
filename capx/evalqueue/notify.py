"""Messages for the organizers about things only a person can fix, such as accepting a
gated repo's terms on HuggingFace."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Protocol

_logger = logging.getLogger(__name__)

SLACK_WEBHOOK_ENV = "CAPX_SLACK_WEBHOOK_URL"

_TIMEOUT_S = 10.0


class Notifier(Protocol):
    def notify(self, text: str) -> None:
        """Never raises: a message that cannot be sent must not hold up the queue."""


class LogNotifier:
    def notify(self, text: str) -> None:
        _logger.warning("%s", text)


class SlackNotifier:
    """Posts to a Slack incoming webhook. Every message is logged as well."""

    def __init__(self, webhook_url: str) -> None:
        self._url = webhook_url

    def notify(self, text: str) -> None:
        _logger.warning("%s", text)
        request = urllib.request.Request(
            self._url,
            data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                response.read()
        except (OSError, ValueError) as exc:
            # The URL is the credential, and some of these exceptions print it.
            if isinstance(exc, urllib.error.HTTPError):
                detail = f"HTTP {exc.code}"
            else:
                detail = type(exc).__name__
            _logger.error("could not post to Slack: %s", detail)


def notifier_from_env() -> Notifier:
    url = os.environ.get(SLACK_WEBHOOK_ENV)
    if not url:
        _logger.warning(
            "%s is not set; notices for the organizers go to this log only", SLACK_WEBHOOK_ENV
        )
        return LogNotifier()
    return SlackNotifier(url)


__all__ = ["SLACK_WEBHOOK_ENV", "LogNotifier", "Notifier", "SlackNotifier", "notifier_from_env"]
