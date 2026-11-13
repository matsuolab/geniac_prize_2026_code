"""Notices for the organizers: posted to Slack when a webhook is set, logged always."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from email.message import Message

import pytest

from capx.evalqueue import notify
from capx.evalqueue.notify import LogNotifier, SlackNotifier, notifier_from_env

URL = "https://hooks.slack.com/services/T000/B000/secret"


class _Response:
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_exc: object) -> None:
        pass

    def read(self) -> bytes:
        return b"ok"


def test_a_notice_is_posted_as_slack_text(monkeypatch, caplog) -> None:
    sent: list[urllib.request.Request] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        sent.append(request)
        return _Response()

    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    with caplog.at_level(logging.WARNING):
        SlackNotifier(URL).notify("approve org/model")

    (request,) = sent
    assert request.full_url == URL and request.get_method() == "POST"
    assert isinstance(request.data, bytes)
    assert json.loads(request.data) == {"text": "approve org/model"}
    assert "approve org/model" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.URLError(f"cannot reach {URL}"),
        urllib.error.HTTPError(URL, 404, "Not Found", hdrs=Message(), fp=None),
        TimeoutError(),
    ],
)
def test_a_failed_post_neither_raises_nor_logs_the_webhook(monkeypatch, caplog, error) -> None:
    def urlopen(_request: urllib.request.Request, timeout: float) -> _Response:
        raise error

    monkeypatch.setattr(notify.urllib.request, "urlopen", urlopen)
    with caplog.at_level(logging.WARNING):
        SlackNotifier(URL).notify("approve org/model")
    assert "could not post to Slack" in caplog.text
    assert "secret" not in caplog.text


def test_without_a_webhook_notices_only_go_to_the_log(monkeypatch) -> None:
    monkeypatch.delenv(notify.SLACK_WEBHOOK_ENV, raising=False)
    assert isinstance(notifier_from_env(), LogNotifier)
    monkeypatch.setenv(notify.SLACK_WEBHOOK_ENV, URL)
    assert isinstance(notifier_from_env(), SlackNotifier)
