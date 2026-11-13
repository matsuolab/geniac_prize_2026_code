"""A reply that arrived as thinking must not read as no reply at all.

vLLM's ``--reasoning-parser`` splits a completion in two: ``content`` carries the
answer and ``reasoning_content`` the thinking. A model still thinking when it hits
the token limit produces an empty ``content`` with everything in
``reasoning_content`` — and both readers of that response treated it as nothing.

It cost more than a lost turn. In the agent loop the ``None`` reached
``_extract_code``, which raised out of the lane's thread and ended a whole
submission. In the fallback object localizer every one of 795 replies came back
empty, so "object not found" was final on 130 LIBERO tasks.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest

MOLMO = "capx.integrations.vision.molmo"


def _molmo(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CAPX_MOLMO_URL", "http+unix://model.sock/v1")
    sys.modules.pop(MOLMO, None)
    return importlib.import_module(MOLMO)


def _fake_session(monkeypatch, molmo, message: dict) -> dict:
    sent: dict = {}

    class FakeSession:
        def post(self, url, json, headers, timeout):  # noqa: ANN001
            sent.update(json)

            class Response:
                def raise_for_status(self) -> None:
                    pass

                @staticmethod
                def json() -> dict:
                    return {"choices": [{"message": message}]}

            return Response()

    monkeypatch.setattr(molmo.unix_http_client, "new_session", lambda: FakeSession())
    return sent


def test_a_point_that_arrived_as_thinking_is_still_a_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molmo = _molmo(monkeypatch)
    _fake_session(
        monkeypatch,
        molmo,
        {"content": None, "reasoning_content": '<point x="50" y="50">stove</point>'},
    )
    from PIL import Image

    found = molmo.init_molmo()(Image.new("RGB", (100, 100)), objects=["stove"])
    assert found["stove"] != (None, None)


def test_the_answer_still_wins_when_both_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molmo = _molmo(monkeypatch)
    _fake_session(
        monkeypatch,
        molmo,
        {
            "content": '<point x="10" y="10">stove</point>',
            "reasoning_content": '<point x="90" y="90">stove</point>',
        },
    )
    from PIL import Image

    found = molmo.init_molmo()(Image.new("RGB", (100, 100)), objects=["stove"])
    assert found["stove"] == (10, 10)


def test_the_localizer_asks_the_model_not_to_think(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pointing at a named object is a look-up. Spending the token budget thinking
    is how every reply came back empty in the first place."""
    molmo = _molmo(monkeypatch)
    sent = _fake_session(monkeypatch, molmo, {"content": ""})
    from PIL import Image

    molmo.init_molmo()(Image.new("RGB", (100, 100)), objects=["stove"])
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"content": "answer", "reasoning_content": "thought"}, "answer"),
        ({"content": None, "reasoning_content": "thought"}, "thought"),
        ({"content": "", "reasoning_content": "thought"}, "thought"),
        ({"content": "answer"}, "answer"),
    ],
)
def test_the_agent_client_reads_a_reply_that_only_thought(
    monkeypatch: pytest.MonkeyPatch, message, expected
) -> None:
    from capx.llm import client as llm_client

    import json

    class Raw:
        connection = None

        def __init__(self) -> None:
            self._chunks = [json.dumps({"choices": [{"message": message}], "usage": {}}).encode()]

        def read1(self, amt=None, decode_content=None) -> bytes:  # noqa: ANN001
            return self._chunks.pop() if self._chunks else b""

    class Response:
        status_code = 200

        def __init__(self) -> None:
            self.raw = Raw()

        def close(self) -> None:
            pass

    monkeypatch.setattr(llm_client.unix_http_client, "post", lambda *a, **k: Response())
    args = SimpleNamespace(
        model="Qwen/Qwen3.6-27B",
        server_url="http+unix://model.sock/v1/chat/completions",
        api_key=None,
        temperature=0.0,
        max_tokens=64,
        reasoning_effort=None,
        debug=False,
    )
    out = llm_client.query_model(args, [{"role": "user", "content": "hi"}])
    assert out["content"] == expected
