"""How ``capx.llm.client`` and the LLM proxies handle a model server that fails.

The client used to retry 404/500/502/503/504 forever, four minutes apart, and the
proxies turned every upstream error into a 500: a vLLM engine that died, or a model
name the server does not serve, hung every lane until the queue killed the job.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import pytest
import requests

from capx.llm import client as llm_client

_OK_BODY = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode()


class _Socket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class _Raw:
    """urllib3's response body: ``read1`` hands out the next chunk, b"" at the end."""

    def __init__(self, chunks: Iterable[bytes], on_read: Callable[[], None] | None = None) -> None:
        self._chunks = iter(chunks)
        self._on_read = on_read
        self.connection = SimpleNamespace(sock=_Socket())

    def read1(self, amt: int | None = None, decode_content: bool | None = None) -> bytes:
        if self._on_read is not None:
            self._on_read()
        return next(self._chunks, b"")


class _Response:
    def __init__(
        self,
        status_code: int,
        chunks: Iterable[bytes] = (_OK_BODY,),
        *,
        content_type: str = "application/json",
        on_read: Callable[[], None] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.raw = _Raw(chunks if status_code < 400 else [f"status {status_code}".encode()], on_read)

    def close(self) -> None:
        pass

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        model="Qwen/Qwen3.6-27B",
        server_url="http://127.0.0.1:8110/v1/chat/completions",
        api_key=None,
        temperature=0.0,
        max_tokens=64,
        reasoning_effort=None,
        debug=False,
    )


def _serve(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> list[dict[str, Any]]:
    """Answer each POST with the next reply (a status code or an exception)."""
    script = list(replies)
    sent: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> _Response:
        sent.append(kwargs)
        reply = script.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return _Response(reply)

    monkeypatch.setattr(llm_client.requests, "post", post)
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    return sent


@pytest.mark.parametrize(
    "transient",
    [503, 429, 502, 504, requests.ConnectionError("refused"), requests.ReadTimeout("slow")],
    ids=str,
)
def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch, transient: Any) -> None:
    sent = _serve(monkeypatch, transient, 200)
    assert llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "ok"
    assert len(sent) == 2


@pytest.mark.parametrize("status", [400, 401, 404, 500])
def test_an_error_that_would_repeat_is_raised_at_once(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    sent = _serve(monkeypatch, status, 200)
    with pytest.raises(requests.HTTPError, match=str(status)):
        llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])
    assert len(sent) == 1


def test_retries_stop_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPX_LLM_RETRY_DEADLINE_S", "60")
    sent = _serve(monkeypatch, *([503] * 50))
    clock = [0.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    with pytest.raises(requests.HTTPError, match="503"):
        llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])
    assert 2 <= len(sent) < 10
    assert clock[0] <= 60


def test_no_attempt_waits_past_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAPX_LLM_TIMEOUT", "900")
    monkeypatch.setenv("CAPX_LLM_RETRY_DEADLINE_S", "100")
    sent = _serve(monkeypatch, 200)
    llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])
    assert sent[0]["timeout"] <= 100


def test_the_server_url_is_used_for_every_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """``openrouter/...`` models went to a hard-coded localhost:8110 whatever
    ``--server-url`` said."""
    urls: list[str] = []

    def post(url: str, **kwargs: Any) -> _Response:
        urls.append(url)
        return _Response(200)

    monkeypatch.setattr(llm_client.requests, "post", post)
    args = _args()
    args.model = "openrouter/qwen/qwen3-235b-a22b"
    args.server_url = "http://127.0.0.1:9999/chat/completions"
    llm_client.query_model(args, [{"role": "user", "content": "hi"}])
    assert urls == ["http://127.0.0.1:9999/chat/completions"]


def test_streaming_reads_vllms_reasoning_and_falls_back_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming missed vLLM's ``reasoning_content`` deltas, so a reply still thinking
    at the token limit came back empty."""
    stream = (
        'data: {"choices": [{"delta": {"reasoning_content": "think "}}]}\r\n\r\n'
        'data: {"choices": [{"delta": {"reasoning_content": "more"}}]}\n\n'
        "data: [DONE]\n\n"
    ).encode()
    # Split mid-line, as a network read can.
    chunks = [stream[:30], stream[30:90], stream[90:]]
    monkeypatch.setattr(
        llm_client.requests,
        "post",
        lambda url, **kw: _Response(200, chunks, content_type="text/event-stream"),
    )
    chunks = list(llm_client.query_model_streaming(_args(), [{"role": "user", "content": "hi"}]))
    assert chunks[-1] == {"type": "done", "content": "think more", "reasoning": "think more"}


# --- the proxies -------------------------------------------------------------


def _proxy_client(
    monkeypatch: pytest.MonkeyPatch, create: Any, *, provider: str = "openai_server"
) -> Any:
    """A test client for ``capx.serving.<provider>.create_app``, whose upstream calls
    ``create``; the upstream client's own arguments land in ``create.client_kwargs``."""
    pytest.importorskip("fastapi")
    import importlib

    from fastapi.testclient import TestClient

    from capx.serving import openai_server

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            create.client_kwargs = kwargs
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(openai_server, "AsyncOpenAI", _FakeAsyncOpenAI)
    module = importlib.import_module(f"capx.serving.{provider}")
    return TestClient(module.create_app("key", "http://upstream/v1/"))


def _completion(model: str) -> Any:
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(
        {
            "id": "c",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}
            ],
        }
    )


def test_the_proxy_keeps_the_upstream_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx
    import openai

    async def create(**kwargs: Any) -> Any:
        response = httpx.Response(404, request=httpx.Request("POST", "http://upstream/v1/"))
        raise openai.NotFoundError("model not found", response=response, body=None)

    client = _proxy_client(monkeypatch, create)
    reply = client.post("/chat/completions", json={"model": "nope", "messages": []})
    assert reply.status_code == 404


def test_the_proxy_forwards_only_what_the_client_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults of max_tokens=256 and temperature=0.2 cut off every reply that did
    not set its own limit."""
    from openai.types.chat import ChatCompletion

    seen: dict[str, Any] = {}

    async def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return ChatCompletion.model_validate(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi", "reasoning": "r"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
        )

    client = _proxy_client(monkeypatch, create)
    reply = client.post(
        "/chat/completions",
        json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert reply.status_code == 200
    assert "max_tokens" not in seen and "temperature" not in seen
    body = reply.json()
    assert body["usage"]["completion_tokens"] == 2
    assert body["choices"][0]["message"]["reasoning"] == "r"


def _trickle(monkeypatch: pytest.MonkeyPatch, *, content_type: str) -> tuple[list[float], _Response]:
    """A server that sends one byte every 30 s and never finishes, on a fake clock."""
    monkeypatch.setenv("CAPX_LLM_RETRY_DEADLINE_S", "300")
    clock = [0.0]
    monkeypatch.setattr(llm_client.time, "monotonic", lambda: clock[0])
    response = _Response(
        200,
        iter(lambda: b" ", None),
        content_type=content_type,
        on_read=lambda: clock.__setitem__(0, clock[0] + 30.0),
    )
    monkeypatch.setattr(llm_client.requests, "post", lambda url, **kw: response)
    return clock, response


def test_a_trickled_reply_is_cut_off_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    """requests' timeout only bounds each wait on the socket, so a server sending a
    byte now and then held a request open past the deadline for as long as it liked."""
    clock, response = _trickle(monkeypatch, content_type="application/json")
    with pytest.raises(requests.Timeout):
        llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])
    assert clock[0] <= 300 + 30
    # No single read may wait past the deadline either.
    timeouts = response.raw.connection.sock.timeouts
    assert timeouts and all(0 < t <= 300 for t in timeouts)


def test_a_trickled_stream_is_cut_off_at_the_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    clock, _ = _trickle(monkeypatch, content_type="text/event-stream")
    with pytest.raises(requests.Timeout):
        list(llm_client.query_model_streaming(_args(), [{"role": "user", "content": "hi"}]))
    assert clock[0] <= 300 + 30


@pytest.mark.parametrize(
    ("sent", "upstream"),
    [
        ("openrouter/qwen/qwen3-235b-a22b", "qwen/qwen3-235b-a22b"),
        # A native OpenRouter name; the OpenAI proxy would strip "openai/".
        ("openai/gpt-4.1", "openai/gpt-4.1"),
    ],
)
def test_the_openrouter_app_sends_openrouters_names_and_headers(
    monkeypatch: pytest.MonkeyPatch, sent: str, upstream: str
) -> None:
    seen: dict[str, Any] = {}

    async def create(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return _completion(kwargs["model"])

    client = _proxy_client(monkeypatch, create, provider="openrouter_server")
    reply = client.post("/chat/completions", json={"model": sent, "messages": []})
    assert reply.status_code == 200
    assert seen["model"] == upstream
    assert create.client_kwargs["default_headers"]["X-Title"] == "CaP-X"


def test_a_connection_lost_mid_body_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """The retry loop ended once the headers arrived, so a connection dropped while
    the body was read failed the query with retry time left."""
    import urllib3

    def broken() -> None:
        raise urllib3.exceptions.ProtocolError("Connection broken: IncompleteRead")

    replies = iter([_Response(200, on_read=broken), _Response(200)])
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(llm_client.requests, "post", lambda url, **kw: sent.append(kw) or next(replies))
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: None)
    assert llm_client.query_model(_args(), [{"role": "user", "content": "hi"}])["content"] == "ok"
    assert len(sent) == 2


_CHUNKED_HEADERS = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nTransfer-Encoding: chunked\r\n\r\n"


@pytest.mark.parametrize(
    ("at_once", "trickled"),
    [
        # The status line and headers never finish.
        (b"", b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n" * 100),
        # The headers arrive; the body's chunk-size line never finishes, inside one read.
        (_CHUNKED_HEADERS, b"1" * 1000),
    ],
    ids=["headers", "chunk-framing"],
)
def test_a_trickling_server_is_cut_off_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch, at_once: bytes, trickled: bytes
) -> None:
    """requests' timeout bounds each wait on the socket, so a server sending a byte
    every 50 ms held the request past any deadline."""
    import socket
    import threading
    import time

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    stop = threading.Event()

    def trickle() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.recv(65536)
            conn.sendall(at_once)
            for byte in trickled:
                if stop.is_set():
                    return
                conn.sendall(bytes([byte]))
                time.sleep(0.05)

    threading.Thread(target=trickle, daemon=True).start()
    monkeypatch.setenv("CAPX_LLM_RETRY_DEADLINE_S", "1")
    args = _args()
    args.server_url = f"http://127.0.0.1:{listener.getsockname()[1]}/v1/chat/completions"
    started = time.monotonic()
    try:
        with pytest.raises(requests.Timeout):
            llm_client.query_model(args, [{"role": "user", "content": "hi"}])
        assert time.monotonic() - started < 2.0
    finally:
        stop.set()
        listener.close()



def test_ensemble_votes_are_counted_by_the_decision_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Counting the words anywhere made "FINISH. No need to REGENERATE." a vote for
    both, and the synthesis prompt then reported more votes to regenerate than any
    candidate cast."""
    synthesis: list[str] = []

    def query_model(args: Any, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        if prompt[0]["role"] == "system":
            synthesis.append(prompt[0]["content"])
            return {"content": "FINISH", "reasoning": None, "usage": {}}
        if args.temperature <= 0.3:
            return {"content": "REGENERATE\n```python\nx = 1\n```", "reasoning": None}
        return {"content": "FINISH. No need to REGENERATE.", "reasoning": None}

    monkeypatch.setattr(llm_client, "query_model", query_model)
    llm_client.query_single_model_ensemble(
        _args(), [{"role": "user", "content": [{"type": "text", "text": "go"}]}], "m", is_multiturn=True
    )
    assert "- 3 candidates voted REGENERATE" in synthesis[0]
    assert "- 6 candidates voted FINISH" in synthesis[0]


def test_ensemble_leaves_replies_cut_off_at_the_token_limit_out_of_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nine replies still thinking at the token limit overflowed the synthesis prompt."""
    synthesis: list[str] = []

    def query_model(args: Any, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        if prompt[0]["role"] == "system":
            synthesis.append(prompt[1]["content"][0]["text"])
            return {"content": "FINISH", "reasoning": None, "usage": {}}
        if args.temperature <= 0.2:
            return {"content": "FINISH", "reasoning": None, "truncated": False}
        return {"content": "thinking " * 100, "reasoning": None, "truncated": True}

    monkeypatch.setattr(llm_client, "query_model", query_model)
    llm_client.query_single_model_ensemble(
        _args(), [{"role": "user", "content": [{"type": "text", "text": "go"}]}], "m", is_multiturn=True
    )
    assert synthesis[0].count("--- Candidate") == 2
    assert "thinking" not in synthesis[0]


def test_ensemble_falls_back_to_a_majority_candidate_when_synthesis_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A synthesis prompt over the model's context length was a 400, which ended the trial."""

    def query_model(args: Any, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        if prompt[0]["role"] == "system":
            raise requests.HTTPError("model server answered 400: maximum context length")
        if args.temperature <= 0.3:
            return {"content": f"FINISH ({args.temperature})", "reasoning": None}
        return {"content": f"REGENERATE\n```python\nx = {args.temperature}\n```", "reasoning": None}

    monkeypatch.setattr(llm_client, "query_model", query_model)
    prompt = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    multi_turn = llm_client.query_single_model_ensemble(_args(), prompt, "m", is_multiturn=True)
    assert multi_turn["content"] == "REGENERATE\n```python\nx = 0.4\n```"
    code = llm_client.query_single_model_ensemble(_args(), prompt, "m")
    assert code["content"] == "FINISH (0.1)"
