"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import copy
import functools
import json
import os
import queue
import random
import re
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

import requests
import urllib3

from capx.utils import turn_timing, unix_http_client

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

_T = TypeVar("_T")

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "openai/gpt-5.4",
    "openai/o4-mini",
]
# The web UI's list of models that take images. Trials (capx.envs.trial) go by the
# config's use_visual_feedback / use_img_differencing / use_video_differencing alone.
VLM_MODELS = [
    "Qwen/Qwen3.6-27B",
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

# (model, temperatures) the multimodel ensemble (use_multimodel) queries.
ENSEMBLE_CONFIGS = [
    ("openai/gpt-5.4", [0.1, 0.5, 0.9]),
]
SINGLE_MODEL_ENSEMBLE_TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------

# A request that got no answer, or an answer saying the server is overloaded or
# behind a gateway that timed out, is worth sending again; any other error status
# (a 400 for an overlong prompt, a 404 for a model the server does not serve, a 500
# from a dead engine) would only come back the same.
_RETRY_STATUS = frozenset({429, 502, 503, 504})
_FIRST_BACKOFF_S = 5.0
_MAX_BACKOFF_S = 120.0
_READ_SIZE = 64 * 1024
# How long an error answer's body may take to arrive, for its message.
_ERROR_BODY_S = 10.0


def _payload(
    args: LaunchArgs | ModelQueryArgs, prompt: list[dict], *, stream: bool
) -> dict[str, Any]:
    if args.model in GPT_MODELS:
        payload: dict[str, Any] = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            # Reasoning and answer together.
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
        if args.model in CLAUDE_MODELS:
            payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}
    if stream:
        payload["stream"] = True
    return payload


def _headers(args: LaunchArgs | ModelQueryArgs) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
    return headers


def _deadline() -> float:
    """When a query gives up: ``CAPX_LLM_RETRY_DEADLINE_S`` from now, retries and
    reading the reply included."""
    return time.monotonic() + float(os.getenv("CAPX_LLM_RETRY_DEADLINE_S", "1800"))


def _body(response: requests.Response, deadline: float) -> Iterator[bytes]:
    """The response body as it arrives, cut off at ``deadline`` with ``requests.Timeout``.

    requests' timeout bounds each wait on the socket, and one read of a chunked
    body can wait several times (the chunk-size line, the trailers), so a server
    that sends a byte now and then could hold a read open as long as it liked. The
    reads run on a daemon thread, and the caller stops waiting for them at the
    deadline; the thread stops reading once the deadline has passed.
    """
    chunks: queue.Queue[bytes | Exception | None] = queue.Queue()
    abandoned = threading.Event()

    def read() -> None:
        raw = response.raw
        sock = raw.connection.sock if raw.connection is not None else None
        try:
            while not abandoned.is_set():
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    return
                if sock is not None:
                    sock.settimeout(remaining_s)
                chunk = raw.read1(_READ_SIZE, decode_content=True)
                chunks.put(chunk or None)
                if not chunk:
                    return
        except urllib3.exceptions.ReadTimeoutError as exc:
            chunks.put(requests.Timeout(str(exc)))
        except urllib3.exceptions.ProtocolError as exc:
            chunks.put(requests.ConnectionError(str(exc)))
        except Exception as exc:
            chunks.put(exc)
        finally:
            response.close()

    threading.Thread(target=read, daemon=True).start()
    try:
        while True:
            try:
                item = chunks.get(timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Empty:
                raise requests.Timeout("model server reply still incomplete at the deadline") from None
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        abandoned.set()


def _lines(chunks: Iterable[bytes]) -> Iterator[bytes]:
    """The lines of a byte stream, without their line endings."""
    rest = b""
    for chunk in chunks:
        *lines, rest = (rest + chunk).split(b"\n")
        for line in lines:
            yield line.rstrip(b"\r")
    if rest:
        yield rest.rstrip(b"\r")


def _status_error(response: requests.Response, deadline: float) -> requests.HTTPError:
    try:
        body = b"".join(_body(response, min(deadline, time.monotonic() + _ERROR_BODY_S)))
        text = body.decode(errors="replace")[:2000]
    except requests.RequestException:
        text = "(no body)"
    return requests.HTTPError(
        f"model server answered {response.status_code}: {text}", response=response
    )


def _before(deadline: float, call: Callable[[], _T]) -> _T:
    """``call()``'s result, or ``requests.Timeout`` once ``deadline`` has passed.

    requests' timeout bounds each wait on the socket, so a server that trickles
    the bytes of its status line and headers would hold ``requests.post`` past any
    deadline. The call runs on a daemon thread, left behind if the deadline comes
    first; a response it gets after that is closed.
    """
    outcome: list[Any] = []
    abandoned = threading.Event()

    def run() -> None:
        try:
            result = call()
        except BaseException as exc:
            outcome.append(exc)
            return
        outcome.append(result)
        if abandoned.is_set() and isinstance(result, requests.Response):
            result.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(max(deadline - time.monotonic(), 0.0))
    if not outcome:
        abandoned.set()
        raise requests.Timeout("model server did not answer before the deadline")
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]


def _post(
    args: LaunchArgs | ModelQueryArgs,
    payload: dict[str, Any],
    deadline: float,
    *,
    read_body: bool,
) -> tuple[requests.Response, bytes | None, int]:
    """POST ``payload`` to the model server; the response, its body if ``read_body``
    (else left unread, for a stream), and how many retries it took.

    A transient failure, including a connection lost while the body is read, is
    retried with backoff until ``deadline``; each attempt waits at most
    ``CAPX_LLM_TIMEOUT`` for the reply to start, and nothing waits past the deadline.
    """
    attempt_timeout_s = float(os.getenv("CAPX_LLM_TIMEOUT", "900"))
    headers = _headers(args)
    data = json.dumps(payload)
    retries = 0
    while True:
        remaining_s = max(deadline - time.monotonic(), 1.0)
        error: Exception
        try:
            response = _before(
                deadline,
                functools.partial(
                    unix_http_client.post,
                    args.server_url,
                    headers=headers,
                    data=data,
                    timeout=min(attempt_timeout_s, remaining_s),
                    stream=True,
                ),
            )
            if response.status_code < 400:
                body = b"".join(_body(response, deadline)) if read_body else None
                return response, body, retries
            error = _status_error(response, deadline)
            if response.status_code not in _RETRY_STATUS:
                raise error
        except (requests.ConnectionError, requests.Timeout) as exc:
            error = exc
        backoff_s = min(_FIRST_BACKOFF_S * 2**retries, _MAX_BACKOFF_S) * random.uniform(0.75, 1.25)
        if time.monotonic() + backoff_s > deadline:
            raise error
        print(f"Retry {retries + 1}: model query failed ({error}); retrying in {backoff_s:.0f}s")
        time.sleep(backoff_s)
        retries += 1


def _reply(message: dict[str, Any]) -> tuple[str | None, str | None]:
    """A chat message's answer and reasoning.

    A server running a reasoning parser (vLLM's --reasoning-parser) splits the reply:
    ``content`` holds the answer and the reasoning goes under ``reasoning_content``
    (vLLM) or ``reasoning`` (OpenRouter). A model still thinking at the token limit
    leaves ``content`` empty; the reasoning then stands in for the answer rather
    than reaching the code extractor as None.
    """
    reasoning = message.get("reasoning") or message.get("reasoning_content")
    return message.get("content") or reasoning, reasoning


def query_model(args: LaunchArgs | ModelQueryArgs, prompt: list[dict]) -> dict[str, Any]:
    """Query the model server at ``args.server_url`` for one reply.

    Returns ``{"content", "reasoning", "usage"}``.
    """
    start_time = time.time()
    deadline = _deadline()
    _, body_bytes, retries = _post(
        args, _payload(args, prompt, stream=False), deadline, read_body=True
    )
    body = json.loads(body_bytes or b"")
    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        choice = body["choices"][0]
        content, reasoning = _reply(choice["message"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    out: dict[str, Any] = {
        "content": content,
        "reasoning": reasoning,
        "truncated": choice.get("finish_reason") == "length",
    }

    # Record token usage so callers can write input/output/reasoning(thinking)
    # token counts into the per-trial summary. Different providers / model
    # families expose these under slightly different keys; we normalize here.
    usage = body.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    out["usage"] = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        # gpt-5 / o-series reasoning models expose hidden reasoning ("thinking")
        # tokens here. They are still billed at the output rate.
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "cached_prompt_tokens": prompt_details.get("cached_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }
    # Convenience aliases requested by the eval harness:
    #   input_tokens     = prompt_tokens
    #   output_tokens    = completion_tokens (includes reasoning if any)
    #   thinking_tokens  = reasoning_tokens
    out["usage"]["input_tokens"] = out["usage"]["prompt_tokens"]
    out["usage"]["output_tokens"] = out["usage"]["completion_tokens"]
    out["usage"]["thinking_tokens"] = out["usage"]["reasoning_tokens"]

    print(
        "[llm.client] tokens "
        f"input={out['usage']['input_tokens']} "
        f"output={out['usage']['output_tokens']} "
        f"thinking={out['usage']['thinking_tokens']}"
    )
    # Retries back off, so a turn that looks slow is worth separating from a turn
    # that was slow.
    turn_timing.record_call(
        model=args.model,
        elapsed_s=end_time - start_time,
        retries=retries,
        usage=out["usage"],
    )
    return out


def query_model_streaming(
    args: LaunchArgs | ModelQueryArgs,
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}
    """
    start_time = time.time()
    deadline = _deadline()
    response, _, _ = _post(args, _payload(args, prompt, stream=True), deadline, read_body=False)
    with response:
        content_type = response.headers.get("content-type", "")
        # A server that does not stream answers with one JSON body.
        if "application/json" in content_type and "text/event-stream" not in content_type:
            body = json.loads(b"".join(_body(response, deadline)))
            try:
                content, reasoning = _reply(body["choices"][0]["message"])
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc
            yield {"type": "content_delta", "content": content or ""}
            yield {"type": "done", "content": content or "", "reasoning": reasoning}
            elapsed_s = time.time() - start_time
            print(f"Time taken to query model (streaming fallback): {elapsed_s:.2f} seconds")
            return

        full_content = ""
        full_reasoning = ""
        for line in _lines(_body(response, deadline)):
            if not line:
                continue
            data_str = line.decode("utf-8")
            # SSE lines are "data: {...}" or "data: [DONE]"; anything else is raw JSON.
            if data_str.startswith("data: "):
                data_str = data_str[len("data: ") :]
                if data_str == "[DONE]":
                    break
            try:
                choices = json.loads(data_str).get("choices") or []
            except json.JSONDecodeError:
                continue
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content_delta = delta.get("content")
            if content_delta:
                full_content += content_delta
                yield {"type": "content_delta", "content": content_delta}
            reasoning_delta = delta.get("reasoning") or delta.get("reasoning_content")
            if reasoning_delta:
                full_reasoning += reasoning_delta
                yield {"type": "reasoning_delta", "content": reasoning_delta}

    print(f"Time taken to query model (streaming): {time.time() - start_time:.2f} seconds")
    yield {
        "type": "done",
        "content": full_content or full_reasoning,
        "reasoning": full_reasoning or None,
    }


# A decision is a line that starts with the word, perhaps under markdown emphasis
# (**FINISH**, __REGENERATE__). Not a substring anywhere: "FINISH. No need to
# REGENERATE." is a finish.
_DECISION_LINE = re.compile(
    r"^[\s*#>_]*(REGENERATE|FINISH)[*_]*(?![A-Za-z0-9])", re.MULTILINE
)


def parse_multi_turn_decision(content: str | None) -> tuple[str, str | None]:
    """The reply's decision and, for "regenerate", the text after the word.

    The first line that starts with REGENERATE or FINISH decides: a synthesis reply
    may give its reasoning first. A reply with neither is a finish.
    """
    text = content or ""
    match = _DECISION_LINE.search(text)
    if match is not None and match.group(1) == "REGENERATE":
        return "regenerate", text[match.end():].strip()
    return "finish", content


def _vote(candidate: dict[str, Any]) -> str | None:
    """The decision an ensemble candidate states, if any; only those count as votes."""
    content = candidate.get("content")
    match = _DECISION_LINE.search(content) if isinstance(content, str) else None
    return match.group(1) if match else None


def _fallback_reply(
    candidates: list[dict[str, Any]], votes: list[str], *, is_multiturn: bool
) -> dict[str, Any]:
    """The lowest-temperature candidate, among those voting the majority decision
    when there are votes; a tie goes to REGENERATE."""
    pool = sorted(candidates, key=lambda r: r["temp"])
    if is_multiturn and votes:
        majority = "REGENERATE" if votes.count("REGENERATE") >= votes.count("FINISH") else "FINISH"
        pool = [r for r in pool if _vote(r) == majority]
    return {"content": pool[0]["content"], "reasoning": None}


def _synthesis_system_prompt(
    n_candidates: int, regenerate: int, finish: int, *, is_multiturn: bool
) -> str:
    if is_multiturn:
        return f"""You are synthesizing {n_candidates} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate} candidates voted REGENERATE
    - {finish} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    return f"""You are synthesizing {n_candidates} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """


def _ensemble(
    args: LaunchArgs | ModelQueryArgs,
    prompt: list[dict],
    candidates: list[tuple[str, float]],
    synthesis_model: str,
    *,
    is_multiturn: bool,
    log_name: str,
    name_models: bool,
) -> dict[str, Any]:
    """Query every ``(model, temperature)`` in parallel, then have
    ``synthesis_model`` synthesize one answer from the replies."""
    # The candidates run on pool threads of their own, where the lane's timer is
    # not visible; without this their calls go unrecorded.
    lane_timer = turn_timing.active()

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        try:
            with turn_timing.bind(lane_timer):
                result = query_model(query_args, copy.deepcopy(prompt))
            return {
                "model": model,
                "temp": temp,
                "content": result["content"],
                "truncated": result.get("truncated", False),
                "ok": True,
            }
        except Exception as e:
            print(f"[{log_name}] {model} temp={temp} FAILED: {e}")
            return {"model": model, "temp": temp, "content": str(e), "ok": False}

    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as executor:
        futures = [executor.submit(query_single, m, t) for m, t in candidates]
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp["ok"]:
                print(f"[{log_name}] {resp['model']} temp={resp['temp']} ok=True")

    successful = [r for r in responses if r["ok"]]
    if not successful:
        print(f"\n=== All {log_name.lower()} queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        raise RuntimeError(f"All {log_name.lower()} queries failed")
    # A reply cut off at the token limit has no complete answer, and a few of them
    # overflow the synthesis prompt.
    complete = [r for r in successful if not r["truncated"]] or successful

    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidate_text = "\n\n".join(
        f"--- Candidate ({r['model'] + ', ' if name_models else ''}temp={r['temp']}) ---\n"
        f"{r['content']}"
        for r in complete
    )
    votes = [vote for r in complete if (vote := _vote(r))]
    synthesis_system_prompt = _synthesis_system_prompt(
        len(complete),
        votes.count("REGENERATE"),
        votes.count("FINISH"),
        is_multiturn=is_multiturn,
    )
    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidate_text}
    </candidate_solutions>
    """
    synthesis_prompt = [
        {"role": "system", "content": synthesis_system_prompt},
        {"role": "user", "content": [{"type": "text", "text": synthesis_user_prompt}]},
    ]
    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    try:
        final = query_model(synth_args, synthesis_prompt)
    except requests.RequestException as exc:
        print(f"[{log_name}] synthesis failed, using one candidate's reply: {exc}")
        final = _fallback_reply(complete, votes, is_multiturn=is_multiturn)

    rule = "=" * 60
    candidates_txt = "\n\n".join(
        f"{rule}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{rule}\n"
        f"{r['content']}"
        for r in responses
    )
    synthesis_txt = (
        f"Model: {synthesis_model}\n\n"
        f"{rule}\nREASONING\n{rule}\n{final.get('reasoning') or '(none)'}\n\n"
        f"{rule}\nOUTPUT\n{rule}\n{final['content']}"
    )

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_model_ensemble(
    args: LaunchArgs | ModelQueryArgs,
    prompt: list[dict],
    synthesis_model: str = "openai/gpt-5.4",
    is_multiturn: bool = False,
) -> dict[str, Any]:
    """Query every (model, temperature) in ``ENSEMBLE_CONFIGS`` and synthesize one answer."""
    candidates = [(model, temp) for model, temps in ENSEMBLE_CONFIGS for temp in temps]
    return _ensemble(
        args,
        prompt,
        candidates,
        synthesis_model,
        is_multiturn=is_multiturn,
        log_name="Multimodel Ensemble",
        name_models=True,
    )


def query_single_model_ensemble(
    args: LaunchArgs | ModelQueryArgs,
    prompt: list[dict],
    model: str,
    is_multiturn: bool = False,
) -> dict[str, Any]:
    """Query ``model`` at each of ``SINGLE_MODEL_ENSEMBLE_TEMPERATURES``, then have it
    synthesize one answer."""
    candidates = [(model, temp) for temp in SINGLE_MODEL_ENSEMBLE_TEMPERATURES]
    return _ensemble(
        args,
        prompt,
        candidates,
        model,
        is_multiturn=is_multiturn,
        log_name="Single Model Ensemble",
        name_models=False,
    )
