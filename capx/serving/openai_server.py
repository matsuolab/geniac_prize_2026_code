"""OpenAI-compatible proxy server for CaP-X, in front of OpenAI's API.

The proxy exposes `/chat/completions` on the port (8110) that CaP-X YAML configs
default to, so task configs work unmodified — only the `--model` flag differs.
`openrouter_server.py` is the same proxy in front of OpenRouter.

It listens on 127.0.0.1 unless `--host` says otherwise: anyone who can reach it
spends the operator's key.

API key resolution order:
  1. `--api-key` CLI flag (discouraged — ends up in shell history)
  2. `OPENAI_API_KEY` environment variable (preferred)
  3. `--key-file` contents (one key per line; defaults to ``.openaikey``)

Example::

    export OPENAI_API_KEY=sk-...
    python capx/serving/openai_server.py --port 8110

    # then, in another shell:
    python capx/envs/launch.py \\
        --config-path env_configs/libero/franka_libero_spatial_0_smoke.yaml \\
        --model gpt-4o-mini \\
        --server-url http://127.0.0.1:8110/chat/completions
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import openai
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ImageUrl(BaseModel):
    url: str


class ContentItem(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentItem] | None = None


class ChatCompletionRequest(BaseModel):
    """Unset fields stay unset upstream, so the provider's own defaults apply."""

    model: str
    messages: list[Message]
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    top_p: float | None = None
    reasoning_effort: str | None = None
    max_completion_tokens: int | None = None


# Reasoning-only models (o-series, gpt-5) reject `temperature` / `max_tokens`
# in favour of `max_completion_tokens`. We rewrite the request on the fly so
# upstream CaP-X code can stay oblivious.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _looks_like_reasoning_model(model: str) -> bool:
    m = model.lower()
    return any(m.startswith(p) or m.startswith(f"openai/{p}") for p in _REASONING_PREFIXES)


def _normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Strip an optional ``openai/`` prefix and adapt params for reasoning models."""
    model = kwargs.get("model", "")
    if model.startswith("openai/"):
        model = model[len("openai/"):]
        kwargs["model"] = model

    if _looks_like_reasoning_model(model):
        # Reasoning models: temperature must be default, max_tokens is renamed.
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        else:
            kwargs.pop("max_tokens", None)
    else:
        # Chat models don't understand reasoning_effort.
        kwargs.pop("reasoning_effort", None)

    return kwargs


def load_api_keys(key_file: str) -> list[str]:
    """One key per line; blank lines and ``#`` comments are skipped."""
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Key file not found: {key_file}")
    keys = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    if not keys:
        raise ValueError(f"No API keys found in {key_file}")
    return keys


def create_app(
    api_key: str,
    base_url: str,
    *,
    normalize: Callable[[dict[str, Any]], dict[str, Any]] = _normalize_kwargs,
    default_headers: dict[str, str] | None = None,
    title: str = "OpenAI Proxy (CaP-X)",
) -> FastAPI:
    """The proxy app. ``normalize`` rewrites each request's arguments for the provider.

    Upstream errors keep their status code, so a client can tell a request to retry
    (429, 503) from one that will only fail again (400, 401, 404). The upstream reply
    is passed through whole, usage and reasoning included.
    """
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=default_headers)

    app = FastAPI(title=title, version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/completions")
    async def chat_completions(request: ChatCompletionRequest) -> Any:
        kwargs = normalize(request.model_dump(exclude_none=True))
        try:
            response = await client.chat.completions.create(**kwargs)
        except openai.APIStatusError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except openai.APITimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if request.stream:

            async def event_stream():
                async for chunk in response:
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")
        return response.model_dump(exclude_none=True)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app


def main(
    key_file: str = ".openaikey",
    api_key: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8110,
    base_url: str = "https://api.openai.com/v1/",
):
    """Start the OpenAI proxy server.

    Resolution order for the API key: ``--api-key`` > ``$OPENAI_API_KEY`` > ``--key-file``.
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            logger.info("Loaded API key from $OPENAI_API_KEY")
    if api_key is None:
        api_key = load_api_keys(key_file)[0]
        logger.info(f"Loaded API key from {key_file}")

    uvicorn.run(create_app(api_key=api_key, base_url=base_url), host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
