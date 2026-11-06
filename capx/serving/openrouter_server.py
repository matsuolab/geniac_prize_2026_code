"""OpenAI-compatible proxy server for CaP-X, in front of OpenRouter.

``openai_server.py``'s proxy with OpenRouter's base URL and headers. A model named
``openrouter/<provider>/<model>`` is sent as ``<provider>/<model>``. It listens on
127.0.0.1:8110, the URL CaP-X configs default to, unless told otherwise.

    python capx/serving/openrouter_server.py --key-file .openrouterkey
"""

from __future__ import annotations

import logging
from typing import Any

import tyro
import uvicorn
from fastapi import FastAPI

from capx.serving import openai_server

logger = logging.getLogger(__name__)

_BASE_URL = "https://openrouter.ai/api/v1/"

_DEFAULT_HEADERS = {
    "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
    "X-Title": "CaP-X",
}


def _strip_prefix(kwargs: dict[str, Any]) -> dict[str, Any]:
    model = kwargs.get("model", "")
    if model.startswith("openrouter/"):
        kwargs["model"] = model[len("openrouter/"):]
    return kwargs


def create_app(api_key: str, base_url: str = _BASE_URL) -> FastAPI:
    """The proxy app, sending OpenRouter's model names and headers."""
    return openai_server.create_app(
        api_key,
        base_url,
        normalize=_strip_prefix,
        default_headers=_DEFAULT_HEADERS,
        title="OpenRouter Proxy (CaP-X)",
    )


def main(
    key_file: str = ".openrouterkey",
    api_key: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8110,
    base_url: str = _BASE_URL,
):
    """Start the OpenRouter proxy server; the key is ``--api-key``, else the key file's first."""
    if api_key is None:
        api_key = openai_server.load_api_keys(key_file)[0]
        logger.info(f"Loaded API key from {key_file}")

    uvicorn.run(create_app(api_key, base_url), host=host, port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(main)
