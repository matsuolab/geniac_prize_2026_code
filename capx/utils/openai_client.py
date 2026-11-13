from __future__ import annotations

import importlib.util
from urllib.parse import unquote, urlsplit

from openai import DefaultHttpxClient, OpenAI

from capx.utils.unix_http import SCHEME, check_server_url, resolve_socket


def _unix_socket_client(socket_path: str) -> DefaultHttpxClient:
    # openai 2.x (uv.lock) is built on httpx and later releases on httpx2; the transport
    # has to come from the one the installed SDK uses, so whichever branch does not
    # match the installed SDK fails type checking. trust_env=False because an
    # HTTP(S)_PROXY from the environment would replace the transport.
    if importlib.util.find_spec("httpx2") is not None:
        import httpx2  # pyright: ignore[reportMissingImports]

        if issubclass(DefaultHttpxClient, httpx2.Client):
            return DefaultHttpxClient(
                transport=httpx2.HTTPTransport(uds=socket_path),  # pyright: ignore[reportArgumentType]
                trust_env=False,
            )
    import httpx

    return DefaultHttpxClient(
        transport=httpx.HTTPTransport(uds=socket_path),  # pyright: ignore[reportArgumentType]
        trust_env=False,
    )


def openai_client(base_url: str, api_key: str) -> OpenAI:
    """An OpenAI SDK client for ``base_url``: a local proxy's ``http+unix://`` URL, or a
    remote API's ``https://`` one."""
    parts = urlsplit(check_server_url(base_url))
    if parts.scheme != SCHEME:
        return OpenAI(base_url=base_url, api_key=api_key)
    socket_path = str(resolve_socket(unquote(parts.netloc)))
    return OpenAI(
        base_url=f"http://localhost{parts.path or '/'}",
        api_key=api_key,
        http_client=_unix_socket_client(socket_path),
    )
