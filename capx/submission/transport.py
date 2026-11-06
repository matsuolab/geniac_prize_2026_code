"""Connect the submission to the gateway."""

from __future__ import annotations

import os

from capx.protocol import EnvClient, HttpTransport, UnixTransport
from capx.submission.interfaces import SubmissionConfig


def build_env_client(cfg: SubmissionConfig) -> EnvClient:
    """``CAPX_ENV_SOCKET`` (the eval queue's AF_UNIX socket) wins; otherwise TCP, with
    ``CAPX_ENV_HOST``/``CAPX_ENV_PORT`` over the config's host and port."""
    socket_path = os.environ.get("CAPX_ENV_SOCKET", "").strip()
    if socket_path:
        return EnvClient(UnixTransport(socket_path))
    host = os.environ.get("CAPX_ENV_HOST", cfg.gateway_host)
    port = int(os.environ.get("CAPX_ENV_PORT", str(cfg.gateway_port)))
    return EnvClient(HttpTransport(host, port))


def build_env_clients(cfg: SubmissionConfig, lanes: int) -> list[EnvClient]:
    """One connection per lane: that is how the gateway hands out lanes."""
    return [build_env_client(cfg) for _ in range(lanes)]


__all__ = ["build_env_client", "build_env_clients"]
