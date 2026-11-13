"""Connect the submission to the gateway."""

from __future__ import annotations

import os

from capx.protocol import EnvClient, UnixTransport
from capx.submission.interfaces import SubmissionConfig


def build_env_client(cfg: SubmissionConfig) -> EnvClient:
    """``CAPX_ENV_SOCKET`` (set by the eval queue) wins over the config's socket."""
    socket_path = os.environ.get("CAPX_ENV_SOCKET", "").strip() or cfg.gateway_socket
    return EnvClient(UnixTransport(socket_path))


def build_env_clients(cfg: SubmissionConfig, lanes: int) -> list[EnvClient]:
    """One connection per lane: that is how the gateway hands out lanes."""
    return [build_env_client(cfg) for _ in range(lanes)]


__all__ = ["build_env_client", "build_env_clients"]
