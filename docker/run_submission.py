"""Example submission entrypoint: run the baseline policies against the gateway.

A real submission swaps the baseline policies for its own stack. Scores are never
returned to a submission; the gateway's results dump has them.

Env vars: ``CAPX_ENV_SOCKET`` (AF_UNIX path to the gateway — set by the eval queue,
whose submission containers have no network stack) or ``CAPX_ENV_HOST`` /
``CAPX_ENV_PORT`` (the environment container over TCP). Trial and step counts
come from the manifest.
"""

from __future__ import annotations

import os

from capx.agent_client import baseline_policy_factory, run_eval
from capx.config import load_eval_config
from capx.protocol import EnvClient, HttpTransport, UnixTransport


def main() -> None:
    socket_path = os.environ.get("CAPX_ENV_SOCKET", "").strip()

    if socket_path:
        client = EnvClient(UnixTransport(socket_path))
    else:
        host = os.environ.get("CAPX_ENV_HOST", "gateway")
        port = int(os.environ.get("CAPX_ENV_PORT", "8000"))
        client = EnvClient(HttpTransport(host, port))
    embodiments = {entry["embodiment"] for entry in client.manifest()}
    policy_factory = baseline_policy_factory(
        embodiments, r1pro_action_dim=load_eval_config().embodiments.r1pro.action_dim
    )
    run_eval(client, policy_factory)


if __name__ == "__main__":
    main()
