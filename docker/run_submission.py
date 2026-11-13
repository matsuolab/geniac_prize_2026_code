"""Example submission entrypoint: run the baseline policies against the gateway.

A real submission swaps the baseline policies for its own stack. Scores are never
returned to a submission; the gateway's results dump has them.

Env vars: ``CAPX_ENV_SOCKET`` (AF_UNIX path to the gateway, default
``/capx/run/gateway.sock``). Trial and step counts come from the manifest.
"""

from __future__ import annotations

import os

from capx.agent_client import baseline_policy_factory, run_eval
from capx.config import load_eval_config
from capx.protocol import EnvClient, UnixTransport


def main() -> None:
    socket_path = os.environ.get("CAPX_ENV_SOCKET", "").strip() or "/capx/run/gateway.sock"
    client = EnvClient(UnixTransport(socket_path))
    embodiments = {entry["embodiment"] for entry in client.manifest()}
    policy_factory = baseline_policy_factory(
        embodiments, r1pro_action_dim=load_eval_config().embodiments.r1pro.action_dim
    )
    run_eval(client, policy_factory)


if __name__ == "__main__":
    main()
