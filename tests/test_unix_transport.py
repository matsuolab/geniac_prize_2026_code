"""AF_UNIX socket handling: the only way the gateway and the env backends listen.

On a shared node any port is reachable from other users' jobs, so the sockets must
be reachable only by their owner.
"""

from __future__ import annotations

import os
import stat

import numpy as np
import pytest
from test_socket_transport import CAMERA, VALID_JOINTS, FakeFactory, make_tasks

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.env_backend import EnvBackendServer, RemoteEnvFactory, parse_backend_routes
from capx.gateway.server import GatewayServer
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    InProcessTransport,
)
from capx.protocol.framing import MAX_UNIX_PATH_BYTES, bind_unix_socket, check_unix_path


def serve_unix(tmp_path, mode=EnforcementMode.DEV, n_trials=2) -> GatewayServer:
    gateway = Gateway(make_tasks(n_trials), FakeFactory(), mode)
    return GatewayServer(gateway, str(tmp_path / "gw.sock")).start_background()


def test_socket_file_is_removed_on_close(tmp_path) -> None:
    """A leftover socket would make the next run's bind fail with EADDRINUSE."""
    server = serve_unix(tmp_path)
    path = tmp_path / "gw.sock"
    assert path.exists()
    server.close()
    assert not path.exists()


def test_over_long_socket_path_fails_with_a_readable_error(tmp_path) -> None:
    too_long = str(tmp_path / ("x" * MAX_UNIX_PATH_BYTES))
    with pytest.raises(ValueError, match="over the .*-byte kernel limit"):
        check_unix_path(too_long)


def test_env_backend_over_a_unix_socket(tmp_path) -> None:
    """The gateway<->backend link too: it carries raw obs, so it must not need a port."""
    sock = tmp_path / "robosuite.sock"
    backend = EnvBackendServer(FakeFactory(), str(sock)).start_background()
    try:
        routes = parse_backend_routes([f"robosuite=unix:{sock}"])
        assert routes["robosuite"][0].path == str(sock)
        gateway = Gateway(make_tasks(), RemoteEnvFactory(routes), EnforcementMode.DEV)
        with gateway:
            client = EnvClient(InProcessTransport(gateway.acquire()))
            reset = client.reset("t")
            result = client.step(
                reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0)
            )
            np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-4)
    finally:
        backend.close()
