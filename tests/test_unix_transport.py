"""AF_UNIX transport: the same EnvClient, over a socket file instead of a port.

This is the link the eval queue puts a submission on — its container has no
network stack, so the gateway is only reachable through a socket bind-mounted in.
The point of these tests is that nothing about the *protocol* changes: identical
results and identical error codes to the TCP path.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_http_transport import CAMERA, VALID_JOINTS, FakeFactory, make_tasks

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.env_backend import EnvBackendServer, RemoteEnvFactory, parse_backend_routes
from capx.gateway.server import GatewayServer
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    HttpTransport,
    InProcessTransport,
    UnixTransport,
)
from capx.protocol.framing import MAX_UNIX_PATH_BYTES, check_unix_path


def serve_unix(tmp_path, mode=EnforcementMode.DEV, n_trials=2) -> GatewayServer:
    gateway = Gateway(make_tasks(n_trials), FakeFactory(), mode)
    return GatewayServer(gateway, unix_path=str(tmp_path / "gw.sock")).start_background()


def test_unix_reset_step_get_obs(tmp_path) -> None:
    server = serve_unix(tmp_path)
    client = EnvClient(UnixTransport(server.host))
    try:
        reset = client.reset("t")
        assert isinstance(reset.obs, FullObs)
        assert reset.obs.cameras[CAMERA].rgb.shape == (4, 4, 3)

        result = client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))
        np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-4)
        assert CAMERA in client.get_observation(reset.episode_id).cameras
    finally:
        server.close()


def test_unix_matches_tcp_for_the_same_action(tmp_path) -> None:
    unix_server = serve_unix(tmp_path)
    tcp_server = GatewayServer(
        Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV)
    ).start_background()
    try:
        action = FrankaRawAction(joints=VALID_JOINTS, gripper=0.5)
        over_unix = EnvClient(UnixTransport(unix_server.host))
        over_tcp = EnvClient(HttpTransport(tcp_server.host, tcp_server.port))
        u = over_unix.step(over_unix.reset("t").episode_id, action)
        t = over_tcp.step(over_tcp.reset("t").episode_id, action)
        np.testing.assert_allclose(u.obs.joint_pos, t.obs.joint_pos, atol=1e-4)
    finally:
        unix_server.close()
        tcp_server.close()


def test_unix_propagates_gateway_error(tmp_path) -> None:
    server = serve_unix(tmp_path, EnforcementMode.OFFICIAL, n_trials=1)
    client = EnvClient(UnixTransport(server.host))
    try:
        client.reset("t")
        with pytest.raises(GatewayError) as exc:
            client.reset("t")
        assert exc.value.code is ErrorCode.TRIALS_EXHAUSTED
    finally:
        server.close()


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
    backend = EnvBackendServer(FakeFactory(), unix_path=str(sock)).start_background()
    try:
        routes = parse_backend_routes([f"robosuite=unix:{sock}"])
        assert routes["robosuite"][0].unix_path == str(sock)
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
