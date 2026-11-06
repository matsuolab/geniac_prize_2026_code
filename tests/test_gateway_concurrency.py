"""Lanes: N episodes at once behind one referee.

Simulation is ~0.2% of an episode's wall clock (tests/test_gateway_timing.py is
where that split is measured), so the only way a thousand trials fit in a day is
running many of them at the same time. What must survive that is the accounting:
trial caps, the job budget and seed allocation are properties of the *team*, not of
a lane, and a wide run has to charge exactly what a serial one would.
"""

from __future__ import annotations

import contextlib
import threading
import socket
import struct
import time

import msgpack
import numpy as np
import pytest
from test_http_transport import VALID_JOINTS, FakeEnv, FakeFactory, make_tasks

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.server import GatewayServer
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    GatewayError,
    HttpTransport,
    InProcessTransport,
)
from capx.protocol.taskpack import SeedPolicy


class IsolatingFactory(FakeFactory):
    """Hands out an independent env per build — a stand-in for a backend pool."""

    supports_parallel = True

    def __init__(self) -> None:
        self.built: list[FakeEnv] = []

    def build(self, spec) -> FakeEnv:  # noqa: ANN001
        env = FakeEnv()
        env.closed = False
        env.seeds = []
        original_close, original_reset = env.close, env.reset

        def close() -> None:
            env.closed = True
            original_close()

        def reset(*, seed: int) -> None:
            env.seeds.append(seed)
            original_reset(seed=seed)

        env.close = close  # type: ignore[method-assign]
        env.reset = reset  # type: ignore[method-assign]
        self.built.append(env)
        return env


def gateway_with(lanes: int, *, mode=EnforcementMode.DEV, n_trials: int = 2) -> Gateway:
    return Gateway(make_tasks(n_trials), IsolatingFactory(), mode, max_sessions=lanes)


def step(lane, episode_id: str):  # noqa: ANN001
    return EnvClient(InProcessTransport(lane)).step(
        episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0)
    )


def test_two_lanes_hold_two_episodes_at_once() -> None:
    gateway = gateway_with(2)
    a, b = gateway.acquire(), gateway.acquire()

    first, second = a.reset("t"), b.reset("t")

    assert first.episode_id != second.episode_id
    assert sorted(gateway.live_episodes()) == sorted([first.episode_id, second.episode_id])
    # Both are steppable — neither reset superseded the other.
    step(a, first.episode_id)
    step(b, second.episode_id)


def test_a_lane_supersedes_only_its_own_episode() -> None:
    """The single-episode gateway's rule, now scoped to a lane."""
    gateway = gateway_with(2)
    a, b = gateway.acquire(), gateway.acquire()
    kept = b.reset("t")
    superseded = a.reset("t")

    a.reset("t")

    with pytest.raises(GatewayError) as gone:
        step(a, superseded.episode_id)
    assert gone.value.code is ErrorCode.CLIENT_TIMEOUT
    step(b, kept.episode_id)  # the other lane is untouched


def test_lanes_charge_trials_to_one_ledger() -> None:
    """Two lanes must not each restart the trial sequence from zero — otherwise a
    parallel run draws different seeds, and the trial cap means nothing."""
    gateway = gateway_with(2, mode=EnforcementMode.OFFICIAL, n_trials=2)
    a, b = gateway.acquire(), gateway.acquire()

    a.reset("t")
    b.reset("t")

    assert gateway.trials_used("t") == 2
    with pytest.raises(GatewayError) as exhausted:
        a.reset("t")
    assert exhausted.value.code is ErrorCode.TRIALS_EXHAUSTED


def test_lanes_draw_the_seeds_a_serial_run_would() -> None:
    tasks = make_tasks(n_trials=2)  # seed_policy fixed on [1, 2]
    gateway = gateway_with(2, mode=EnforcementMode.OFFICIAL)
    a, b = gateway.acquire(), gateway.acquire()
    a.reset("t")
    b.reset("t")

    factory = gateway._shared.factory
    assert sorted(env.seeds[0] for env in factory.built) == list(tasks["t"].seed_policy.seeds)


def test_a_voided_trial_s_retry_replays_its_own_seed() -> None:
    """Lane A runs trial 0 while lane B runs trial 1, and A's simulator crashes. A's
    retry must replay trial 0's instance, not take B's: otherwise one instance is
    never scored and another is scored twice."""
    tasks = make_tasks(n_trials=3)
    tasks["t"] = tasks["t"].model_copy(
        update={"seed_policy": SeedPolicy(mode="fixed", seeds=[100, 200, 300])}
    )
    factory = IsolatingFactory()
    gateway = Gateway(tasks, factory, EnforcementMode.OFFICIAL, max_sessions=2)
    a, b = gateway.acquire(), gateway.acquire()
    doomed = a.reset("t")
    b.reset("t")

    def crash(action):  # noqa: ANN001
        raise RuntimeError("sim died")

    factory.built[0].apply_raw_action = crash  # type: ignore[method-assign]
    with pytest.raises(GatewayError) as exc:
        step(a, doomed.episode_id)
    assert exc.value.code is ErrorCode.ENV_CRASH

    a.reset("t")
    b.reset("t")

    assert factory.built[0].seeds == [100]  # voided
    assert factory.built[1].seeds == [200, 300]  # lane B
    assert factory.built[2].seeds == [100]  # lane A's retry, on a rebuilt env
    assert gateway.trials_used("t") == 3


def test_each_lane_gets_its_own_env() -> None:
    gateway = gateway_with(2)
    a, b = gateway.acquire(), gateway.acquire()
    a.reset("t")
    b.reset("t")

    built = gateway._shared.factory.built
    assert len(built) == 2
    assert built[0] is not built[1]


def test_closing_one_lane_leaves_the_others_running() -> None:
    gateway = gateway_with(2)
    a, b = gateway.acquire(), gateway.acquire()
    kept = b.reset("t")
    doomed = a.reset("t")

    a.close()

    with pytest.raises(GatewayError):
        step(a, doomed.episode_id)
    step(b, kept.episode_id)


def test_a_gateway_hands_out_exactly_max_sessions_lanes() -> None:
    gateway = gateway_with(2)
    gateway.acquire()
    gateway.acquire()
    with pytest.raises(ValueError, match="2 concurrent lane"):
        gateway.acquire()


def test_releasing_a_lane_hands_it_back() -> None:
    """A submission that disconnects must not cost the run a lane for good."""
    gateway = gateway_with(2)
    a, b = gateway.acquire(), gateway.acquire()
    reset = a.reset("t")
    gateway.release(a)

    assert gateway._shared.factory.built[0].closed is True  # the backend went back too
    again = gateway.acquire()
    assert again.reset("t").episode_id != reset.episode_id
    assert b is not again


def test_concurrency_is_refused_on_a_factory_that_shares_one_sim() -> None:
    """RobosuiteEnvFactory caches one MuJoCo sim; N lanes would race on it."""
    with pytest.raises(ValueError, match="not parallel-safe"):
        Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV, max_sessions=4)


def test_two_connections_are_two_lanes() -> None:
    """The wire is unchanged: a submission gets concurrency by opening connections."""
    gateway = gateway_with(2)
    server = GatewayServer(gateway).start_background()
    clients = [EnvClient(HttpTransport(server.host, server.port)) for _ in range(2)]
    try:
        first, second = (client.reset("t") for client in clients)
        assert first.episode_id != second.episode_id
        assert len(gateway.live_episodes()) == 2
        for client, reset in zip(clients, (first, second)):
            client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))
    finally:
        for client in clients:
            client.close()
        server.close()


def test_a_connection_past_the_cap_is_told_why() -> None:
    """An unexplained EOF would look like a crashed gateway to the submission."""
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    held = EnvClient(HttpTransport(server.host, server.port))
    try:
        held.reset("t")
        extra = EnvClient(HttpTransport(server.host, server.port))
        with pytest.raises(RuntimeError, match="concurrent lane"):
            extra.reset("t")
    finally:
        held.close()
        server.close()


def test_a_disconnect_frees_the_lane() -> None:
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    try:
        first = EnvClient(HttpTransport(server.host, server.port))
        first.reset("t")
        first._transport.close()  # the submission container died mid-run

        second = EnvClient(HttpTransport(server.host, server.port))
        for _ in range(50):  # the server frees the lane on its own thread
            try:
                second.reset("t")
                break
            except RuntimeError:
                time.sleep(0.02)
                second._transport.close()
        else:
            raise AssertionError("the lane was never handed back")
    finally:
        server.close()


def test_closing_a_client_hands_its_lane_to_the_next_connection() -> None:
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    try:
        first = EnvClient(HttpTransport(server.host, server.port))
        first.reset("t")
        first.close()

        second = EnvClient(HttpTransport(server.host, server.port))
        second.reset("t")
        second.close()
        first.reset("t")  # a closed client reconnects as a new lane
        first.close()
    finally:
        server.close()


def test_closing_a_client_over_an_already_connected_transport_frees_the_lane() -> None:
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    try:
        transport = HttpTransport(server.host, server.port)
        transport.call("manifest", {})
        EnvClient(transport).close()
        assert not transport.connected

        other = EnvClient(HttpTransport(server.host, server.port))
        other.reset("t")
        other.close()
    finally:
        server.close()


def test_a_transport_reconnects_after_a_close_call() -> None:
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    transport = HttpTransport(server.host, server.port)
    try:
        transport.call("close", {})
        assert not transport.connected
        assert transport.call("manifest", {})["tasks"]
    finally:
        transport.close()
        server.close()


def test_a_client_reconnects_right_after_close() -> None:
    gateway = gateway_with(1)
    server = GatewayServer(gateway, max_connections=1).start_background()
    client = EnvClient(HttpTransport(server.host, server.port))
    try:
        for _ in range(20):
            client.reset("t")
            client.close()
    finally:
        server.close()


def test_a_call_only_transport_still_closes_its_lane() -> None:
    class CallOnly:
        def __init__(self, inner: InProcessTransport) -> None:
            self.calls: list[str] = []
            self._inner = inner

        def call(self, method, payload):
            self.calls.append(method)
            return self._inner.call(method, payload)

    transport = CallOnly(InProcessTransport(gateway_with(1).acquire()))
    client = EnvClient(transport)
    client.reset("t")
    client.close()
    assert transport.calls == ["reset", "close"]


def test_a_closing_peer_that_stops_reading_releases_its_slot(monkeypatch) -> None:
    """The reply to close is the one send made after the lane is handed back; a peer
    that never reads it must not keep its connection slot."""
    import capx.gateway.server as server_module

    real_send = server_module.send_framed
    unblock = threading.Event()
    stalled_once = threading.Event()

    def send_to_a_peer_that_stopped_reading(conn, obj):
        if obj == {"ok": True, "result": {}} and not stalled_once.is_set():  # close's reply
            stalled_once.set()
            timeout = conn.gettimeout()
            unblock.wait(timeout)
            if timeout is not None:
                raise TimeoutError("send timed out")
        real_send(conn, obj)

    monkeypatch.setattr(server_module, "send_framed", send_to_a_peer_that_stopped_reading)
    monkeypatch.setattr(server_module, "REFUSAL_TIMEOUT_S", 0.2)
    gateway = gateway_with(1)
    server = GatewayServer(gateway, max_connections=1).start_background()
    stalled = HttpTransport(server.host, server.port)
    try:
        stalled.call("manifest", {})
        def close_and_never_read() -> None:
            with contextlib.suppress(Exception):
                stalled.call("close", {})

        threading.Thread(target=close_and_never_read, daemon=True).start()
        time.sleep(0.5)

        other = EnvClient(HttpTransport(server.host, server.port))
        try:
            other.reset("t")
        except (ConnectionError, OSError, RuntimeError) as exc:
            raise AssertionError("the stalled close kept its connection slot") from exc
        other.close()
    finally:
        unblock.set()
        stalled.close()
        server.close()


def test_connections_past_the_connection_cap_are_dropped() -> None:
    """A flood of idle connections cannot hold a thread each."""
    gateway = gateway_with(1)
    server = GatewayServer(gateway, max_connections=2).start_background()
    held = EnvClient(HttpTransport(server.host, server.port))
    held.reset("t")
    idle = socket.create_connection((server.host, server.port))
    flood = socket.create_connection((server.host, server.port))
    try:
        flood.settimeout(5)
        assert flood.recv(1) == b""  # closed by the server, not left hanging
    finally:
        held.close()
        idle.close()
        flood.close()
        server.close()


def test_a_refused_connection_that_never_asks_is_dropped(monkeypatch) -> None:
    monkeypatch.setattr("capx.gateway.server.REFUSAL_TIMEOUT_S", 0.2)
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    held = EnvClient(HttpTransport(server.host, server.port))
    held.reset("t")
    silent = socket.create_connection((server.host, server.port))
    try:
        silent.settimeout(5)
        assert silent.recv(1) == b""
    finally:
        held.close()
        silent.close()
        server.close()


def test_a_refused_connection_cannot_trickle_its_request(monkeypatch) -> None:
    """Each recv gets a byte well within the timeout; only a deadline over the whole
    frame ends this."""
    monkeypatch.setattr("capx.gateway.server.REFUSAL_TIMEOUT_S", 0.5)
    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    held = EnvClient(HttpTransport(server.host, server.port))
    held.reset("t")
    slow = socket.create_connection((server.host, server.port))
    try:
        started = time.monotonic()
        slow.sendall(struct.pack("!I", 100))
        slow.settimeout(0.1)
        closed = False
        while not closed and time.monotonic() - started < 5:
            try:
                slow.sendall(b"\x00")
                closed = slow.recv(1) == b""
            except TimeoutError:
                continue
            except OSError:
                closed = True
        assert closed and time.monotonic() - started < 3
    finally:
        held.close()
        slow.close()
        server.close()


def test_a_non_string_method_is_answered_and_the_lane_stays_usable() -> None:
    from capx.protocol.framing import packb, recv_framed

    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    sock = socket.create_connection((server.host, server.port))
    try:
        for method in (np.zeros(2), ["close"], 3):
            payload = packb({"method": method, "payload": {}})
            sock.sendall(struct.pack("!I", len(payload)) + payload)
            assert recv_framed(sock)["ok"] is False
        payload = packb({"method": "manifest", "payload": {}})
        sock.sendall(struct.pack("!I", len(payload)) + payload)
        assert recv_framed(sock)["ok"] is True
    finally:
        sock.close()
        server.close()


def test_a_malformed_frame_is_answered_and_the_lane_stays_usable() -> None:
    from capx.protocol.framing import recv_framed

    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    sock = socket.create_connection((server.host, server.port))
    try:
        sock.sendall(struct.pack("!I", 3) + b"\xc1\xc1\xc1")
        reply = recv_framed(sock)
        assert reply["ok"] is False and reply["error"] is None

        payload = msgpack.packb({"method": "manifest", "payload": {}}, use_bin_type=True)
        sock.sendall(struct.pack("!I", len(payload)) + payload)
        assert recv_framed(sock)["ok"] is True
    finally:
        sock.close()
        server.close()


def test_an_oversized_request_closes_the_connection() -> None:
    from capx.gateway.server import MAX_REQUEST_BYTES

    gateway = gateway_with(1)
    server = GatewayServer(gateway).start_background()
    sock = socket.create_connection((server.host, server.port))
    try:
        sock.sendall(struct.pack("!I", MAX_REQUEST_BYTES + 1))
        sock.settimeout(5)
        assert sock.recv(1) == b""
    finally:
        sock.close()
        server.close()
