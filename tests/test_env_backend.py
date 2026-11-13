"""3-tier seam: Gateway -> RemoteEnvFactory -> EnvBackendServer over a socket.

Proves the gateway can drive a simulator hosted in a separate backend (here a fake
in a thread over a Unix socket) without importing the sim, keeping
referee / ObsFilter / scoring central. Reuses the same fake as the in-process
gateway tests, so equivalence to the in-process path is direct.
"""

from __future__ import annotations

import socket
import threading
import time

import numpy as np
import pytest

from capx.agent_client import run_eval
from capx.gateway import EnforcementMode, Gateway
from capx.gateway.env_backend import (
    BackendAddress,
    EnvBackendServer,
    RemoteEnvFactory,
    parse_backend_routes,
)
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    InProcessTransport,
    ProprioObs,
)
from capx.protocol.framing import connect_unix_socket, recv_framed, send_framed
from capx.protocol.taskpack import SeedPolicy, TaskSpec
from socket_paths import fresh_socket_path
from test_gateway import (
    CAMERA,
    VALID_JOINTS,
    FakeEnv,
    FakeFactory,
    FleetingSuccessFactory,
    hold_factory,
    make_tasks,
)


def backend(**env_kwargs) -> EnvBackendServer:
    return EnvBackendServer(FakeFactory(**env_kwargs), fresh_socket_path()).start_background()


def gateway_over(server: EnvBackendServer, mode=EnforcementMode.DEV, **kw):
    routes = {"robosuite": [server.address]}
    gw = Gateway(make_tasks(), RemoteEnvFactory(routes), mode, **kw)
    return EnvClient(InProcessTransport(gw.acquire())), gw


def act() -> FrankaRawAction:
    return FrankaRawAction(joints=VALID_JOINTS, gripper=1.0)


def test_remote_reset_step_over_the_socket() -> None:
    server = backend()
    try:
        client, _ = gateway_over(server)
        reset = client.reset("t")
        assert isinstance(reset.obs, FullObs)
        assert set(reset.obs.cameras) == {CAMERA}  # privileged cube_poses stripped centrally

        result = client.step(reset.episode_id, act())
        assert isinstance(result.obs, ProprioObs)
        np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-4)

        full = client.get_observation(reset.episode_id)
        assert CAMERA in full.cameras
    finally:
        server.close()


def test_remote_scores_success_internally() -> None:
    server = backend(solve_at_cycle=1)
    try:
        client, gw = gateway_over(server)
        reset = client.reset("t")
        result = client.step(reset.episode_id, act())
        assert gw.aggregate()["t"]["success_rate"] == 1.0
        assert result.terminated is False  # success never leaks over the seam
    finally:
        server.close()


def test_remote_success_inside_a_repeat_counts() -> None:
    server = EnvBackendServer(FleetingSuccessFactory(), fresh_socket_path()).start_background()
    try:
        client, gw = gateway_over(server)
        reset = client.reset("t")
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=3))
        assert gw.aggregate()["t"]["successes"] == 1
    finally:
        server.close()


class CountingBackend(EnvBackendServer):
    def __init__(self, factory) -> None:  # noqa: ANN001
        super().__init__(factory, fresh_socket_path())
        self.calls: list[str] = []

    def _dispatch(self, request, env):  # noqa: ANN001
        self.calls.append(request["method"])
        return super()._dispatch(request, env)


def test_a_step_costs_one_backend_call_per_cycle_plus_three() -> None:
    """A cycle reports its success in the same call that runs it; reward, proprio and
    sim_time follow once per step."""
    server = CountingBackend(FakeFactory()).start_background()
    try:
        client, _ = gateway_over(server)
        reset = client.reset("t")
        server.calls.clear()
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=5))
        assert server.calls.count("apply_raw_action") == 5
        assert len(server.calls) == 5 + 3
    finally:
        server.close()


def test_remote_env_crash_propagates_and_refunds() -> None:
    server = backend(crash_at_cycle=1)
    try:
        client, gw = gateway_over(server, mode=EnforcementMode.OFFICIAL)
        reset = client.reset("t")
        with pytest.raises(GatewayError) as exc:
            client.step(reset.episode_id, act())
        assert exc.value.code is ErrorCode.ENV_CRASH
        assert gw.trials_used("t") == 0  # trial refunded across the seam
    finally:
        server.close()


def test_remote_routing_rejects_unknown_simulator() -> None:
    server = backend()
    try:
        # route only "libero"; the task's simulator is "robosuite" -> no route -> crash
        gw = Gateway(
            make_tasks(),
            RemoteEnvFactory({"libero": [server.address]}),
            EnforcementMode.DEV,
        )
        client = EnvClient(InProcessTransport(gw.acquire()))
        with pytest.raises(GatewayError) as exc:
            client.reset("t")
        assert exc.value.code is ErrorCode.ENV_CRASH
    finally:
        server.close()


def _task(task_id: str, simulator: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        embodiment="franka",
        simulator=simulator,
        instruction="go",
        camera_names=[CAMERA],
        env_target="fake",
        n_trials=1,
        trial_steplimit=100,
        seed_policy=SeedPolicy(mode="fixed", seeds=[1]),
    )


def test_routes_multiple_simulators_in_one_run() -> None:
    """One gateway drives two simulators at once, routing each task by its simulator."""
    robo = backend()  # stands in for the robosuite backend
    lib = backend()  # stands in for the libero backend
    try:
        tasks = {"ra": _task("ra", "robosuite"), "lb": _task("lb", "libero")}
        routes = {"robosuite": [robo.address], "libero": [lib.address]}
        gw = Gateway(tasks, RemoteEnvFactory(routes), EnforcementMode.DEV)
        client = EnvClient(InProcessTransport(gw.acquire()))
        run_eval(client, hold_factory, max_trials_per_task=1, max_steps=2)
        assert set(gw.aggregate()) == {"ra", "lb"}  # both simulators driven in one run
    finally:
        robo.close()
        lib.close()


def test_a_backend_that_is_still_starting_is_waited_for() -> None:
    # A simulator backend binds only once its imports finish (~15 s for BEHAVIOR);
    # a submission that resets first must not get an env_crash for it.
    path = fresh_socket_path()
    servers: list[EnvBackendServer] = []

    def start_late() -> None:
        servers.append(EnvBackendServer(FakeFactory(), path).start_background())

    timer = threading.Timer(1.0, start_late)
    timer.start()
    try:
        routes = {"robosuite": [BackendAddress(path)]}
        gw = Gateway(make_tasks(), RemoteEnvFactory(routes), EnforcementMode.DEV)
        reset = EnvClient(InProcessTransport(gw.acquire())).reset("t")
        assert isinstance(reset.obs, FullObs)
    finally:
        timer.join()
        for server in servers:
            server.close()


def test_a_backend_that_never_listens_still_fails() -> None:
    routes = {"robosuite": [BackendAddress(fresh_socket_path())]}
    factory = RemoteEnvFactory(routes, start_timeout_s=0.5)
    with pytest.raises(FileNotFoundError):
        factory.build(make_tasks()["t"])


def test_parse_backend_routes() -> None:
    routes = parse_backend_routes(
        ["robosuite=unix:/run/a.sock", "robosuite=unix:/run/b.sock", "libero=unix:/run/c.sock"]
    )
    assert routes == {
        "robosuite": [BackendAddress("/run/a.sock"), BackendAddress("/run/b.sock")],
        "libero": [BackendAddress("/run/c.sock")],
    }
    with pytest.raises(ValueError, match="bad backend spec"):
        parse_backend_routes(["robosuite"])


@pytest.mark.parametrize("spec", ["robosuite=127.0.0.1:8100", "robosuite=env-robosuite:8100"])
def test_parse_backend_routes_rejects_tcp_addresses(spec: str) -> None:
    with pytest.raises(ValueError, match="only on Unix sockets"):
        parse_backend_routes([spec])


def test_a_live_env_holds_its_backend_against_the_next_build() -> None:
    """The simulator inside a backend is not safe to share between lanes."""
    b1, b2 = backend(), backend()
    try:
        factory = RemoteEnvFactory({"robosuite": [b1.address, b2.address]})
        first = factory.build(make_tasks()["t"])
        second = factory.build(make_tasks()["t"])
        built: list[object] = []
        waiter = threading.Thread(target=lambda: built.append(factory.build(make_tasks()["t"])))
        waiter.start()
        waiter.join(timeout=0.3)
        assert waiter.is_alive()  # both backends leased: the third build waits

        first.close()
        waiter.join(timeout=5)
        assert len(built) == 1  # the freed backend came back
        built[0].close()
        second.close()
    finally:
        b1.close()
        b2.close()


def test_an_on_demand_backend_is_asked_for_before_the_first_connection(tmp_path) -> None:
    """The queue starts a backend only once the gateway needs it."""
    sock = tmp_path / "robosuite-0.sock"
    started: list[EnvBackendServer] = []

    def start_when_asked() -> None:
        want = tmp_path / "robosuite-0.sock.want"
        while not want.exists():
            time.sleep(0.01)
        started.append(EnvBackendServer(FakeFactory(), str(sock)).start_background())

    starter = threading.Thread(target=start_when_asked)
    starter.start()
    try:
        factory = RemoteEnvFactory(
            {"robosuite": [BackendAddress(str(sock))]},
            on_demand=True,
            start_timeout_s=10,
        )
        env = factory.build(make_tasks()["t"])
        env.close()
    finally:
        starter.join(timeout=10)
        for server in started:
            server.close()
    assert started


class CrashFirstResetFactory(FakeFactory):
    """The first env built crashes on reset (a backend dying mid-load); later ones work."""

    def build(self, spec) -> FakeEnv:  # noqa: ANN001
        env = super().build(spec)
        env._crash_on_reset = len(self.built) == 1
        return env


def _reset_within(client: EnvClient, seconds: float) -> None:
    done = threading.Event()
    thread = threading.Thread(target=lambda: (client.reset("t"), done.set()), daemon=True)
    thread.start()
    assert done.wait(seconds), f"reset did not return within {seconds} s"


def test_a_crash_in_the_first_reset_gives_the_backend_back() -> None:
    """Before the fix the new env was never closed, its lease never came back, and the
    lane's next reset on that simulator waited forever."""
    server = EnvBackendServer(CrashFirstResetFactory(), fresh_socket_path()).start_background()
    try:
        factory = RemoteEnvFactory({"robosuite": [server.address]})
        gw = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL)
        client = EnvClient(InProcessTransport(gw.acquire()))
        with pytest.raises(GatewayError) as exc:
            client.reset("t")
        assert exc.value.code is ErrorCode.ENV_CRASH
        assert factory._leased == set()

        _reset_within(client, 5)
        assert gw.trials_used("t") == 1
    finally:
        server.close()


def test_a_reset_waits_for_a_backend_only_so_long() -> None:
    """Every backend of the simulator is leased: the reset waits, then fails as an
    env_crash that refunds the trial instead of hanging the lane."""
    server = backend()
    try:
        factory = RemoteEnvFactory({"robosuite": [server.address]}, lease_timeout_s=0.2)
        gw = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL, max_sessions=2)
        holder, waiter = (EnvClient(InProcessTransport(gw.acquire())) for _ in range(2))
        holder.reset("t")

        with pytest.raises(GatewayError) as exc:
            waiter.reset("t")

        assert exc.value.code is ErrorCode.ENV_CRASH
        assert gw.trials_used("t") == 1  # only the holder's
    finally:
        server.close()


class HangFirstResetFactory(FakeFactory):
    """The first env built does not finish its reset in time (a hung simulator).
    ``shared`` records, per build, whether an earlier env was still open."""

    def __init__(self, release: threading.Event) -> None:
        super().__init__()
        self._release = release
        self.shared: list[bool] = []

    def build(self, spec) -> FakeEnv:  # noqa: ANN001
        self.shared.append(any(not env.closed for env in self.built))
        env = super().build(spec)
        if len(self.built) == 1:
            reset = env.reset

            def hang(*, seed: int) -> None:
                self._release.wait(10)
                reset(seed=seed)

            env.reset = hang  # type: ignore[method-assign]
        return env


def _wait_until(condition, seconds: float = 5.0) -> None:  # noqa: ANN001
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.01)


def test_a_backend_that_stops_answering_is_held_until_its_call_ends() -> None:
    """The gateway gives up on the call, but the backend is still running it: no env
    may be built beside it (BEHAVIOR keeps one env per process) until it is done."""
    release = threading.Event()
    sim = HangFirstResetFactory(release)
    server = EnvBackendServer(sim, fresh_socket_path()).start_background()
    try:
        factory = RemoteEnvFactory(
            {"robosuite": [server.address]}, call_timeout_s=0.3, lease_timeout_s=0.3
        )
        gw = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL)
        client = EnvClient(InProcessTransport(gw.acquire()))
        with pytest.raises(GatewayError) as exc:
            client.reset("t")
        assert exc.value.code is ErrorCode.ENV_CRASH

        with pytest.raises(GatewayError) as retry:
            client.reset("t")  # the backend is still inside the first reset
        assert retry.value.code is ErrorCode.ENV_CRASH
        assert len(sim.built) == 1

        release.set()
        _wait_until(lambda: not factory._leased)
        _reset_within(client, 5)
        assert sim.shared == [False, False]
        assert gw.trials_used("t") == 1
    finally:
        release.set()
        server.close()


def test_a_backend_that_is_gone_goes_back_to_the_pool_at_once() -> None:
    """A backend process that exited is restarted by the runner when next asked for,
    so there is nothing to wait for."""
    address = BackendAddress(fresh_socket_path())
    factory = RemoteEnvFactory({"robosuite": [address]})
    factory._take("robosuite", [address])

    factory._give_back("robosuite", address, broken=True)

    _wait_until(lambda: not factory._leased)


def test_a_backend_serves_one_connection_at_a_time() -> None:
    server = backend()
    try:
        first = connect_unix_socket(server.address.path)
        send_framed(first, {"method": "build", "payload": {"spec": make_tasks()["t"].model_dump(mode="json")}})
        assert recv_framed(first)["ok"] is True

        second = connect_unix_socket(server.address.path)
        second.settimeout(0.3)
        send_framed(second, {"method": "ping", "payload": {}})
        with pytest.raises(TimeoutError):
            recv_framed(second)  # waits while the first connection holds its env

        first.close()
        second.settimeout(5)
        assert recv_framed(second)["ok"] is True
        second.close()
    finally:
        server.close()
