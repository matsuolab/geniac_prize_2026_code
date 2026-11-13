"""Run a simulator in its own process behind the gateway.

The gateway never imports a sim: it drives a :class:`RemoteLowLevelEnv` proxy that
RPCs to an :class:`EnvBackendServer` hosting the real sim in its own venv. That is
what lets Robosuite, LIBERO and BEHAVIOR (mutually incompatible vendored stacks) run
in one evaluation, and :class:`RemoteEnvFactory` routes each task by
``TaskSpec.simulator``.

The backend link is internal: it carries raw sim obs (privileged fields included)
and reward/success, so it must never be reachable by a submission. It is the
``capx.protocol.framing`` wire over an AF_UNIX socket (``unix:/path`` in a route),
kept in a run-private directory.

One connection is one env lifecycle: ``build`` (carrying the ``TaskSpec``), then
reset / apply_raw_action (which also reports success) / get_* / reward / sim_time
until ``close`` or disconnect. A backend serves one connection at a time.
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from capx.gateway.gateway import EnvFactory
from capx.gateway.low_level import LowLevelEnv, StepOutcome
from capx.protocol.actions import RawAction, decode_action, encode_action
from capx.protocol.framing import (
    bind_unix_socket,
    connect_unix_socket,
    recv_framed,
    send_framed,
)
from capx.protocol.taskpack import TaskSpec

_logger = logging.getLogger(__name__)


class BackendAddress(NamedTuple):
    """The Unix socket one env-backend process listens on."""

    path: str

    def __str__(self) -> str:
        return self.path


class EnvBackendServer:
    """Threaded socket server hosting one simulator via a local ``EnvFactory``.

    Its sim is not safe to share across concurrent connections; run one backend
    process per lane and let :class:`RemoteEnvFactory` route across them. A second
    connection waits until the first is done and its env closed: a gateway that
    gave up on a call may connect again while that call still runs here.
    """

    def __init__(
        self,
        factory: EnvFactory,
        path: str,
    ) -> None:
        self._factory = factory
        self._path = path
        self._sock = bind_unix_socket(path)
        self._stop = threading.Event()
        self._serving = threading.Lock()

    @property
    def address(self) -> BackendAddress:
        return BackendAddress(self._path)

    def start_background(self) -> EnvBackendServer:
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        env: LowLevelEnv | None = None
        with conn, self._serving:
            try:
                while not self._stop.is_set():
                    request = recv_framed(conn)
                    reply, env = self._dispatch(request, env)
                    send_framed(conn, reply)
            except (ConnectionError, EOFError, OSError):
                pass
            finally:
                if env is not None:
                    with contextlib.suppress(Exception):
                        env.close()

    def _dispatch(
        self, request: dict[str, Any], env: LowLevelEnv | None
    ) -> tuple[dict[str, Any], LowLevelEnv | None]:
        method = request["method"]
        payload = request.get("payload", {})
        try:
            if method == "ping":
                return {"ok": True, "result": {}}, env
            if method == "build":
                env = self._factory.build(TaskSpec.model_validate(payload["spec"]))
                return {"ok": True, "result": {"embodiment": env.embodiment}}, env
            if env is None:
                raise RuntimeError("no env built (send 'build' first)")
            return {"ok": True, "result": self._call(env, method, payload)}, env
        except Exception as exc:  # sim error travels back as a coded failure
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, env

    def _call(self, env: LowLevelEnv, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        if method == "reset":
            env.reset(seed=int(payload["seed"]))
            return {}
        if method == "apply_raw_action":
            outcome = env.apply_raw_action(decode_action(payload["action"]))
            return {
                "terminated": outcome.terminated,
                "truncated": outcome.truncated,
                "success": outcome.success,
            }
        if method == "get_proprio":
            return {"obs": env.get_proprio()}
        if method == "get_observation":
            return {"obs": env.get_observation()}
        if method == "compute_reward":
            return {"reward": float(env.compute_reward())}
        if method == "sim_time":
            return {"sim_time": float(env.sim_time)}
        if method == "close":
            env.close()
            return {}
        raise ValueError(f"unknown method: {method!r}")

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        with contextlib.suppress(OSError):
            os.unlink(self._path)


# A backend that is still importing its simulator refuses connections (or has not
# created its AF_UNIX socket yet). That is a slow start, not a crash, so the first
# connection waits this long for it.
BACKEND_START_TIMEOUT_S = 120.0

# A backend that stops answering must not hold its lane forever. Generous, because
# the first build of a BEHAVIOR scene takes minutes.
BACKEND_CALL_TIMEOUT_S = 1800.0

# How long a build waits for a free backend of its simulator when every one is
# leased (PRELIM_RULEBOOK.md §4.4); past it the reset fails as env_crash.
LEASE_TIMEOUT_S = 1800.0


def _connect_when_listening(address: BackendAddress, timeout_s: float) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return connect_unix_socket(address.path)
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


class RemoteLowLevelEnv:
    """Gateway-side ``LowLevelEnv`` proxy: one connected socket to one backend-hosted
    env, one RPC per method. A call the backend does not answer within
    ``call_timeout_s`` fails like any other sim error.

    ``on_close`` gets whether the connection broke; then the backend may still be
    running the call this side gave up on.
    """

    def __init__(
        self,
        sock: socket.socket,
        spec: TaskSpec,
        *,
        on_close: Callable[[bool], None] | None = None,
        call_timeout_s: float = BACKEND_CALL_TIMEOUT_S,
    ) -> None:
        self._on_close = on_close
        # Set once a call fails on the socket itself: its reply may still be on the
        # way, so no later call can trust what it reads.
        self._broken = False
        self._sock = sock
        self._sock.settimeout(call_timeout_s)
        try:
            self.embodiment: str = self._rpc("build", {"spec": spec.model_dump(mode="json")})[
                "embodiment"
            ]
        except BaseException:
            self.close()
            raise

    def _rpc(self, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._broken:
            raise ConnectionError("the connection to the env backend failed earlier")
        try:
            send_framed(self._sock, {"method": method, "payload": payload or {}})
            resp = recv_framed(self._sock)
        except BaseException:
            self._broken = True
            raise
        if not resp.get("ok", False):
            raise RuntimeError(resp.get("error", "env backend error"))
        return resp["result"]

    @property
    def sim_time(self) -> float:
        return float(self._rpc("sim_time")["sim_time"])

    def reset(self, *, seed: int) -> None:
        self._rpc("reset", {"seed": seed})

    def apply_raw_action(self, action: RawAction) -> StepOutcome:
        r = self._rpc("apply_raw_action", {"action": encode_action(action)})
        return StepOutcome(
            terminated=bool(r["terminated"]),
            truncated=bool(r["truncated"]),
            success=bool(r["success"]),
        )

    def get_proprio(self) -> dict[str, Any]:
        return self._rpc("get_proprio")["obs"]

    def get_observation(self) -> dict[str, Any]:
        return self._rpc("get_observation")["obs"]

    def compute_reward(self) -> float:
        return float(self._rpc("compute_reward")["reward"])

    def close(self) -> None:
        try:
            # On a broken connection the backend closes the env when it sees EOF.
            if not self._broken:
                self._rpc("close")
        except Exception:
            pass
        finally:
            # The lease has to come back even if closing the socket throws, or the
            # backend is lost to the pool for the rest of the run.
            try:
                self._sock.close()
            finally:
                on_close, self._on_close = self._on_close, None
                if on_close is not None:
                    on_close(self._broken)


class RemoteEnvFactory:
    """Routes a task to a simulator backend by ``TaskSpec.simulator``.

    ``routes`` maps a simulator id to a pool of :class:`BackendAddress`, one backend
    process each. A built env leases its backend until it is closed, so two live envs
    never share a process; when a simulator's whole pool is leased, a build waits up
    to ``lease_timeout_s`` for one to come back, then fails. A backend whose
    connection broke stays leased until it answers again (or is gone), because it
    may still be running the call the gateway gave up on.

    ``on_demand`` is for backends that are not running yet: before the first
    connection to an AF_UNIX backend whose socket does not exist, it creates
    ``<socket>.want`` next to it, and whoever runs the backends starts that one.
    """

    supports_parallel = True

    def __init__(
        self,
        routes: dict[str, list[BackendAddress]],
        *,
        start_timeout_s: float = BACKEND_START_TIMEOUT_S,
        call_timeout_s: float = BACKEND_CALL_TIMEOUT_S,
        lease_timeout_s: float = LEASE_TIMEOUT_S,
        on_demand: bool = False,
    ) -> None:
        self._routes = {sim: list(pool) for sim, pool in routes.items()}
        self._start_timeout_s = start_timeout_s
        self._call_timeout_s = call_timeout_s
        self._lease_timeout_s = lease_timeout_s
        self._on_demand = on_demand
        self._leased: set[tuple[str, BackendAddress]] = set()
        self._freed = threading.Condition()

    def build(self, spec: TaskSpec) -> RemoteLowLevelEnv:
        sim = spec.simulator
        pool = self._routes.get(sim)
        if not pool:
            raise KeyError(f"no backend route for simulator {sim!r}")
        address = self._take(sim, pool)
        try:
            if self._on_demand:
                _request_start(Path(address.path))
            sock = _connect_when_listening(address, self._start_timeout_s)
        except BaseException:
            self._release(sim, address)
            raise
        return RemoteLowLevelEnv(
            sock,
            spec,
            on_close=lambda broken: self._give_back(sim, address, broken=broken),
            call_timeout_s=self._call_timeout_s,
        )

    def _take(self, sim: str, pool: list[BackendAddress]) -> BackendAddress:
        def free() -> BackendAddress | None:
            return next((a for a in pool if (sim, a) not in self._leased), None)

        with self._freed:
            address = self._freed.wait_for(free, self._lease_timeout_s)
            if address is None:
                raise TimeoutError(
                    f"every {sim!r} backend stayed leased for {self._lease_timeout_s:.0f} s"
                )
            self._leased.add((sim, address))
            return address

    def _release(self, sim: str, address: BackendAddress) -> None:
        with self._freed:
            self._leased.discard((sim, address))
            self._freed.notify_all()

    def _give_back(self, sim: str, address: BackendAddress, *, broken: bool) -> None:
        if not broken:
            self._release(sim, address)
            return
        _logger.warning(
            "the %s backend at %s stopped answering; no env is built on it until it "
            "answers again or is restarted",
            sim,
            address,
        )
        threading.Thread(
            target=self._release_when_free,
            args=(sim, address),
            name=f"quarantine-{sim}",
            daemon=True,
        ).start()

    def _release_when_free(self, sim: str, address: BackendAddress) -> None:
        """Wait for the backend to answer a ping, which it does only after the call
        the gateway gave up on has ended and that env is closed (it serves one
        connection at a time). A backend that is gone is free too: the runner
        restarts a process that exited when it is next asked for."""
        try:
            with connect_unix_socket(address.path) as sock:
                send_framed(sock, {"method": "ping", "payload": {}})
                recv_framed(sock)
            _logger.warning("the %s backend at %s answers again", sim, address)
        except Exception:
            _logger.warning("the %s backend at %s is gone", sim, address)
        self._release(sim, address)


def _request_start(socket_path: Path) -> None:
    if socket_path.exists():
        return
    socket_path.with_name(socket_path.name + ".want").touch()


def parse_backend_routes(entries: list[str]) -> dict[str, list[BackendAddress]]:
    """Parse ``"simulator=unix:/path"`` specs into a route map.

    e.g. ``["robosuite=unix:/run/capx/robosuite.sock", "libero=unix:/run/capx/libero.sock"]``.
    Repeat a simulator to give it a backend pool, one process per lane.
    """
    routes: dict[str, list[BackendAddress]] = {}
    for entry in entries:
        sim, sep, addr = entry.partition("=")
        addr = addr.strip()
        if not sep or not addr.startswith("unix:"):
            raise ValueError(
                f"bad backend spec {entry!r}; expected 'simulator=unix:/path' "
                "(backends listen only on Unix sockets)"
            )
        path = addr[len("unix:") :]
        if not path:
            raise ValueError(f"bad backend spec {entry!r}; 'unix:' needs a socket path")
        routes.setdefault(sim.strip(), []).append(BackendAddress(path))
    return routes


__all__ = [
    "BackendAddress",
    "EnvBackendServer",
    "RemoteLowLevelEnv",
    "RemoteEnvFactory",
    "parse_backend_routes",
]
