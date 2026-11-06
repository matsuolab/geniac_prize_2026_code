"""Gateway-container entrypoint: host the refereed gateway over the msgpack socket.

Loads a task pack, builds the :class:`Gateway` over ``RemoteEnvFactory`` (this
container imports no simulator) and serves it with :class:`GatewayServer`.

Env vars:

- ``CAPX_TASKPACK`` — directory containing ``pack.yaml`` (default the public pack).
- ``CAPX_MODE``     — ``dev`` | ``official`` (enforcement).
- ``CAPX_SIMULATORS`` — comma list restricting the pack to these simulators (same
  as ``capx eval --simulators``). Unset serves every task in the pack.
- ``CAPX_HOST`` / ``CAPX_PORT`` — TCP bind address for the submission-facing socket.
- ``CAPX_UNIX`` — bind the submission-facing socket at this AF_UNIX path instead of
  TCP. The eval queue uses this: the submission container has no network stack, so
  the only surface it can reach is the socket bind-mounted into it.
- ``CAPX_CLOCK_STARTED_PATH`` — where to write the wall-clock time (epoch seconds)
  the job budget started, i.e. the submission's first contact. The eval queue
  reads it to know when to stop the submission.
- ``CAPX_JOB_FINISHED_PATH`` — where to write the wall-clock time the submission
  called ``finish()``. The eval queue stops the submission when it appears and ends
  its time taken there.
- ``CAPX_MASTER_SEED`` — root of the per-(task, trial) seed derivation; organizers
  pick a fresh one per official run.
- ``CAPX_BACKENDS`` (required) — comma list of ``simulator=host:port`` or
  ``simulator=unix:/path`` (e.g. ``"robosuite=env-robosuite:8100,libero=env-libero:8100"``).
- ``CAPX_BACKENDS_ON_DEMAND`` — ``1`` when the routed AF_UNIX backends are started
  only when first needed (``RemoteEnvFactory``'s ``on_demand``), as the eval queue
  does. ``CAPX_BACKEND_START_TIMEOUT_S`` is how long to wait for one to listen.
- ``CAPX_RESULTS_PATH`` — operator-side results dump file (``capx.gateway.results_dump``),
  rewritten every ``CAPX_RESULTS_INTERVAL_S`` seconds (default ``60``). Unset -> off.
  On SIGTERM the gateway ends the job and writes the final dump, which is how the eval
  queue stops scoring a submission it stops.

Each concurrent episode needs its own backend process, so a simulator with N routes
runs at most N episodes at once, and the gateway accepts as many connections (lanes)
as the largest pool.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.env_backend import (
    BACKEND_START_TIMEOUT_S,
    BackendAddress,
    RemoteEnvFactory,
    parse_backend_routes,
)
from capx.gateway.results_dump import install_results_dump
from capx.gateway.server import GatewayServer
from capx.protocol.taskpack import load_pack, load_pack_visibility

# Connections over the lane cap are refused with the reason; this many at once.
REFUSED_CONNECTION_SLOTS = 16


def _routes() -> dict[str, list[BackendAddress]]:
    backends = [e for e in os.environ.get("CAPX_BACKENDS", "").split(",") if e.strip()]
    if not backends:
        raise SystemExit("CAPX_BACKENDS is required: 'simulator=host:port,...'")
    return parse_backend_routes(backends)


def _maybe_install_results_dump(gateway: Gateway, taskpack: str) -> None:
    path = os.environ.get("CAPX_RESULTS_PATH", "").strip()
    if not path:
        return
    interval_s = float(os.environ.get("CAPX_RESULTS_INTERVAL_S", "60"))
    visibility = load_pack_visibility(taskpack)
    install_results_dump(gateway, visibility, path, interval_s)
    print(f"results dump enabled: writing to {path!r} every {interval_s}s", flush=True)


def _timestamp_writer(variable: str) -> Callable[[], None] | None:
    """Writes the current epoch time to the file ``variable`` names, if it names one."""
    raw = os.environ.get(variable, "").strip()
    if not raw:
        return None
    path = Path(raw)

    def write() -> None:
        partial = path.with_name(path.name + ".partial")
        partial.write_text(f"{time.time()}\n")
        partial.replace(path)

    return write


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )

    taskpack = os.environ.get("CAPX_TASKPACK", "taskpacks/original")
    mode = EnforcementMode(os.environ.get("CAPX_MODE", "official"))
    host = os.environ.get("CAPX_HOST", "0.0.0.0")
    port = int(os.environ.get("CAPX_PORT", "8000"))
    unix_path = os.environ.get("CAPX_UNIX", "").strip() or None
    master_seed = int(os.environ.get("CAPX_MASTER_SEED", "0"))

    pack, _ = load_pack(taskpack)
    tasks = {spec.task_id: spec for spec in pack.tasks}
    simulators = {s.strip() for s in os.environ.get("CAPX_SIMULATORS", "").split(",") if s.strip()}
    if simulators:
        tasks = {
            task_id: spec
            for task_id, spec in tasks.items()
            if spec.simulator in simulators
        }
        if not tasks:
            raise SystemExit(f"CAPX_SIMULATORS={sorted(simulators)} left no tasks in {taskpack!r}")
    routes = _routes()
    lanes = max(len(pool) for pool in routes.values())
    caps = {sim: len(pool) for sim, pool in sorted(routes.items())}
    factory = RemoteEnvFactory(
        routes,
        start_timeout_s=float(
            os.environ.get("CAPX_BACKEND_START_TIMEOUT_S", BACKEND_START_TIMEOUT_S)
        ),
        on_demand=os.environ.get("CAPX_BACKENDS_ON_DEMAND", "") == "1",
    )
    gateway = Gateway(
        tasks,
        factory,
        mode,
        job_timeout_s=pack.job_timeout_s(simulators or None)
        if mode is EnforcementMode.OFFICIAL
        else None,
        master_seed=master_seed,
        max_sessions=lanes,
        on_clock_start=_timestamp_writer("CAPX_CLOCK_STARTED_PATH"),
        on_finish=_timestamp_writer("CAPX_JOB_FINISHED_PATH"),
    )
    _maybe_install_results_dump(gateway, taskpack)

    server = GatewayServer(
        gateway,
        host=host,
        port=port,
        unix_path=unix_path,
        max_connections=lanes + REFUSED_CONNECTION_SLOTS,
    )
    print(
        f"gateway serving {len(tasks)} task(s) from {taskpack!r} [{mode.value}] "
        f"via remote backends {caps} on {unix_path or f'{server.host}:{server.port}'} "
        f"across {lanes} lane(s)",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
