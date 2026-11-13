"""Env-backend entrypoint: host one simulator behind the gateway
(``capx.gateway.env_backend``). Run one per simulator and lane.

Env vars:

- ``CAPX_SIMULATOR`` — the simulator id this backend serves, the one tasks are
  routed by: ``robosuite`` (default) | ``libero`` | ``behavior``, or a deployment's
  own id.
- ``CAPX_ENV_FACTORY`` — the ``Simulator`` whose factory builds the envs; defaults to
  ``CAPX_SIMULATOR``. An image serving its own id sets it.
- ``CAPX_TASKPACK`` — the pack directory, when the pack ships its own envs. Its
  ``register_modules`` for this simulator are imported here, since this is the
  process that builds them.
- ``CAPX_BACKEND_UNIX`` (required) — AF_UNIX path of the gateway-facing socket. There
  is no TCP option: on a shared node any port is reachable by other users' jobs,
  whereas a socket in a run-private directory is not.
- ``CAPX_BACKEND_UNIX_OWNER`` — ``uid:gid`` to hand the socket to. The socket is
  0600, so a backend running as root in Docker must give it to the user whose ssh
  forwards it to the gateway (the eval queue's remote backends).
"""

from __future__ import annotations

import os

from capx.gateway.env_backend import EnvBackendServer
from capx.gateway.gateway import EnvFactory
from capx.protocol.taskpack import Simulator, import_pack_modules


def _local_factory(simulator: Simulator) -> EnvFactory:
    if simulator is Simulator.ROBOSUITE:
        from capx.gateway.robosuite_adapter import RobosuiteEnvFactory

        return RobosuiteEnvFactory()
    if simulator is Simulator.LIBERO:
        from capx.gateway.libero_adapter import LiberoEnvFactory

        return LiberoEnvFactory()
    from capx.gateway.r1pro_adapter import R1ProEnvFactory

    return R1ProEnvFactory()


def _factory_kind(simulator: str, factory: str) -> Simulator:
    """The ``Simulator`` whose factory serves ``simulator``: ``factory``, else itself."""
    name = factory or simulator
    try:
        return Simulator(name)
    except ValueError:
        raise SystemExit(
            f"CAPX_ENV_FACTORY / CAPX_SIMULATOR {name!r} is not one of "
            f"{[member.value for member in Simulator]}"
        ) from None


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    simulator = os.environ.get("CAPX_SIMULATOR", "robosuite")
    kind = _factory_kind(simulator, os.environ.get("CAPX_ENV_FACTORY", "").strip())
    taskpack = os.environ.get("CAPX_TASKPACK", "").strip()
    if taskpack:
        import_pack_modules(taskpack, simulator)
    unix_path = os.environ.get("CAPX_BACKEND_UNIX", "").strip()
    if not unix_path:
        raise SystemExit("CAPX_BACKEND_UNIX is required: backends listen only on a Unix socket")
    server = EnvBackendServer(_local_factory(kind), unix_path)
    owner = os.environ.get("CAPX_BACKEND_UNIX_OWNER", "").strip()
    if owner:
        uid, gid = (int(part) for part in owner.split(":"))
        os.chown(unix_path, uid, gid)
    print(f"env backend ({simulator}) serving on {server.address}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
