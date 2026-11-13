"""``capx eval`` — run a task pack through the gateway with the baseline policy.

Without ``--backend`` the gateway builds Robosuite envs in-process; select the
Robosuite tasks of a mixed pack with ``--simulators robosuite``. With ``--backend``
it routes each task to a simulator backend process; list every
``simulator=unix:/path`` after one ``--backend`` flag.

    python -m capx.cli.eval --taskpack taskpacks/original --mode dev --simulators robosuite
    python -m capx.cli.eval --taskpack <pack> \\
        --backend robosuite=unix:$HOME/.capx/robosuite.sock libero=unix:$HOME/.capx/libero.sock
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import tyro
import yaml
from pydantic import ValidationError

from capx.agent_client import PolicyFactory, baseline_policy_factory, run_eval
from capx.config import EvalConfig, load_eval_config
from capx.gateway import EnforcementMode, Gateway
from capx.gateway.gateway import EnvFactory
from capx.gateway.scoring import build_leaderboard
from capx.protocol import EnvClient, InProcessTransport
from capx.protocol.taskpack import (
    Simulator,
    TaskSpec,
    import_pack_modules,
    load_pack,
)


@dataclass
class EvalArgs:
    taskpack: str = "taskpacks/original"
    mode: Literal["dev", "official"] = "dev"
    config: str | None = None  # override the default numeric config
    max_trials_per_task: int = 1
    max_steps: int = 20
    master_seed: int = 0
    # 'simulator=unix:/path', several space-separated after one --backend flag
    # (repeating the flag does not accumulate). Empty -> in-process Robosuite.
    backend: list[str] = field(default_factory=list)
    # Run only the pack's tasks on these simulators; empty runs every task.
    simulators: list[str] = field(default_factory=list)


def _build_factory(backend: list[str], tasks: dict[str, TaskSpec]) -> EnvFactory:
    if not backend:
        non_robosuite = {spec.simulator for spec in tasks.values()} - {"robosuite"}
        if non_robosuite:
            raise SystemExit(
                f"the in-process env only hosts robosuite, but the pack needs "
                f"{sorted(non_robosuite)}; pass --backend routes for them (or run "
                "the full stack via docker compose)"
            )
        from capx.gateway.robosuite_adapter import RobosuiteEnvFactory

        return RobosuiteEnvFactory()

    from capx.gateway.env_backend import RemoteEnvFactory, parse_backend_routes

    routes = parse_backend_routes(backend)
    needed = {spec.simulator for spec in tasks.values()}
    missing = needed - set(routes)
    if missing:
        raise SystemExit(f"no --backend route for simulator(s): {sorted(missing)}")
    return RemoteEnvFactory(routes)


def _policy_factory(tasks: dict[str, TaskSpec], config: EvalConfig) -> PolicyFactory:
    try:
        return baseline_policy_factory(
            {spec.embodiment.value for spec in tasks.values()},
            r1pro_action_dim=config.embodiments.r1pro.action_dim,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main(args: EvalArgs) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")

    mode = EnforcementMode(args.mode)
    try:
        config = load_eval_config(args.config)
        pack, _ = load_pack(args.taskpack)
    except FileNotFoundError as exc:
        raise SystemExit(f"not found: {exc.filename}") from exc
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"invalid config or task pack: {exc}") from exc

    tasks = {spec.task_id: spec for spec in pack.tasks}
    if args.simulators:
        available = {spec.simulator for spec in tasks.values()}
        unknown = set(args.simulators) - available
        if unknown:
            raise SystemExit(
                f"--simulators {sorted(unknown)} not in the pack (it has {sorted(available)})"
            )
        tasks = {
            task_id: spec
            for task_id, spec in tasks.items()
            if spec.simulator in args.simulators
        }

    job_timeout_s = (
        pack.job_timeout_s(args.simulators or None) if mode is EnforcementMode.OFFICIAL else None
    )

    if not args.backend:
        import_pack_modules(args.taskpack, Simulator.ROBOSUITE)
    factory = _build_factory(args.backend, tasks)
    policy_factory = _policy_factory(tasks, config)

    with Gateway(
        tasks,
        factory,
        mode,
        config=config,
        job_timeout_s=job_timeout_s,
        master_seed=args.master_seed,
    ) as gateway:
        client = EnvClient(InProcessTransport(gateway.acquire()))
        run_eval(
            client,
            policy_factory,
            max_trials_per_task=args.max_trials_per_task,
            max_steps=args.max_steps,
        )
        aggregate = gateway.aggregate()

    board = build_leaderboard(aggregate, pack.visibility)
    print(json.dumps(board, indent=2))


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
