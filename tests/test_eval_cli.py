"""``capx eval`` wiring, with fake simulator backends in place of real ones."""

from __future__ import annotations

import json

import pytest

from capx.cli.eval import EvalArgs, _build_factory, main
from capx.gateway.env_backend import EnvBackendServer
from socket_paths import fresh_socket_path
from test_gateway import CAMERA, FakeFactory, make_tasks


def _write_taskpack(tmp_path, *, n_trials: int = 2) -> str:
    (tmp_path / "pack.yaml").write_text(
        "pack_id: p\n"
        "visibility: public\n"
        "limits: {trial_time_s: 60, n_trials: 1, trial_steplimit: 100}\n"
        "tasks:\n"
        "  - task_id: t\n"
        "    embodiment: franka\n"
        "    simulator: robosuite\n"
        "    instruction: go\n"
        f"    camera_names: [{CAMERA}]\n"
        "    env_target: fake\n"
        f"    n_trials: {n_trials}\n"
        "    seed_policy:\n"
        "      mode: fixed\n"
        "      seeds: [7, 8]\n"
    )
    return str(tmp_path)


def test_build_factory_rejects_in_process_for_non_robosuite_pack() -> None:
    tasks = make_tasks(simulator="libero")
    with pytest.raises(SystemExit, match="only hosts robosuite"):
        _build_factory([], tasks)


def test_build_factory_needs_a_route_for_every_simulator() -> None:
    with pytest.raises(SystemExit, match="no --backend route"):
        _build_factory(["libero=unix:/run/libero.sock"], make_tasks())


def _write_multi_sim_taskpack(tmp_path) -> str:
    (tmp_path / "pack.yaml").write_text(
        "pack_id: p\n"
        "visibility: public\n"
        "limits: {trial_time_s: 60, n_trials: 1, trial_steplimit: 100}\n"
        "tasks:\n"
        "  - task_id: t\n"
        "    embodiment: franka\n"
        "    simulator: robosuite\n"
        "    instruction: go\n"
        f"    camera_names: [{CAMERA}]\n"
        "    env_target: fake\n"
        "    n_trials: 1\n"
        "    seed_policy: {mode: fixed, seeds: [7]}\n"
        "  - task_id: u\n"
        "    embodiment: franka\n"
        "    simulator: libero\n"
        "    instruction: go\n"
        f"    camera_names: [{CAMERA}]\n"
        "    env_target: fake\n"
        "    n_trials: 1\n"
        "    seed_policy: {mode: fixed, seeds: [7]}\n"
    )
    return str(tmp_path)


def test_eval_cli_simulators_filter_selects_a_subset(tmp_path, capsys) -> None:
    b = EnvBackendServer(FakeFactory(solve_at_cycle=1), fresh_socket_path()).start_background()
    try:
        main(
            EvalArgs(
                taskpack=_write_multi_sim_taskpack(tmp_path),
                mode="dev",
                backend=[f"robosuite=unix:{b.address.path}"],
                simulators=["robosuite"],
                max_steps=2,
            )
        )
        board = json.loads(capsys.readouterr().out)
        assert set(board["tasks"]) == {"t"}  # the libero task was filtered out
    finally:
        b.close()


def test_eval_cli_simulators_filter_rejects_unknown_simulator(tmp_path) -> None:
    with pytest.raises(SystemExit, match="not in the pack"):
        main(
            EvalArgs(
                taskpack=_write_multi_sim_taskpack(tmp_path),
                mode="dev",
                simulators=["behavior"],
            )
        )
