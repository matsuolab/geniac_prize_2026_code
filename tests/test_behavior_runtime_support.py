from pathlib import Path
from unittest.mock import Mock
import os
import sys
import types
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_r1pro_low_level import r1pro_b1k

from capx.behavior_runtime_support import configure_cpu_threads, configure_isaac_threads, read_camera_intrinsics, resolve_task_scene


def test_lane_thread_budget_applies_even_when_parent_exports_oversubscribed_defaults(monkeypatch):
    monkeypatch.setenv("CAPX_LANE_CPUS", "2")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")
    assert configure_cpu_threads() == 2
    for name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PXR_WORK_THREAD_LIMIT"]:
        assert os.environ[name] == "2"
    assert os.environ["TORCHINDUCTOR_COMPILE_THREADS"] == "1"


def test_isaac_does_not_override_lane_budget_with_32_workers(monkeypatch):
    app = types.SimpleNamespace(DEFAULT_LAUNCHER_CONFIG={"limit_cpu_threads": 32})
    monkeypatch.setitem(sys.modules, "isaacsim", types.SimpleNamespace(SimulationApp=app))
    monkeypatch.setenv("CAPX_LANE_CPUS", "1")
    assert configure_isaac_threads() == 4
    assert app.DEFAULT_LAUNCHER_CONFIG["limit_cpu_threads"] == 4
    assert app.DEFAULT_LAUNCHER_CONFIG["extra_args"] == ["--/persistent/physics/numThreads=1"]


def test_thread_configuration_module_does_not_import_numeric_libraries_first():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-I", "-c",
        f"import sys; sys.path.insert(0, {str(root)!r}); import capx.behavior_runtime_support; "
        "assert not any(x in sys.modules for x in ['numpy', 'torch', 'gymnasium', 'omnigibson'])"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_camera_warmup_returns_real_valid_matrix_after_delayed_renderer():
    real = np.array([[100., 0, 64], [0, 100, 64], [0, 0, 1]])
    read = Mock(side_effect=[AssertionError("intrinsic matrix for sensor: camera is degenerate!"), np.diag([0, 0, 1]), real])
    render = Mock()
    assert read_camera_intrinsics(read, render, sleep=lambda _: None) is real
    assert render.call_count == 2


def test_camera_warmup_does_not_swallow_other_assertions():
    with pytest.raises(AssertionError, match="unrelated"):
        read_camera_intrinsics(Mock(side_effect=AssertionError("unrelated")), Mock())


def test_invalid_calibration_has_bounded_failure_without_fabricated_values():
    with pytest.raises(RuntimeError, match="bounded"):
        read_camera_intrinsics(lambda: np.diag([0, 0, 1]), Mock(), attempts=3, sleep=lambda _: None)


@pytest.mark.parametrize("house", ["house_double_floor_lower", "house_single_floor", "house_double_floor_upper"])
def test_task_scene_resolver_uses_actual_house(tmp_path, house):
    target = tmp_path / "2025-challenge-task-instances/scenes" / house / "json/example_task_t_0_0_template.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    assert resolve_task_scene(tmp_path, "t") == (house, str(target))


def test_scene_resolver_reports_missing_template_before_omnigibson(tmp_path):
    with pytest.raises(FileNotFoundError, match="No scene template"):
        resolve_task_scene(tmp_path, "missing")


def test_public_pack_tasks_resolve_across_house_templates(tmp_path):
    import yaml
    root = Path(__file__).resolve().parents[1]
    pack = yaml.safe_load((root / "taskpacks/original/pack.yaml").read_text())
    tasks = [task for task in pack["tasks"] if task["simulator"] == "behavior"]
    houses = ["house_double_floor_lower", "house_single_floor", "house_double_floor_upper"]
    assert tasks
    for index, task in enumerate(tasks):
        kwargs = task.get("env_kwargs", {})
        activity = kwargs.get("activity_name")
        if activity is None:
            cfg = kwargs.get("controller_cfg", "r1pro_primitives.yaml")
            activity = yaml.safe_load((root / "capx/envs/simulators/configs" / cfg).read_text())["task"]["activity_name"]
        house = houses[index % len(houses)]
        target = tmp_path / "2025-challenge-task-instances/scenes" / house / "json" / f"example_task_{activity}_0_0_template.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}")
        assert resolve_task_scene(tmp_path, activity) == (house, str(target))


def test_native_reward_does_not_enumerate_legacy_demo_groundings(r1pro_b1k):
    from test_r1pro_low_level import _env
    for task in ["assembling_gift_baskets", "storing_food", "turning_on_radio", "picking_up_trash"]:
        env = _env(r1pro_b1k, task)
        env.metrics = [Mock(aggregate=Mock(side_effect=AssertionError("exponential demo metric")))]
        assert env.compute_reward() == 0.0
        env._last_success = True
        assert env.compute_reward() == 1.0


def test_close_releases_playing_global_scene_and_is_idempotent(r1pro_b1k):
    from test_r1pro_low_level import _env
    env = _env(r1pro_b1k)
    r1pro_b1k.og.reset_mock()
    env.close()
    env.close()
    r1pro_b1k.og.sim.stop.assert_called_once()
    r1pro_b1k.og.clear.assert_called_once()
    assert env.env is None


def test_backend_close_discards_reference_and_accepts_a_fresh_build():
    from capx.gateway.env_backend import EnvBackendServer
    from test_gateway import make_tasks
    server = object.__new__(EnvBackendServer)
    first, second = Mock(embodiment="franka"), Mock(embodiment="franka")
    server._factory = Mock(build=Mock(side_effect=[first, second]))
    request = {"method": "build", "payload": {"spec": make_tasks()["t"].model_dump(mode="json")}}
    reply, env = server._dispatch(request, None)
    assert reply["ok"] and env is first
    reply, env = server._dispatch({"method": "close"}, env)
    assert reply["ok"] and env is None
    first.close.assert_called_once()
    reply, env = server._dispatch(request, env)
    assert reply["ok"] and env is second


def test_backend_does_not_orphan_a_live_environment_on_second_build():
    from capx.gateway.env_backend import EnvBackendServer
    server = object.__new__(EnvBackendServer)
    server._factory = Mock()
    current = Mock()
    reply, env = server._dispatch({"method": "build"}, current)
    assert not reply["ok"] and env is current
    server._factory.build.assert_not_called()


def test_real_backend_dispatch_uses_actual_process_main_thread(monkeypatch):
    from capx.gateway.r1pro_adapter import _on_sim_thread
    monkeypatch.setenv("CAPX_SIM_MAIN_THREAD", "1")
    assert _on_sim_thread(threading.current_thread) is threading.main_thread()
    with ThreadPoolExecutor(max_workers=1) as worker:
        with pytest.raises(RuntimeError, match="process main thread"):
            worker.submit(_on_sim_thread, lambda: None).result()
