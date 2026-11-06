"""The LIBERO loader, and a smoke test of the LIBERO install itself.

The loader tests use a stand-in env and run anywhere. The smoke test builds a real
LIBERO-PRO scene without a renderer, so it needs only LIBERO and MuJoCo.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _stub_capx_integrations_package() -> None:
    """Import ``capx.integrations.libero`` without the parent package's heavy imports.

    ``capx/integrations/__init__.py`` pulls the whole perception stack (open3d, jax),
    which a sim-free checkout lacks. When it does not import, a bare package object
    stands in for it, as in ``tests/submission/test_code_env.py``.
    """
    if "capx.integrations" in sys.modules:
        return
    try:
        import capx.integrations  # noqa: F401

        return
    except ImportError:
        pass
    import capx

    pkg = types.ModuleType("capx.integrations")
    pkg.__path__ = [str(Path(capx.__file__).parent / "integrations")]
    sys.modules["capx.integrations"] = pkg


_stub_capx_integrations_package()

import capx.integrations.libero as libero_loader  # noqa: E402
from capx.integrations.libero import LiberoHandle  # noqa: E402


class _FakeLiberoEnv:
    """Returns a different observation from ``reset`` and ``set_init_state``, as LIBERO does."""

    def __init__(self) -> None:
        self.seeds: list[int | None] = []

    def seed(self, seed: int | None) -> None:
        self.seeds.append(seed)

    def reset(self) -> dict[str, Any]:
        return {"source": "reset"}

    def set_init_state(self, state: Any) -> dict[str, Any]:
        return {"source": "init_state", "state": state}


def test_reset_returns_the_observation_of_the_init_state_it_loaded() -> None:
    handle = LiberoHandle(
        env=_FakeLiberoEnv(),
        suite_name="libero_spatial",
        task_id=0,
        task_language="pick up the bowl",
        init_states=["first", "second"],
    )
    obs, info = handle.reset(seed=3)
    assert obs == {"source": "init_state", "state": "first"}
    assert info == {}


def test_reset_without_init_states_returns_the_reset_observation() -> None:
    handle = LiberoHandle(
        env=_FakeLiberoEnv(), suite_name="s", task_id=0, task_language="", init_states=None
    )
    assert handle.reset(seed=None)[0] == {"source": "reset"}


def _fake_libero_utils(monkeypatch, init_states_root: Path) -> None:
    libero = types.ModuleType("libero")
    utils = types.ModuleType("libero.utils")
    vars(utils)["get_libero_path"] = lambda kind: str(init_states_root)
    vars(libero)["utils"] = utils
    monkeypatch.setitem(sys.modules, "libero", libero)
    monkeypatch.setitem(sys.modules, "libero.utils", utils)


class _FakeSuite:
    def get_task(self, task_id: int) -> Any:
        return types.SimpleNamespace(problem_folder="folder", init_states_file=f"task{task_id}.init")


def test_init_states_load_without_a_global_pickle_override(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    states = np.arange(12.0).reshape(3, 4)
    (tmp_path / "folder").mkdir()
    torch.save(states, tmp_path / "folder" / "task2.init")
    _fake_libero_utils(monkeypatch, tmp_path)
    monkeypatch.delenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", raising=False)

    loaded = libero_loader.load_init_states(_FakeSuite(), 2)

    np.testing.assert_array_equal(loaded, states)
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" not in os.environ


def test_missing_init_states_name_both_places_looked(tmp_path, monkeypatch) -> None:
    pytest.importorskip("torch")
    _fake_libero_utils(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="task7.init"):
        libero_loader.load_init_states(_FakeSuite(), 7)


def test_libero_scene_builds_and_steps() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    benchmark = pytest.importorskip("libero.benchmark")
    env_wrapper = pytest.importorskip("libero.envs.env_wrapper")
    from libero.utils import get_libero_path

    task = benchmark.get_benchmark_dict()["libero_10"]().get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = env_wrapper.ControlEnv(
        bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False
    )
    try:
        env.seed(0)
        env.reset()
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 7)
        assert np.asarray(obs["robot0_joint_pos"]).shape == (7,)
    finally:
        env.close()


def test_reset_leaves_the_whole_step_limit_to_the_episode() -> None:
    """The settling steps after a reset must not count against the task's step limit."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    pytest.importorskip("libero.benchmark")
    libero_env = pytest.importorskip("capx.envs.simulators.libero")
    step_limit = 5  # fewer than the 10 settling steps
    env = libero_env.FrankaLiberoTask(suite_name="libero_spatial", task_id=0, max_steps=step_limit)
    try:
        env.reset(seed=1)
        assert env.handle.env.env.timestep == 0
        assert env._sim_step_count == 0
        for _ in range(step_limit):
            env._step_once()
        with pytest.raises(ValueError, match="terminated episode"):
            env._step_once()
    finally:
        env.close()
