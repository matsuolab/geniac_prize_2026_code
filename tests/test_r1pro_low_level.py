"""BEHAVIOR's low-level env: scoring a task must never move the simulation.

The gateway checks reward and success after every step, so a success check that
stepped the sim would push the robot between the team's own actions. OmniGibson
needs Isaac Sim, so where it is not installed its modules are stood in for here;
the env itself runs on fakes of the OmniGibson objects it touches.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import capx.envs.simulators  # noqa: F401  — before any stand-in, so the registry stays clean

_STUBBED = (
    "omnigibson",
    "omnigibson.macros",
    "omnigibson.action_primitives",
    "omnigibson.action_primitives.starter_semantic_action_primitives",
    "omnigibson.action_primitives.curobo",
    "omnigibson.action_primitives.action_primitive_set_base",
    "omnigibson.utils",
    "omnigibson.utils.transform_utils",
    "omnigibson.utils.python_utils",
    "omnigibson.object_states",
    "omnigibson.object_states.toggle",
    "omnigibson.metrics",
    "torch",
)


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


@pytest.fixture(scope="module")
def r1pro_b1k():
    with pytest.MonkeyPatch.context() as mp:
        if importlib.util.find_spec("omnigibson") is None:
            for name in _STUBBED:
                if name == "torch" and importlib.util.find_spec("torch") is not None:
                    continue
                mp.setitem(sys.modules, name, _StubModule(name))
        mp.delitem(sys.modules, "capx.envs.simulators.r1pro_b1k", raising=False)
        module = importlib.import_module("capx.envs.simulators.r1pro_b1k")
        # The real module stays importable afterwards; this copy only lives here.
        mp.setitem(sys.modules, "capx.envs.simulators.r1pro_b1k", module)
        mp.setattr(module, "og", MagicMock(name="og"))
        yield module


class _Vec(np.ndarray):
    def clone(self) -> _Vec:
        return self.copy()


def _vec(*values: float) -> _Vec:
    return np.asarray(values, dtype=np.float64).view(_Vec)


class _FakeOgEnv:
    def __init__(self) -> None:
        self.steps = 0
        self.next_success = False
        self.instance = "before load"

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self.steps += 1
        info = {"done": {"success": self.next_success}}
        return {"robot_r1": {}}, 0.0, False, False, info

    def reset(self) -> None:
        pass

    def get_obs(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"robot_r1": {}, "instance": self.instance}, {}


class _FakeMetric:
    """OmniGibson 3.9's metric hooks."""

    def reset(self, env: Any) -> None:
        pass

    def step(self, env: Any, *args: Any) -> None:
        pass

    def aggregate(self, env: Any) -> dict[str, Any]:
        return {"q_score": {"final": 0.25}}


class _FakeObject:
    def __init__(self, z: float) -> None:
        self.z = z

    def get_position_orientation(self) -> tuple[_Vec, _Vec]:
        return _vec(0.0, 0.0, self.z), _vec(0.0, 0.0, 0.0, 1.0)


def _env(module, task_name: str = "assembling_gift_baskets"):
    env = object.__new__(module.R1ProBehaviourLowLevel)
    env.task_name = task_name
    env.env = _FakeOgEnv()
    env.robot = MagicMock(name="robot")
    env.controller = MagicMock(name="controller")
    env.metrics = [_FakeMetric()]
    env.task_relevant_obj = {}
    env._sim_step_count = 0
    env._last_success = False
    env._record_frames = False
    env._frame_buffer = []
    env.loaded: list[int] = []
    env.load_task_instance = env.loaded.append
    return env


def test_scoring_reads_the_last_step_and_never_steps(r1pro_b1k) -> None:
    env = _env(r1pro_b1k)
    env.step(np.zeros(3))
    assert env.task_completed() is False
    assert env.compute_reward() == 0.25

    env.env.next_success = True
    env.step(np.zeros(3))
    for _ in range(3):
        assert env.task_completed() is True
        assert env.compute_reward() == 1
    assert env.env.steps == 2
    assert env._sim_step_count == 2


def test_reset_forgets_the_previous_episodes_success(r1pro_b1k) -> None:
    env = _env(r1pro_b1k)
    env.env.next_success = True
    env.step(np.zeros(3))
    env.reset(options={"trial": 4})
    assert env.loaded == [4]
    assert env.task_completed() is False
    assert env._sim_step_count == 0


def test_reset_takes_the_baseline_and_observation_from_the_loaded_instance(r1pro_b1k) -> None:
    env = _env(r1pro_b1k, task_name="turning_on_radio")
    radio = _FakeObject(z=0.5)
    env.task_relevant_obj = {"radio_receiver.n.01_1": radio}

    def load(instance_id: int) -> None:
        radio.z = 0.8
        env.env.instance = f"instance {instance_id}"

    env.load_task_instance = load
    obs, _ = env.reset(seed=2)
    assert env.initial_radio_pose[2] == pytest.approx(0.8)
    assert obs["instance"] == "instance 2"
