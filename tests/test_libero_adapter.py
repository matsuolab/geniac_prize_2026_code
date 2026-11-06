"""GPU-free structural tests for the LIBERO ``LowLevelEnv`` adapter.

``capx/gateway/libero_adapter.py`` needs LIBERO's vendored MuJoCo stack to run for
real (see the PR description for what real-sim E2E was / wasn't run). These tests
stand in a fake ``FrankaLiberoEnv`` double — mirroring the fields the real base env
exposes (``capx/envs/simulators/libero.py``) — to prove, without any sim installed:

- ``LiberoLowLevelEnv`` satisfies the ``LowLevelEnv`` protocol shape.
- ``apply_raw_action`` / ``get_proprio`` / ``get_observation`` route to the base
  env's existing control-state + observation methods correctly, and a cycle
  reports the base env's success right after it.
- ``LiberoEnvFactory`` stays importable (and its module-registration import doesn't
  raise) without LIBERO installed, only failing at ``build()`` — a registry lookup.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("viser")

from capx.gateway.low_level import LowLevelEnv  # noqa: E402
from capx.gateway.libero_adapter import LiberoEnvFactory, LiberoLowLevelEnv  # noqa: E402
from capx.protocol.actions import FrankaRawAction  # noqa: E402

VALID_JOINTS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])


class _FakeSim:
    def __init__(self) -> None:
        self.data = type("Data", (), {"time": 1.5})()


class _FakeLiberoControlEnv:
    """LIBERO's ``ControlEnv``: ``close`` frees the MuJoCo sim it wraps."""

    def __init__(self) -> None:
        self.sim = _FakeSim()
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeHandle:
    def __init__(self) -> None:
        self.env = _FakeLiberoControlEnv()


class FakeLiberoBase:
    """Stands in for ``FrankaLiberoEnv``: same public/private surface the adapter reads."""

    def __init__(self) -> None:
        self.handle = _FakeHandle()
        self._current_joints = np.zeros(7)
        self._gripper_fraction = 1.0
        self.gripper_metric_length = 0.04
        # Coincident base/gripper link frames -> get_proprio's transform reduces to
        # just the adapter's own fixed TCP offset, so the expected output is exact.
        self.base_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.gripper_link_wxyz_xyz = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._current_obs: dict[str, Any] = {
            "robot0_joint_pos": VALID_JOINTS.copy(),
            "robot0_gripper_qpos": np.array([0.04]),  # normalizes to 1.0 (open)
        }
        self.reset_seed: int | None = None

    def reset(self, *, seed: int | None = None) -> None:
        self.reset_seed = seed

    def _set_gripper(self, fraction: float) -> None:
        self._gripper_fraction = float(fraction)

    def _step_once(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        return {"agentview": {"images": {"rgb": np.zeros((2, 2, 3), np.uint8)}}}

    def compute_reward(self) -> float:
        return 0.5

    def task_completed(self) -> bool:
        return True

    def close(self) -> None:
        # The real env's own close, run on this stand-in.
        libero_env = pytest.importorskip("capx.envs.simulators.libero")
        libero_env.FrankaLiberoEnv.close(self)


def test_libero_low_level_env_satisfies_protocol() -> None:
    adapter = LiberoLowLevelEnv(FakeLiberoBase())
    assert isinstance(adapter, LowLevelEnv)
    assert adapter.embodiment == "franka"


def test_reset_delegates_to_base() -> None:
    base = FakeLiberoBase()
    adapter = LiberoLowLevelEnv(base)
    adapter.reset(seed=7)
    assert base.reset_seed == 7


def test_apply_raw_action_sets_control_state_and_steps() -> None:
    base = FakeLiberoBase()
    adapter = LiberoLowLevelEnv(base)
    outcome = adapter.apply_raw_action(FrankaRawAction(joints=VALID_JOINTS, gripper=0.25))
    np.testing.assert_allclose(base._current_joints, VALID_JOINTS)
    assert base._gripper_fraction == pytest.approx(0.25)
    assert outcome.success is True
    assert outcome.terminated is False
    assert outcome.truncated is False


def test_get_proprio_matches_the_obs_filter_contract() -> None:
    adapter = LiberoLowLevelEnv(FakeLiberoBase())
    proprio = adapter.get_proprio()
    assert set(proprio) == {"robot_joint_pos", "robot_cartesian_pos"}
    np.testing.assert_allclose(proprio["robot_joint_pos"], np.append(VALID_JOINTS, 1.0))
    cart = proprio["robot_cartesian_pos"]
    assert cart.shape == (8,)
    # Coincident base/gripper frames -> cartesian pos is exactly the adapter's fixed
    # TCP offset: translate (0, 0, -0.107), rotate pi/2 about Z.
    np.testing.assert_allclose(cart[:3], [0.0, 0.0, -0.107], atol=1e-9)
    np.testing.assert_allclose(cart[3:7], [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)], atol=1e-9)
    assert cart[7] == pytest.approx(1.0)  # gripper fraction, normalized


def test_get_observation_and_reward_delegate_to_base() -> None:
    base = FakeLiberoBase()
    adapter = LiberoLowLevelEnv(base)
    obs = adapter.get_observation()
    np.testing.assert_array_equal(obs["agentview"]["images"]["rgb"], base.get_observation()["agentview"]["images"]["rgb"])
    assert adapter.compute_reward() == pytest.approx(0.5)


def test_sim_time_reads_libero_mujoco_data() -> None:
    adapter = LiberoLowLevelEnv(FakeLiberoBase())
    assert adapter.sim_time == pytest.approx(1.5)


def test_close_frees_the_libero_sim() -> None:
    base = FakeLiberoBase()
    adapter = LiberoLowLevelEnv(base)
    adapter.close()
    assert base.handle.env.closed is True


def test_factory_importable_without_libero_installed() -> None:
    """capx.envs.simulators swallows the missing-LIBERO ImportError (prints and
    moves on, see its module docstring), so building the factory must not need
    LIBERO — only actually building an env does."""
    factory = LiberoEnvFactory()
    assert factory.supports_parallel is False

    from capx.envs.base import list_envs
    from capx.protocol.taskpack import TaskSpec

    env_target = "franka_libero_pick_place_low_level"
    if env_target in list_envs():
        pytest.skip(f"LIBERO is installed and {env_target!r} is registered in this env")

    with pytest.raises(KeyError, match="not registered"):
        factory.build(
            TaskSpec(
                task_id="t",
                embodiment="franka",
                simulator="libero",
                instruction="go",
                camera_names=["agentview"],
                env_target=env_target,
                n_trials=1,
                trial_steplimit=10,
            )
        )


def test_a_backend_serves_its_own_simulator_id_with_a_named_factory() -> None:
    import importlib

    from capx.protocol.taskpack import Simulator

    serve_env_backend = importlib.import_module("docker.serve_env_backend")
    assert serve_env_backend._factory_kind("libero", "") is Simulator.LIBERO
    kind = serve_env_backend._factory_kind("libero_variant", "libero")
    assert isinstance(serve_env_backend._local_factory(kind), LiberoEnvFactory)
    with pytest.raises(SystemExit, match="libero_variant"):
        serve_env_backend._factory_kind("libero_variant", "")
