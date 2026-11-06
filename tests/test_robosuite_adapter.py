"""GPU-free checks of the Robosuite adapters: what ``get_proprio`` sends over the
backend link, and that the factory refuses an env with the wrong number of arms."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from test_gateway import make_tasks

from capx.gateway.robosuite_adapter import RobosuiteEnvFactory, RobosuiteLowLevelEnv
from capx.gateway.robosuite_bimanual_adapter import RobosuiteBimanualLowLevelEnv


class _FakeRobosuiteEnv:
    def _get_observations(self) -> dict[str, Any]:
        # Robosuite returns every enabled observable from its cache, camera images too.
        return {
            "robot0_joint_pos": np.zeros(7),
            "robot0_robotview_image": np.zeros((512, 512, 3), np.uint8),
            "robot0_robotview_depth": np.zeros((512, 512, 1), np.float32),
        }


class FakeRobosuiteBase:
    robosuite_env = _FakeRobosuiteEnv()

    def _compute_gripper_obs(self, obs: dict[str, Any]) -> None:
        obs["robot_joint_pos"] = np.zeros(8)
        obs["robot_cartesian_pos"] = np.zeros(8)


def test_proprio_carries_only_the_self_state() -> None:
    proprio = RobosuiteLowLevelEnv(FakeRobosuiteBase()).get_proprio()
    assert set(proprio) == {"robot_joint_pos", "robot_cartesian_pos"}


class OneArmEnv:
    """Stands in for a ``RobosuiteBaseEnv``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TwoArmEnv(OneArmEnv):
    """Stands in for ``RobosuiteTwoArmLiftEnv`` / ``RobosuiteHandoverEnv``."""

    def move_to_joints_blocking_both(self, *args: Any, **kwargs: Any) -> None: ...


@pytest.fixture
def built() -> Iterator[list[OneArmEnv]]:
    from capx.envs import base

    envs: list[OneArmEnv] = []

    def maker(cls: type[OneArmEnv]):  # noqa: ANN202
        def make(**kwargs: Any) -> OneArmEnv:
            envs.append(cls())
            return envs[-1]

        return make

    base.register_env("one_arm_env_under_test", maker(OneArmEnv))
    base.register_env("two_arm_env_under_test", maker(TwoArmEnv))
    yield envs
    for name in ("one_arm_env_under_test", "two_arm_env_under_test"):
        base._ENV_FACTORIES.pop(name, None)


@pytest.mark.parametrize(
    ("embodiment", "env_target", "arms"),
    [
        ("franka_bimanual", "one_arm_env_under_test", "one"),
        ("franka", "two_arm_env_under_test", "two"),
    ],
)
def test_an_env_with_the_wrong_number_of_arms_is_refused(
    built: list[OneArmEnv], embodiment: str, env_target: str, arms: str
) -> None:
    spec = make_tasks(embodiment=embodiment, env_target=env_target)["t"]
    with pytest.raises(ValueError, match=f"{arms}-armed"):
        RobosuiteEnvFactory(enable_render=False).build(spec)
    assert built[0].closed is True


def test_an_env_with_the_right_number_of_arms_gets_its_adapter(built: list[OneArmEnv]) -> None:
    factory = RobosuiteEnvFactory(enable_render=False)
    one = factory.build(make_tasks(env_target="one_arm_env_under_test")["t"])
    two = factory.build(
        make_tasks(embodiment="franka_bimanual", env_target="two_arm_env_under_test")["t"]
    )
    assert isinstance(one, RobosuiteLowLevelEnv)
    assert isinstance(two, RobosuiteBimanualLowLevelEnv)
