"""Config tests: numeric decisions load and are enforced by the gateway."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from capx.config import load_eval_config
from capx.gateway import EnforcementMode, Gateway, Referee, StepOutcome
from capx.gateway.r1pro_adapter import R1ProLowLevelEnv
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    GatewayError,
    InProcessTransport,
    ProprioObs,
    R1ProRawAction,
)
from capx.protocol.actions import InvalidActionError, check_franka_bounds
from capx.protocol.taskpack import SeedPolicy, TaskSpec
from test_gateway import VALID_JOINTS, FakeFactory, make_tasks

OUT_OF_LIMITS = FrankaRawAction(joints=np.full(7, 10.0), gripper=1.0)


def test_defaults_load() -> None:
    config = load_eval_config()
    assert len(config.embodiments.franka.joint_lower) == 7
    assert config.submission.max_image_gb == 80.0


def test_config_rejects_malformed_limits_at_load() -> None:
    import pydantic

    from capx.config.eval_config import FrankaLimits

    with pytest.raises(pydantic.ValidationError, match="7 entries"):
        FrankaLimits(joint_lower=[-1.0] * 3, joint_upper=[1.0] * 3)
    with pytest.raises(pydantic.ValidationError, match="strictly below"):
        FrankaLimits(joint_lower=[1.0] * 7, joint_upper=[-1.0] * 7)


def test_behavior_tasks_must_pin_fixed_seeds() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError, match="fixed"):
        TaskSpec(
            task_id="t",
            embodiment="r1pro",
            simulator="behavior",
            instruction="go",
            env_target="fake",
            n_trials=1,
            trial_steplimit=100,
            seed_policy=SeedPolicy(mode="random"),
        )


def test_private_pack_rejects_privileged_tasks() -> None:
    import pydantic

    from capx.protocol.taskpack import TaskPackManifest

    task = dict(
        task_id="t",
        embodiment="franka",
        simulator="robosuite",
        instruction="go",
        env_target="fake",
        privileged=True,
    )
    limits = {"trial_time_s": 60, "n_trials": 1, "trial_steplimit": 10}
    with pytest.raises(pydantic.ValidationError, match="privileged"):
        TaskPackManifest.model_validate(
            {"pack_id": "p", "visibility": "private", "limits": limits, "tasks": [task]}
        )


def test_score_weights_cover_every_simulator_of_the_pack() -> None:
    import pydantic

    from capx.protocol.taskpack import TaskPackManifest

    tasks = [
        dict(task_id=f"t{i}", embodiment="franka", simulator=sim, instruction="go", env_target="fake")
        for i, sim in enumerate(["robosuite", "libero"])
    ]
    pack = {"pack_id": "p", "limits": {"n_trials": 1, "trial_steplimit": 10}, "tasks": tasks}
    with pytest.raises(pydantic.ValidationError, match=r"score_weights has no \['libero'\]"):
        TaskPackManifest.model_validate({**pack, "score_weights": {"robosuite": 1.0}})
    with pytest.raises(pydantic.ValidationError, match="greater than 0"):
        TaskPackManifest.model_validate({**pack, "score_weights": {"robosuite": 1.0, "libero": 0}})
    weighed = {"robosuite": 0.5, "libero": 0.3, "behavior": 0.2}
    assert TaskPackManifest.model_validate({**pack, "score_weights": weighed}).score_weights == weighed


def test_check_franka_bounds() -> None:
    ok = FrankaRawAction(joints=np.zeros(7), gripper=0.5)
    check_franka_bounds(ok, [-1.0] * 7, [1.0] * 7)  # in range: no raise
    with pytest.raises(InvalidActionError):
        check_franka_bounds(OUT_OF_LIMITS, [-1.0] * 7, [1.0] * 7)


def test_a_rejected_action_says_which_joint_and_by_how_much() -> None:
    """This ends the episode, so the message is the last thing the team reads.

    "outside published joint limits" does not distinguish a micro-radian of IK
    slop from a target a radian past the stop, and those want opposite fixes.
    """
    joints = np.zeros(7)
    joints[3] = 1.25
    action = FrankaRawAction(joints=joints, gripper=0.5)
    with pytest.raises(InvalidActionError) as exc:
        check_franka_bounds(action, [-1.0] * 7, [1.0] * 7)
    message = str(exc.value)
    assert "joint 3" in message
    assert "0.250000 rad above" in message
    assert "1.250000 vs 1.000000" in message


def test_an_invalid_action_is_refused_and_the_trial_goes_on() -> None:
    factory = FakeFactory()
    gateway = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, OUT_OF_LIMITS)
    assert exc.value.code is ErrorCode.INVALID_ACTION
    assert factory.built[0]._cycles == 0  # never reached the simulator

    client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0))
    assert factory.built[0]._cycles == 1
    assert gateway.trials_used("t") == 1


# -- R1Pro action-dim bounds (no sim: a fake stands in for OmniGibson) -----------


class _FakeR1ProEnv:
    """Bounds-check plumbing only; no real BEHAVIOR/OmniGibson physics."""

    embodiment = "r1pro"
    sim_time = 0.0

    def reset(self, *, seed: int) -> None:
        pass

    def apply_raw_action(self, action) -> StepOutcome:
        return StepOutcome()

    def get_proprio(self):
        return {
            "robot_joint_pos": np.zeros(28, dtype=np.float32),
            "robot_cartesian_pos": np.zeros(14, dtype=np.float32),
        }

    def get_observation(self):
        return self.get_proprio()

    def compute_reward(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class _FakeR1ProFactory:
    supports_parallel = True

    def build(self, spec) -> _FakeR1ProEnv:
        return _FakeR1ProEnv()


def _r1pro_task() -> dict[str, TaskSpec]:
    spec = TaskSpec(
        task_id="t",
        embodiment="r1pro",
        simulator="behavior",
        instruction="do it",
        env_target="fake",
        n_trials=1,
        trial_steplimit=100,
        seed_policy=SeedPolicy(mode="fixed", seeds=[1]),
    )
    return {"t": spec}


def test_r1pro_action_dim_is_pinned() -> None:
    config = load_eval_config()
    assert config.embodiments.r1pro.action_dim == 25


def test_gateway_rejects_wrong_length_r1pro_action() -> None:
    gateway = Gateway(_r1pro_task(), _FakeR1ProFactory(), EnforcementMode.DEV)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, R1ProRawAction(action=np.zeros(10)))
    assert exc.value.code is ErrorCode.INVALID_ACTION


def test_gateway_accepts_correct_length_r1pro_action() -> None:
    gateway = Gateway(_r1pro_task(), _FakeR1ProFactory(), EnforcementMode.DEV)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    result = client.step(reset.episode_id, R1ProRawAction(action=np.zeros(25)))
    assert result.terminated is False


def test_r1pro_low_level_env_satisfies_low_level_env_protocol() -> None:
    """Structural check only: verifies the adapter's shape matches the ``LowLevelEnv``
    seam the gateway drives, without instantiating it or touching OmniGibson.

    Can't use ``isinstance(obj, LowLevelEnv)`` here the way the fakes in
    ``test_gateway.py`` do: ``LowLevelEnv.sim_time`` is a property, and
    ``runtime_checkable`` probes it with ``hasattr`` — which would invoke
    ``R1ProLowLevelEnv.sim_time``'s real (lazy) ``import omnigibson`` and blow up
    with ``ModuleNotFoundError`` instead of the ``AttributeError`` ``hasattr``
    expects. ``inspect.getattr_static`` checks the same shape without triggering
    any descriptor.
    """
    assert hasattr(R1ProLowLevelEnv, "embodiment")
    assert isinstance(inspect.getattr_static(R1ProLowLevelEnv, "sim_time"), property)
    for name in (
        "reset",
        "apply_raw_action",
        "get_proprio",
        "get_observation",
        "compute_reward",
        "close",
    ):
        assert callable(inspect.getattr_static(R1ProLowLevelEnv, name))
