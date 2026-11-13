"""Two arms over the same three-operation protocol.

Two of the paper's seven core Robosuite tasks are bimanual, and the gateway could
not express them: their envs drive 2 x (7 joints + gripper) and ``FrankaRawAction``
carries one arm's worth. What makes them work is a second embodiment, not a second
protocol — reset / step / get_observation are unchanged, and so is everything the
referee does.

The invariant worth holding onto is that **one step moves both arms**. The sim's
action is a single vector covering both, so splitting a cycle between them would be
different physics, and the action type is built so it cannot be split.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_socket_transport import VALID_JOINTS, make_tasks

from capx.gateway import EnforcementMode, Gateway, StepOutcome
from capx.protocol import (
    ArmProprio,
    BimanualFrankaRawAction,
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    InProcessTransport,
    ProprioObs,
)
from capx.protocol.actions import InvalidActionError, decode_action, encode_action
from capx.protocol.observations import ObsFilter, decode_obs, encode_obs
from capx.protocol.taskpack import SeedPolicy, TaskSpec

CAMERA = "agentview"


def valid_joints() -> np.ndarray:
    return np.stack([VALID_JOINTS, VALID_JOINTS])


def action(**kwargs) -> BimanualFrankaRawAction:
    params = {"joints": valid_joints(), "grippers": np.array([1.0, 0.0])}
    params.update(kwargs)
    return BimanualFrankaRawAction(**params)


# --- the action -------------------------------------------------------------


def test_both_arms_travel_in_one_action() -> None:
    """Splitting a cycle between the arms would be different physics."""
    decoded = decode_action(encode_action(action()))
    assert isinstance(decoded, BimanualFrankaRawAction)
    assert decoded.joints.shape == (2, 7)
    np.testing.assert_allclose(decoded.grippers, [1.0, 0.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"joints": np.zeros((1, 7))}, "shape"),
        ({"joints": np.zeros(7)}, "shape"),
        ({"grippers": np.array([1.0])}, "shape"),
        ({"grippers": np.array([1.0, 1.5])}, r"\[0, 1\]"),
        ({"joints": np.full((2, 7), np.nan)}, "non-finite"),
    ],
)
def test_a_malformed_bimanual_action_is_refused(kwargs, message) -> None:
    with pytest.raises(InvalidActionError, match=message):
        encode_action(action(**kwargs))


def test_a_single_arm_action_still_decodes_unchanged() -> None:
    """The second embodiment must not disturb the first."""
    decoded = decode_action(encode_action(FrankaRawAction(joints=VALID_JOINTS, gripper=0.5)))
    assert isinstance(decoded, FrankaRawAction)


# --- the observation --------------------------------------------------------


def test_per_arm_state_survives_the_wire() -> None:
    proprio = ProprioObs(
        joint_pos=np.zeros(8),
        cartesian_pos=np.zeros(8),
        sim_time=1.0,
        arms=[
            ArmProprio(
                joint_pos=np.full(8, float(i)),
                cartesian_pos=np.full(8, float(i) + 0.5),
                base_pose=np.full(7, float(i) + 0.25),
            )
            for i in range(2)
        ],
    )
    decoded = decode_obs(encode_obs(proprio))
    assert isinstance(decoded, ProprioObs)
    assert len(decoded.arms) == 2
    np.testing.assert_allclose(decoded.arms[1].joint_pos, np.ones(8))
    np.testing.assert_allclose(decoded.arms[1].base_pose, np.full(7, 1.25))


def test_a_single_arm_observation_carries_no_arms_field() -> None:
    """Single-arm runs keep the wire they had."""
    encoded = encode_obs(ProprioObs(joint_pos=np.zeros(8), cartesian_pos=np.zeros(8), sim_time=0.0))
    assert "arms" not in encoded
    assert decode_obs(encoded).arms == []


def test_the_filter_allowlists_arms_the_same_way_as_everything_else() -> None:
    """An adapter must not be able to smuggle state through an arm entry."""
    raw = {
        "robot_joint_pos": np.zeros(8),
        "robot_cartesian_pos": np.zeros(8),
        "arms": [
            {
                "robot_joint_pos": np.ones(8),
                "robot_cartesian_pos": np.full(8, 2.0),
                "base_pose": np.full(7, 3.0),
                "pot_poses": np.full(7, 99.0),  # privileged; must not survive
            }
        ],
    }
    proprio = ObsFilter(camera_names=[]).to_proprio(raw, sim_time=0.0)
    assert len(proprio.arms) == 1
    assert not hasattr(proprio.arms[0], "pot_poses")
    np.testing.assert_allclose(proprio.arms[0].base_pose, np.full(7, 3.0))


# --- the gateway ------------------------------------------------------------


class FakeBimanualEnv:
    embodiment = "franka_bimanual"

    def __init__(self) -> None:
        self.sim_time = 0.0
        self.commands: list[tuple[np.ndarray, np.ndarray]] = []
        # A real Panda never sits at all-zero joints -- q4's upper limit is
        # negative -- so starting there would make every hold-in-place command
        # an invalid_action.
        self._joints = valid_joints()

    def reset(self, *, seed: int) -> None:
        self.sim_time = 0.0

    def apply_raw_action(self, action) -> StepOutcome:  # noqa: ANN001
        self.sim_time += 0.05
        self._joints = np.asarray(action.joints)
        self.commands.append((np.asarray(action.joints), np.asarray(action.grippers)))
        return StepOutcome()

    def _arms(self):
        return [
            {
                "robot_joint_pos": np.concatenate([self._joints[i], [1.0]]).astype(np.float32),
                "robot_cartesian_pos": np.zeros(8, dtype=np.float32),
                "base_pose": np.full(7, float(i), dtype=np.float32),
            }
            for i in range(2)
        ]

    def get_proprio(self):
        arms = self._arms()
        return {
            "arms": arms,
            "robot_joint_pos": arms[0]["robot_joint_pos"],
            "robot_cartesian_pos": arms[0]["robot_cartesian_pos"],
        }

    def get_observation(self):
        obs = self.get_proprio()
        obs["pot_poses"] = {"pot": np.zeros(7, dtype=np.float32)}  # privileged
        obs[CAMERA] = {
            "images": {
                "rgb": np.full((4, 4, 3), 7, np.uint8),
                "depth": np.zeros((4, 4), np.float32),
            },
            "intrinsics": np.eye(3, dtype=np.float32),
            "pose": np.zeros(7, dtype=np.float32),
            # Both real two-arm envs publish pose_mat; the shim only falls back to
            # deriving it when an adapter omits it.
            "pose_mat": np.eye(4, dtype=np.float32),
        }
        return obs

    def compute_reward(self) -> float:
        return 0.0

    def close(self) -> None:
        pass


class FakeBimanualFactory:
    def __init__(self) -> None:
        self.built: list[FakeBimanualEnv] = []

    def build(self, spec) -> FakeBimanualEnv:  # noqa: ANN001
        env = FakeBimanualEnv()
        self.built.append(env)
        return env


def bimanual_tasks() -> dict[str, TaskSpec]:
    return {
        "t": TaskSpec(
            task_id="t",
            embodiment="franka_bimanual",
            simulator="robosuite",
            instruction="lift the pot",
            camera_names=[CAMERA],
            env_target="two_arm_lift_robosuite",
            n_trials=3,
            trial_steplimit=100,
            seed_policy=SeedPolicy(mode="fixed", seeds=[0, 1, 2]),
        )
    }


def client_for(factory=None):  # noqa: ANN001
    gateway = Gateway(bimanual_tasks(), factory or FakeBimanualFactory(), EnforcementMode.DEV)
    return EnvClient(InProcessTransport(gateway.acquire())), gateway


def test_a_bimanual_task_runs_the_ordinary_three_operations() -> None:
    client, _ = client_for()
    reset = client.reset("t")
    assert isinstance(reset.obs, FullObs)
    assert len(reset.obs.proprio.arms) == 2

    result = client.step(reset.episode_id, action())
    assert len(result.obs.arms) == 2
    np.testing.assert_allclose(result.obs.arms[0].joint_pos[:7], VALID_JOINTS, atol=1e-4)


def test_the_privileged_pot_pose_does_not_cross() -> None:
    client, _ = client_for()
    reset = client.reset("t")
    obs = client.get_observation(reset.episode_id)
    assert set(obs.cameras) == {CAMERA}
    assert not hasattr(obs.proprio, "pot_poses")


def test_a_single_arm_action_is_refused_for_a_two_arm_task() -> None:
    client, _ = client_for()
    reset = client.reset("t")
    with pytest.raises(GatewayError) as refused:
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0))
    assert refused.value.code is ErrorCode.INVALID_ACTION


def test_either_arm_out_of_limits_is_refused_and_says_which() -> None:
    client, _ = client_for()
    reset = client.reset("t")
    joints = valid_joints()
    joints[1] = np.full(7, 99.0)
    with pytest.raises(GatewayError) as refused:
        client.step(reset.episode_id, action(joints=joints))
    assert refused.value.code is ErrorCode.INVALID_ACTION
    assert "joints[1]" in refused.value.detail


def test_repeat_holds_the_command_for_both_arms() -> None:
    """A held command must not quietly become a one-arm hold."""
    factory = FakeBimanualFactory()
    client, gateway = client_for(factory)
    reset = client.reset("t")
    client.step(reset.episode_id, action(repeat=4))

    commands = factory.built[0].commands
    assert len(commands) == 4
    for joints, grippers in commands:
        assert joints.shape == (2, 7)
        np.testing.assert_allclose(grippers, [1.0, 0.0])
    assert gateway.timings()["episodes"][0]["steps"] == 4


def test_the_bimanual_baseline_holds_both_reset_poses() -> None:
    """A pack with a two-arm task must run under the shipped baseline, not refuse it."""
    from capx.agent_client import BimanualHoldPolicy, baseline_policy_factory, run_eval

    factory = FakeBimanualFactory()
    client, _ = client_for(factory)
    policies = baseline_policy_factory({"franka_bimanual"}, r1pro_action_dim=None)
    assert isinstance(policies("franka_bimanual"), BimanualHoldPolicy)

    run_eval(client, policies, max_steps=3)

    commands = [command for env in factory.built for command in env.commands]
    assert commands
    for joints, grippers in commands:
        np.testing.assert_allclose(joints, np.stack([VALID_JOINTS, VALID_JOINTS]), atol=1e-4)
        np.testing.assert_array_equal(grippers, [1.0, 1.0])
