"""Two-arm tasks against the real Robosuite sim, through the gateway.

Everything else about the bimanual path is covered by fakes; what only a real sim
can answer is whether the adapter's control primitive is what it claims to be.
There is no ``_current_joints`` target on these envs and their ``_step_once`` holds
position, so the adapter drives them with one iteration of
``move_to_joints_blocking_both``. This checks that this really is *one* cycle
toward the commanded targets, and that both arms are actually reachable.

Needs the Robosuite stack and a GPU:

    sbatch -p part-$(id -gn) --gres=gpu:1 \
        CAPX_TEST_IMAGE=singularity/images/capx-env-robosuite.sif \
        scripts/evalq/test-in-image.sbatch tests/integration/test_bimanual_gateway.py
"""

from __future__ import annotations

import numpy as np
import pytest

robosuite = pytest.importorskip("robosuite")

from capx.gateway import EnforcementMode, Gateway  # noqa: E402
from capx.gateway.robosuite_adapter import RobosuiteEnvFactory  # noqa: E402
from capx.protocol import (  # noqa: E402
    BimanualFrankaRawAction,
    EnvClient,
    FullObs,
    InProcessTransport,
)
from capx.protocol.taskpack import SeedPolicy, TaskSpec  # noqa: E402

CAMERA = "agentview"


def two_arm_task(task_id: str, env_target: str) -> dict[str, TaskSpec]:
    return {
        task_id: TaskSpec(
            task_id=task_id,
            embodiment="franka_bimanual",
            simulator="robosuite",
            instruction="coordinate both arms",
            camera_names=[CAMERA],
            env_target=env_target,
            n_trials=2,
            trial_steplimit=5000,
            seed_policy=SeedPolicy(mode="fixed", seeds=[0, 1]),
        )
    }


@pytest.fixture(scope="module", params=[
    ("two_arm_lift", "two_arm_lift_robosuite"),
    ("two_arm_handover", "two_arm_handover_robosuite"),
])
def episode(request):
    task_id, env_target = request.param
    gateway = Gateway(
        two_arm_task(task_id, env_target), RobosuiteEnvFactory(), EnforcementMode.DEV
    )
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset(task_id)
    yield client, gateway, reset
    gateway.close()


def test_reset_reports_both_arms(episode) -> None:
    _, _, reset = episode
    assert isinstance(reset.obs, FullObs)
    arms = reset.obs.proprio.arms
    assert len(arms) == 2
    for arm in arms:
        assert arm.joint_pos.shape == (8,)
        assert arm.cartesian_pos.shape == (8,)
        assert arm.base_pose.shape == (7,)
    # Two arms on opposite sides of a table do not share a base frame.
    assert not np.allclose(arms[0].base_pose[4:], arms[1].base_pose[4:])


def test_the_privileged_object_pose_never_crosses(episode) -> None:
    client, _, reset = episode
    obs = client.get_observation(reset.episode_id)
    assert set(obs.cameras) == {CAMERA}
    assert not hasattr(obs.proprio, "pot_poses")
    assert not hasattr(obs.proprio, "hammer_pos")


def test_one_step_is_one_control_cycle(episode) -> None:
    """The whole basis of the adapter: max_steps=1 with tolerance=0.0 drives the
    arms toward the target for exactly one cycle rather than servoing to it."""
    client, _, reset = episode
    before = reset.obs.proprio.sim_time
    joints = np.stack([reset.obs.proprio.arms[i].joint_pos[:7] for i in range(2)])

    first = client.step(
        reset.episode_id, BimanualFrankaRawAction(joints=joints, grippers=np.ones(2))
    )
    one_cycle = first.obs.sim_time - before

    second = client.step(
        reset.episode_id,
        BimanualFrankaRawAction(joints=joints, grippers=np.ones(2), repeat=3),
    )
    three_cycles = second.obs.sim_time - first.obs.sim_time

    assert one_cycle > 0.0
    assert three_cycles == pytest.approx(3 * one_cycle, rel=1e-6)


def test_a_commanded_arm_moves_and_the_other_holds(episode) -> None:
    client, _, reset = episode
    start = np.stack([reset.obs.proprio.arms[i].joint_pos[:7] for i in range(2)])

    target = start.copy()
    target[0] = target[0] + 0.05  # well inside the Panda limits from any valid pose
    result = client.step(
        reset.episode_id,
        BimanualFrankaRawAction(joints=target, grippers=np.ones(2), repeat=25),
    )

    moved = np.stack([result.obs.arms[i].joint_pos[:7] for i in range(2)])
    assert np.linalg.norm(moved[0] - start[0]) > 1e-3, "the commanded arm did not move"
    assert np.linalg.norm(moved[1] - start[1]) < 5e-2, "the held arm drifted"
