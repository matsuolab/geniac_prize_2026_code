"""Real-backend smoke test for the gateway migration (design §12.2).

Skipped entirely unless ``CAPX_IT=1`` — this needs a Robosuite/MuJoCo venv
(``docker/run-robosuite-smoke.sh`` topology) plus live SAM3 (sam3.sock),
Contact-GraspNet (graspnet.sock), and PyRoKi (pyroki.sock) servers. None of
that is available in the sim-free unit-test venv, and this file must never
import anything from ``capx.gateway.robosuite_adapter``,
``capx.integrations.franka.control``, or the perception/IK client packages at
module scope — only inside the (skipped) test function — so simply collecting
this file never requires the sim stack.

Drives a REAL in-process ``Gateway`` (``RobosuiteEnvFactory`` + the public
pack's ``stack_cubes`` task) through a ``FrankaGatewayShim`` +
``FrankaControlApi`` with a scripted (no-LLM) oracle-style API sequence:
``get_object_pose -> sample_grasp_pose -> goto_pose -> close_gripper ->
goto_pose``. Task success itself is graded gateway-side and is never asserted
here (design §8/§9.3) — this only asserts the client-side plumbing (obs
shapes, primitive completion) works end-to-end with zero protocol errors.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CAPX_IT") != "1",
    reason="real-backend smoke test; set CAPX_IT=1 in a robosuite venv with "
    "SAM3/GraspNet/PyRoKi servers on their sockets in $CAPX_SOCKET_DIR",
)

TASKPACK = "taskpacks/original"
TASK_ID = "stack_cubes"


def test_franka_control_api_oracle_sequence_over_the_gateway() -> None:
    from capx.gateway import EnforcementMode, Gateway
    from capx.gateway.robosuite_adapter import RobosuiteEnvFactory
    from capx.integrations.franka.control import FrankaControlApi
    from capx.protocol import EnvClient, InProcessTransport
    from capx.protocol.taskpack import load_task_packs
    from capx.submission.shims.franka import FrankaGatewayShim

    tasks = load_task_packs([TASKPACK])
    spec = tasks[TASK_ID]

    with Gateway(tasks, RobosuiteEnvFactory(), EnforcementMode.DEV) as gateway:
        client = EnvClient(InProcessTransport(gateway.acquire()))
        shim = FrankaGatewayShim(client, TASK_ID, camera_names=spec.camera_names)
        obs, info = shim.reset()

        # Obs remap sanity (design §5.1): joint/cartesian proprio + the
        # manifest's single main camera, with depth forced to (H, W, 1).
        assert obs["robot_joint_pos"].shape == (8,)
        assert obs["robot_cartesian_pos"].shape == (8,)
        main_camera = spec.camera_names[0]
        assert main_camera in obs
        assert obs[main_camera]["images"]["rgb"].ndim == 3
        assert obs[main_camera]["images"]["depth"].ndim == 3
        assert obs[main_camera]["images"]["depth"].shape[-1] == 1
        assert isinstance(info["task_prompt"], str) and info["task_prompt"]
        initial_rgb = obs[main_camera]["images"]["rgb"].copy()

        api = FrankaControlApi(shim, use_sam3=True)

        # Scripted oracle-style sequence (mirrors franka_pick_place.py's
        # ORACLE_CODE) -- object names match the task's own "primary cube" /
        # "secondary cube" language (capx/envs/simulators/robosuite_cubes.py),
        # not a hardcoded color, so this stays correct if the scene's cube
        # colors change.
        grasp_pos, grasp_quat = api.sample_grasp_pose("primary cube")
        api.goto_pose(grasp_pos, grasp_quat, z_approach=0.1)
        api.goto_pose(grasp_pos, grasp_quat)
        api.close_gripper()

        lift_pos = grasp_pos + np.array([0.0, 0.0, 0.15])
        api.goto_pose(lift_pos, grasp_quat)

        place_pos, _place_quat, _extent = api.get_object_pose("secondary cube", return_bbox_extent=True)
        place_pos = place_pos.copy()
        place_pos[2] += 0.15
        api.goto_pose(place_pos, grasp_quat, z_approach=0.1)
        api.open_gripper()

        assert shim.episode_over is False
        final_obs = shim.get_observation()
        assert final_obs["robot_joint_pos"].shape == (8,)
        assert not np.array_equal(
            final_obs[main_camera]["images"]["rgb"], initial_rgb
        ), "gateway camera observation stayed stale after robot motion"

        shim.close()
