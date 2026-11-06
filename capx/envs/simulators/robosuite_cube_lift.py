"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Lift environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_seeding import reseed


class FrankaRobosuiteCubeLiftLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Cube Lift environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 2

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        super().__init__(
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )

        # Initialize Robosuite environment
        if privileged:
            if not enable_render:
                self.render_camera_names = []
                self.robosuite_env = suite.environments.manipulation.lift.Lift(
                    robots=["Panda"],
                    use_camera_obs=False,
                    has_renderer=False,
                    has_offscreen_renderer=False,
                    camera_names=self.render_camera_names,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                )
            else:
                self.robosuite_env = suite.environments.manipulation.lift.Lift(
                    robots=["Panda"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    camera_names=self.render_camera_names,
                    camera_depths=True,
                    renderer="mujoco",
                    camera_heights=self._render_height,
                    camera_widths=self._render_width,
                    controller_configs=load_composite_controller_config(
                        controller=self.controller_cfg
                    ),
                    horizon=max_steps,
                    reward_shaping=True,
                )
        else:
            self.robosuite_env = suite.environments.manipulation.lift.Lift(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
                # camera_segmentations=self.segmentation_level,
                camera_depths=True,
                renderer="mujoco",
                camera_heights=self._render_height,
                camera_widths=self._render_width,
                controller_configs=load_composite_controller_config(controller=self.controller_cfg),
                horizon=max_steps,
                reward_shaping=True,
            )

        self._init_robot_links()
        self._init_viser_debug(viser_debug)
        # Additional viser attributes specific to cube_lift
        if viser_debug:
            self.grasp_sample = None
            self.grasp_scores = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            reseed(self.robosuite_env, seed)

        self.robosuite_env.reset()
        # Adjust initial orientation
        self.robosuite_env.sim.data.qpos[6] -= np.pi
        self._current_joints = self._joint_positions()

        self._step_count = 0
        self._sim_step_count = 0

        # Settle the environment
        for _ in range(50):
            self.robosuite_env.sim.forward()
            self.robosuite_env.sim.step()
            self._set_gripper(1.0)

        obs = self.get_observation()

        info = {"task_prompt": "Pick up the cube and lift it. Quaternions are WXYZ."}
        return obs, info

    def _cube_pose_dict(self, robosuite_obs: dict[str, Any]) -> dict[str, list[float]]:
        """Get cube poses in robot base frame."""
        base_link_wxyz_xyz = np.concatenate(
            [
                self.robosuite_env.sim.data.xquat[self.base_link_idx],
                self.robosuite_env.sim.data.xpos[self.base_link_idx],
            ]
        )

        cubeA_world = vtf.SE3(
            wxyz_xyz=np.concatenate([robosuite_obs["cube_quat"], robosuite_obs["cube_pos"]])
        )

        base_transform = vtf.SE3(wxyz_xyz=base_link_wxyz_xyz).inverse()
        cubeA_robot_base = base_transform @ cubeA_world

        return {
            "primary": [
                float(x)
                for x in np.concatenate(
                    [cubeA_robot_base.translation(), cubeA_robot_base.rotation().wxyz]
                )
            ],
        }

    def compute_reward(self) -> float:
        """Compute sparse stacking reward."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Check if task is completed."""
        return self.robosuite_env._check_success()

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations(force_update=True)
        pose_dict = self._cube_pose_dict(robosuite_obs)
        cube_pose_array = np.stack(
            [
                np.asarray(pose_dict["primary"], dtype=np.float32),
            ],
            axis=0,
        )

        robosuite_obs["cube_poses"] = {
            "primary": cube_pose_array[0],
        }

        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)

        return robosuite_obs


__all__ = ["FrankaRobosuiteCubeLiftLowLevel"]
