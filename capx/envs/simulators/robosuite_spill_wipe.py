"""Low-level Robosuite Franka environment compatible with FrankaControlApi.

This module provides a thin wrapper around Robosuite's Wipe environment
that implements the same interface as FrankaPickPlaceLowLevel, making it
hot-swappable for code execution environments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_seeding import reseed


class FrankaRobosuiteSpillWipeLowLevel(RobosuiteBaseEnv):
    """Robosuite Franka Wipe environment with FrankaPickPlaceLowLevel-compatible interface."""

    _SUBSAMPLE_RATE = 10
    _ACTION_SLICE = -2  # Wipe env uses action[:-2] instead of action[:-1]

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 4000,
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
                self.robosuite_env = suite.environments.manipulation.wipe.Wipe(
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
                self.robosuite_env = suite.environments.manipulation.wipe.Wipe(
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
            self.robosuite_env = suite.environments.manipulation.wipe.Wipe(
                robots=["Panda"],
                has_renderer=True,
                has_offscreen_renderer=True,
                camera_names=self.render_camera_names,
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
        # Additional viser attributes specific to spill_wipe
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

        obs = self.get_observation()

        info = {"task_prompt": "Wipe the spill off the table. Quaternions are WXYZ."}
        return obs, info

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.005, max_steps: int = 10
    ) -> None:
        """Move to target joint positions using Robosuite's controller.

        Args:
            joints: (7,) target joint positions in radians
            tolerance: Position tolerance for convergence
            max_steps: Maximum simulation steps to reach target
        """
        # Override with spill_wipe-specific defaults
        super().move_to_joints_blocking(joints, tolerance=tolerance, max_steps=max_steps)

    def compute_reward(self) -> float:
        """Compute dense spill wipe reward."""
        return self.robosuite_env.reward()

    def task_completed(self) -> bool:
        """Compute if the task is completed."""
        return self.robosuite_env._check_success()

    def _gripper_opening(self, robosuite_obs: dict[str, Any]) -> float:
        # Wipe mounts a WipingGripper, which has no finger joints: robot0_gripper_qpos is
        # empty. Report it as always open so proprio keeps the (8,) shape of the other arms.
        return 1.0

    def get_observation(self) -> dict[str, Any]:
        """Get observation in FrankaPickPlaceLowLevel format."""
        robosuite_obs = self.robosuite_env._get_observations(force_update=True)
        self._process_camera_observations(robosuite_obs)
        self._compute_gripper_obs(robosuite_obs)
        return robosuite_obs


__all__ = ["FrankaRobosuiteSpillWipeLowLevel"]
