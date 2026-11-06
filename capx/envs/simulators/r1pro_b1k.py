"""BEHAVIOR-1K (OmniGibson) R1Pro environment behind the low-level env interface.

``R1ProBehaviourLowLevel`` wraps an ``og.Environment`` for one BEHAVIOR activity:
it loads presampled task instances on reset, scores them with the task's own
termination conditions, and exposes the motion helpers ``R1ProControlApi`` drives.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from capx.envs.base import BaseEnv

import yaml
import omnigibson as og
from omnigibson.macros import gm
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
)
from omnigibson.utils.transform_utils import euler2quat, quat_multiply, quat2mat
from omnigibson.object_states.toggle import ToggledOn
import torch
from omnigibson.utils.python_utils import recursively_convert_to_torch
import csv
import json
try:
    from omnigibson.action_primitives.curobo import (
        holonomic_base_command_in_world_frame,
    )
except ImportError:
    # OmniGibson 3.9 folds the world-to-root base conversion into Robot.q_to_action.
    def holonomic_base_command_in_world_frame(robot, joint_pos):
        return joint_pos
from omnigibson.action_primitives.action_primitive_set_base import (
    ActionPrimitiveError,
)
import random
from omnigibson.metrics import MetricBase, AgentMetric, TaskMetric

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
# Instances that match the installed assets come first: OmniGibson 3.9 ships the
# 2026 set, 3.7 the 2025 one. Human demo stats only exist in the 2025 set.
TASK_INSTANCE_DATASETS = ("2026-challenge-task-instances", "2025-challenge-task-instances")
HUMAN_STATS_DATASET = "2025-challenge-task-instances"


def _b50_task_index(task_name: str) -> int:
    path = os.path.join(gm.DATA_PATH, HUMAN_STATS_DATASET, "metadata", "B50_task_misc.csv")
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["Task"] == task_name:
                return int(row["Task ID"])
    raise KeyError(f"{task_name} is not a B50 task ({path})")


def _task_instance_dir(scene_model: str, activity_name: str) -> str:
    for dataset in TASK_INSTANCE_DATASETS:
        path = os.path.join(
            gm.DATA_PATH, dataset, "scenes", scene_model, "json",
            f"{scene_model}_task_{activity_name}_instances",
        )
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"no presampled instances for {activity_name} in {scene_model} under {gm.DATA_PATH}"
    )


def _robot_model(robot) -> str:
    # OmniGibson 3.9 replaced the per-robot classes' model_name with Robot.model.
    return getattr(robot, "model_name", None) or robot.model


def _is_system(inst: str, entity) -> bool:
    if hasattr(entity, "is_system"):
        return entity.is_system
    from omnigibson.utils.bddl_utils import is_system_bddl_inst

    return is_system_bddl_inst(inst)


# OmniGibson 3.9 renamed the metric hooks: start/step/end_callback + gather_results
# became reset/step/aggregate.
def _metric_start(metric: MetricBase, env) -> None:
    if hasattr(metric, "start_callback"):
        metric.start_callback(env)
    else:
        metric.reset(env)


def _metric_step(metric: MetricBase, env, action, obs, reward, terminated, truncated, info) -> None:
    if hasattr(metric, "step_callback"):
        metric.step_callback(env)
    else:
        metric.step(env, action, obs, reward, terminated, truncated, info)


def _metric_results(metric: MetricBase, env) -> dict[str, Any]:
    if hasattr(metric, "end_callback"):
        metric.end_callback(env)
        return metric.gather_results()
    return metric.aggregate(env)


# Make sure object states are enabled
gm.ENABLE_OBJECT_STATES = True
gm.USE_GPU_DYNAMICS = False
gm.HEADLESS = True


def get_grasp_poses_for_object_sticky(target_obj, object_obb):
    """
    Obtain a grasp pose for an object from top down, to be used with sticky grasping.
    The grasp pose should be in the world frame.

    Args:
        target_obj (StatefulObject): Object to get a grasp pose for

    Returns:
        list: List of grasp poses.
    """

    # aabb_min_world, aabb_max_world = target_obj.aabb
    # bbox_center_world = (aabb_min_world + aabb_max_world) / 2
    # bbox_extent_world = aabb_max_world - aabb_min_world
    bbox_center_world = torch.tensor(object_obb.center)
    bbox_extent_world = torch.tensor(object_obb.extent)

    grasp_center_pos = bbox_center_world + torch.tensor([0, 0, bbox_extent_world[2] / 2])
    towards_object_in_world_frame = bbox_center_world - grasp_center_pos
    towards_object_in_world_frame /= torch.norm(towards_object_in_world_frame)

    # Identity quaternion for top-down grasping (x-forward, y-right, z-down)
    grasp_quat = euler2quat(torch.tensor([0, 0, 0], dtype=torch.float32))

    grasp_pose = (grasp_center_pos, grasp_quat)
    grasp_poses = [grasp_pose]

    return grasp_poses


class R1ProBehaviourLowLevel(BaseEnv):
    """ 
    Behavior-1K low-level environment.
    """

    def __init__(
        self,
        controller_cfg: str = "r1pro_primitives.yaml",
        privileged: bool = False,
        activity_name: str | None = None,
        motion_planner: bool = True,
        *args, **kwargs
    ) -> None:
        super().__init__()
        config_filename = os.path.join(CONFIG_DIR, controller_cfg)
        if not os.path.isfile(config_filename):
            config_filename = os.path.join(og.example_config_path, controller_cfg)
        self.controller_cfg = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
        if activity_name is not None:
            self.controller_cfg['task']['activity_name'] = activity_name
            # Load all rooms so any task can find its required objects
            self.controller_cfg['scene']['load_room_types'] = None
        self.task_name = self.controller_cfg['task']['activity_name']

        self._sim_step_count = 0
        # The task's own success flag from the last env.step. Reading it is the only
        # way to score a BEHAVIOR task without stepping the sim again.
        self._last_success = False

        # Load config file
        task_name = self.controller_cfg["task"]['activity_name']
        # Now, get human stats of the task
        task_idx = _b50_task_index(task_name)
        self.human_stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        with open(os.path.join(gm.DATA_PATH, HUMAN_STATS_DATASET, "metadata", "episodes.jsonl"), "r") as f:
            episodes = [json.loads(line) for line in f]
        for episode in episodes:
            if episode["episode_index"] // 1e4 == task_idx:
                for k in self.human_stats.keys():
                    self.human_stats[k].append(episode[k])
        # take a mean
        for k in self.human_stats.keys():
            self.human_stats[k] = sum(self.human_stats[k]) / len(self.human_stats[k])
        
        self.env = og.Environment(configs=self.controller_cfg)
        
        self.robot = self.env.robots[0]
        self.task_relevant_obj = self.env.task.object_scope #keys and values of task relevant objects
        self.object_state_key = self.env.task.low_dim_obs_keys # object state values are in state['task']['low_dim']
        print("Initialized Behavior1KLowLevel environment for task: ", self.task_name)
        
        self.metrics = self.load_metrics()
        
        
        # Without a motion planner the primitives still answer grasp queries
        # (_get_obj_in_hand), which is all a gateway backend needs: planning runs in
        # the submission there.
        self.controller = StarterSemanticActionPrimitives(
            self.env,
            self.env.robots[0],
            enable_head_tracking=False,
            **({} if motion_planner else {"skip_curobo_initilization": True}),
        )
        # torso_joint = self.controller.robot.joints["torso_joint2"]
        self.torso_idx_map = {
            "torso_joint1": 6,
            "torso_joint2": 7,
            "torso_joint3": 8,
            "torso_joint4": 9,
        }
        # self.torso_joint_limits = [ torso_joint.lower_limit, torso_joint.upper_limit]

        self._frame_buffer: list[np.ndarray] = []
        self._record_frames = False

    def load_metrics(self) -> list[MetricBase]:
        """
        Load agent and task metrics.
        """
        return [AgentMetric(self.human_stats), TaskMetric(self.human_stats)]
    
    def load_task_instance(self, instance_id: int, test_hidden: bool = False) -> None:
        """
        Loads the configuration for a specific task instance.

        Args:
            instance_id (int): The ID of the task instance to load.
            test_hidden (bool): [Interal use only] Whether to load the hidden test instance.
        """
        self.env.reset()
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        if test_hidden:
            tro_file_path = os.path.join(
                gm.DATA_PATH,
                "2025-challenge-test-instances",
                self.env.task.activity_name,
                f"{tro_filename}-tro_state.json",
            )
        else:
            tro_file_path = os.path.join(
                _task_instance_dir(scene_model, self.env.task.activity_name),
                f"{tro_filename}-tro_state.json",
            )
            
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, tro_state in tro_state.items():
            if tro_key == "robot_poses":
                presampled_robot_poses = {key.lower(): value for key, value in tro_state.items()}
                robot_pose = presampled_robot_poses[_robot_model(self.robot).lower()][0]
                self.robot.set_position_orientation(robot_pose["position"], robot_pose["orientation"])
                # Write robot poses to scene metadata
                self.env.scene.write_task_metadata(key=tro_key, data=tro_state)
            else:
                self.env.task.object_scope[tro_key].load_state(tro_state, serialized=False)

        # Try to ensure that all task-relevant objects are stable
        # They should already be stable from the sampled instance, but there is some issue where loading the state
        # causes some jitter (maybe for small mass / thin objects?)
        if hasattr(og.sim, "update_handles"):
            og.sim.update_handles()
        for _ in range(25):
            og.sim.step_physics()
            for inst, entity in self.env.task.object_scope.items():
                if entity is not None and not _is_system(inst, entity) and getattr(entity, "exists", True):
                    entity.keep_still()

        self.env.scene.update_initial_file()
        self.env.scene.reset()
        for _ in range(5):
            og.sim.render()
        self.env.reset()
    
        
    def reset(
        self,
        seed: int = None,
        options: dict[str, Any] = None,
        *args, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._sim_step_count = 0
        self._last_success = False

        for arm_name in self.robot.gripper_control_idx.keys():
            grpiper_control_idx = self.robot.gripper_control_idx[arm_name]
            self.robot.set_joint_positions(torch.ones_like(grpiper_control_idx), indices=grpiper_control_idx, normalized=True)
        self.robot.keep_still()

        for _ in range(5):
            og.sim.step()

        self.env.reset()

        if options is not None and 'trial' in options:
            self.load_task_instance(int(options['trial']))
        elif seed is not None:
            self.load_task_instance(int(seed))
        else:
            self.load_task_instance(0)

        # Success baselines come from the instance just loaded, not the scene before it.
        if self.task_name == "turning_on_radio":
            self.initial_radio_pose = self.get_object_state("radio")[0].clone()
        if 'trash' in self.task_name:
            self.initial_trash_pose = self.get_object_state("can__of__soda.n.01_3")[0].clone()
        obs = self.get_observation()
        info = {
            "task_prompt": f"Complete the task: {self.task_name}"
        }

        for metric in self.metrics:
            _metric_start(metric, self.env)

        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Low-level step - not typically called directly in code execution mode."""
        self._sim_step_count += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_success = bool(info['done']['success'])
        self._record_frame()
        
        for metric in self.metrics:
            _metric_step(metric, self.env, action, obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
    
    
    def render(self, mode: str = "rgb_array") -> np.ndarray:  # type: ignore[override]
        if mode != "rgb_array":
            raise ValueError("Only rgb_array render mode is supported")
        obs = self.get_observation()
        robot_key = [key for key in obs.keys() if 'robot' in key][0]
        frame = obs[f'{robot_key}'][f'{robot_key}:zed_link:Camera:0']['rgb'][:,:,:3]
        return frame.cpu().numpy()
    
    def pick_up_radio_reward(self) -> float:
        """
        Compute the reward for picking up the radio.
        """
        table_obj = self.env.scene.object_registry("name", "coffee_table_koagbh_0")
        table_pose, table_quat = table_obj.get_position_orientation()
        table_height = table_pose[-1]
        grasp_obj = self.env.scene.object_registry("name", "radio_89")
        radio_pos, radio_quat = grasp_obj.get_position_orientation()
        if self.controller._get_obj_in_hand() == grasp_obj and radio_pos[2] > self.initial_radio_pose[2] + 0.005:
            reward = 1
        else:
            reward = 0
        
        return reward
    
    def pick_up_trash_reward(self) -> float:
        """
        Compute the reward for picking up the trash.
        """
        trash_obj = self.env.scene.object_registry("name", "can_of_soda_113")
        trash_pose, trash_quat = trash_obj.get_position_orientation()
        if self.controller._get_obj_in_hand() == trash_obj and trash_pose[2] > self.initial_trash_pose[2] + 0.005:
            reward = 1
        else:
            reward = 0
        
        return reward
    
    def compute_reward(self) -> float:
        """
        Compute the reward for the current state.
        Args:
            None.
        Returns:
            Reward for the current state.
        """
        if self.task_name == "turning_on_radio":
            return self.pick_up_radio_reward()
        if 'trash' in self.task_name:
            return self.pick_up_trash_reward()

        metrics = {}
        for metric in self.metrics:
            metrics.update(_metric_results(metric, self.env))
            
        if self.task_completed():
            return 1
        else:
            return metrics['q_score']['final']

    def get_object_state(self, object_name: str) -> dict[str, Any]:
        """
        Get the state of an object in the environment.
        Args:
            object_name: Name of the object to get the state of. Must be in self.task_relevant_obj.
        Returns:
            State of the object.
        """
        object_instance = None
        for key, value in self.task_relevant_obj.items():
            if object_name in key:
                object_instance = value
                break
        if object_instance is None:
            raise ValueError(f"Object {object_name} is not in the task relevant objects.")
        object_state = object_instance.get_position_orientation()
        return object_state
    
    def get_observation(self) -> dict[str, Any]:
        """
        Get the observation of the environment.
        Returns:
            Observation of the environment.
            obs:
                - task:
                    - low_dim: A list of object state with the keys in self.object_state_key.
                - robot_wjqjxx:
                    - robot_wjqjxx:left_realsense_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - robot_wjqjxx:right_realsense_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - robot_wjqjxx:zed_link:Camera:0:
                        - rgb: RGB image of the camera.
                        - depth: Depth image of the camera.
                    - proprio: 68

        """
        obs, info = self.env.get_obs()
        return obs
    
    
    def update_torso(self, torso_joint_name: str, angle: float, max_steps=50):
        torso_joint = self.controller.robot.joints[torso_joint_name]
        torso_idx = self.torso_idx_map[torso_joint_name]
        
        # current_torso_joint_goal = angle
        
        # current_torso_joint_goal = torch.clamp(current_torso_joint_goal, self.torso_joint_limits[0], self.torso_joint_limits[1])
        target_joint_positions = self.robot.get_joint_positions().clone()
        target_joint_positions[torso_idx] = angle
        success = self._move_to_joint_positions(target_joint_positions, max_steps)
        return success

    
    def _navigate_to_pose(self, pose_2d):
        """
        Navigate to a pose in the environment.
        Args:
            pose: Pose to navigate to, xy and yaw in world frame.
        Returns:
            success: Whether the pose was navigated to successfully.
        """
        try:
            ctrl_gen = self.controller._navigate_to_pose(pose_2d) #, ignore_all_obstacles=True, skip_obstacle_update=True)
            for a in ctrl_gen:
                _ = self.step(a)
        except TimeoutError:
            raise
        except Exception as e:
            print("Navigate to pose failed:", e)
            return False
        return True

    
    def _navigate_to_obj(self, obj_name: str):
        #self.task_relevant_obj["radio_receiver.n.01_1"].unwrapped.name
        """
        Navigate to an object in the environment.
        Args:
            obj_name: Name of the object to navigate to. Must be in self.task_relevant_obj.
        Returns:
            None.
        """
        grasp_obj = self.env.scene.object_registry("name", obj_name)
        for action in self.controller._navigate_to_obj(grasp_obj):
            self.step(action)
        return self.get_observation()
    
    
    def _open_close_gripper(self, arm=0, max_steps=250, open=True):
        """
        Open the gripper in the environment.
        Args:
            None.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        target_joint_positions = self.controller._get_joint_position_with_fingers_at_limit("upper" if open else "lower")
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(max_steps):
            self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
            
    def _execute_motion_plan(self, q_traj):
        """
        Execute a motion plan in the environment.
        Args:
            q_traj: Joint trajectory to execute.
        Returns:
            None.
        """
        for a in self.controller._execute_motion_plan(q_traj):
            self.step(a)
            
    def _settle_robot(self):
        """
        Settle the robot in the environment.
        Args:
            None.
        Returns:
            None.
        """
        for a in self.controller._settle_robot():
            self.step(a)
            
    def _execute_grasp(self, grasp_pose: np.ndarray, obj_name: str, arm=0):
        """
        Execute the grasp in the environment.
        Args:
            grasp_pose: Grasp pose to execute.
        Returns:
            None.
        """
        # Step a few times to update
        for a in self.controller._settle_robot():
            self.step(a)

        print("Checking grasp")
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            print("Grasp completed, but no object detected in hand after executing grasp")
            grasped = False
            return grasped

        print("Done with grasp")

        # if self.controller._get_obj_in_hand() != obj:
        #     print("An unexpected object was detected in hand after executing grasp. Consider releasing it")
        #     return reached, grasped
        
        grasped = True
        return grasped

    def check_object_in_hand(self, arm=0):
        """
        Execute the grasp in the environment without reaching.
        Args:
            obj_name: Name of the object to grasp.
        Returns:
            success: Whether the grasp was successful.
        """
        self._settle_robot()
        self.controller.overwrite_arm(arm)
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            print("Grasp completed, but no object detected in hand after executing grasp")
            return False

        print("Done with grasp")
        return True

    
    def _sample_grasp_pose(self, obj_name: str, object_obb) -> tuple[np.ndarray, np.ndarray]:
        """Sample a grasp pose for an object in the environment.
        """
        obj = self.env.scene.object_registry("name", obj_name)
        grasp_poses = get_grasp_poses_for_object_sticky(obj, object_obb)
        grasp_pos, grasp_quat = random.choice(grasp_poses)

        # Identity quaternion for top-down grasping (x-forward, y-right, z-down)
        approach_dir = quat2mat(grasp_quat) @ torch.tensor([0.0, 0.0, -1.0])

        avg_finger_offset = torch.mean(
            torch.tensor([length for length in self.controller.robot.eef_to_fingertip_lengths[self.controller.arm].values()])
        )
        pregrasp_offset = avg_finger_offset + 0.25

        pregrasp_pos = grasp_pos - approach_dir * pregrasp_offset

        # The sampled grasp pose is robot-agnostic
        # We need to multiply by the quaternion of the robot's eef frame of top-down grasping (x-forward, y-right, z-down)
        grasp_quat = quat_multiply(grasp_quat, torch.tensor([1.0, 0.0, 0.0, 0.0]))

        pregrasp_pose = (pregrasp_pos, grasp_quat)
        grasp_pose = (grasp_pos, grasp_quat)

        return pregrasp_pose, grasp_pose

    def _grasp_obj(self, obj_name: str, arm=0):
        obj = self.env.scene.object_registry("name", obj_name)
        pregrasp_pose, grasp_pose = self.controller._sample_grasp_pose(obj)

        self._open_close_gripper(arm=arm, max_steps=250, open=True)
        print("Moving hand to grasp pose")
        for a in self.controller._move_hand(pregrasp_pose):
            _ = self.step(a)

        if self.robot.grasping_mode == "sticky":
            print("Sticky grasping: close gripper")
            # Close the gripper

            print("Sticky grasping: approach")
            # Only translate in the z-axis of the goal frame (assuming z-axis points out of the gripper)
            # This is the same as requesting the end-effector to move along the approach_dir direction.
            # By default, it's NOT the z-axis of the world frame unless `project_pose_to_goal_frame=False` is set in curobo.
            # For sticky grasping, we also need to ignore the object during motion planning because the fingers are already closed.
            # ctrl = self.controller._move_hand(grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0], stop_on_ag=True, ignore_objects=[obj])
            ctrl = self.controller._move_hand(grasp_pose, stop_on_ag=True, ignore_objects=[obj])
            for a in ctrl:  _  = self.step(a)
            
        elif self.robot.grasping_mode == "assisted":
            print("Assisted grasping: approach")
            # Same as above in terms of moving along the approach_dir direction, but we don't ignore the object.
            # For this approach motion, we expect the fingers to move towards and eventually "wrap" around the object without collisions.
            for a in self.controller._move_hand(grasp_pose, motion_constraint=[1, 1, 1, 1, 1, 0]): _ = self.step(a)
            # for a in self.controller._move_hand(grasp_pose, ignore_objects=[obj]): _ = self.step(a)

            # Now we close the fingers to grasp the object with AG.
            print("Assisted grasping: close gripper")
            self._open_close_gripper(arm=arm, max_steps=250, open=False)

        # Step a few times to update
        for a in self.controller._settle_robot():
            self.step(a)

        print("Checking grasp")
        obj_in_hand = self.controller._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "Grasp completed, but no object detected in hand after executing grasp",
                {"target object": obj.name},
            )

        print("Done with grasp")

        if self.controller._get_obj_in_hand() != obj:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                "An unexpected object was detected in hand after executing grasp. Consider releasing it",
                {"expected object": obj.name, "actual object": self._get_obj_in_hand().name},
            )
        
        if self.controller._get_obj_in_hand() == obj:
            return True
        else:
            return False
    
    def _release_object(self, arm=0):
        """
        Release the object in the environment in the default arm.
        Args:
            None.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        for action in self.controller._execute_release():
            self.step(action)
        return self.get_observation()
    
    def _move_hand_direct_ik(self, target_pose, arm=0):
        """
        Move the hand to a target pose in the environment using direct inverse kinematics.
        Args:
            target_pose: Target pose to move the hand to.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        try:
            ctrl_gen = self.controller._move_hand_direct_ik(target_pose)
            for a in ctrl_gen:
                _ = self.step(a)
            success = True
        except TimeoutError:
            raise
        except Exception as e:
            print("Move hand failed:", e)
            success = False
        return success

    
    def _move_hand(self, target_pose, arm=0, ignore_objects=None, motion_constraint=None, attached_obj_scale=None, ignore_all_obstacles=False, ik_only=False, ik_world_collision_check=True, skip_obstacle_update=False):
        """
        Move the hand to a target pose in the environment.
        Args:
            target_pose: Target pose to move the hand to.
        Returns:
            None.
        """
        self.controller.overwrite_arm(arm)
        try:
            ctrl_gen = self.controller._move_hand(target_pose, ignore_objects=ignore_objects, motion_constraint=motion_constraint, attached_obj_scale=attached_obj_scale, ignore_all_obstacles=ignore_all_obstacles, ik_only=ik_only, ik_world_collision_check=ik_world_collision_check, skip_obstacle_update=skip_obstacle_update)
            for a in ctrl_gen:
                _ = self.step(a)
            success = True
        except TimeoutError:
            raise
        except Exception as e:
            print("Move hand failed:", e)
            success = False
        return success
    
    def _move_hand_upward(self, arm=0):
        """
        Move the hand upward in the environment.
        Args:
            None.
        Returns:
            None.
        """
        cur_joint_positions = self.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        if arm == 0:
            target_joint_positions[10] -= 0.4
        elif arm == 1:
            target_joint_positions[11] -= 0.4
            
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(20):
            _ = self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
    
    def _move_to_joint_positions(self, target_joint_positions, max_steps=20, settle_steps=10):
        """
        Move the robot to a target joint positions in the environment.
        Args:
            target_joint_positions: Target joint positions to move the robot to.
        Returns:
            success: Whether the joint positions were reached successfully.
        """
        # cur_joint_positions = self.get_joint_positions()
        # delta = 0.2
        # N = int(abs(target_joint_positions - cur_joint_positions).max()/delta)
        # print("N:", N)
        # N = min(N, 10)
        # delta = abs(target_joint_positions - cur_joint_positions).max()/N
        # for idx in range(N):
        #     cur_joint_positions = self.get_joint_positions()
        #     cur_target = cur_joint_positions + (target_joint_positions - cur_joint_positions).clamp(-delta, delta)
        #     action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, cur_target))
        #     for j in range(10):
        #         _ = self.step(action)
        #         current_joint_positions = self.get_joint_positions()
        #         if torch.allclose(current_joint_positions, cur_target, atol=0.005):
        #             break
        
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for idx in range(max_steps):
            _ = self.step(action)
            if idx % int(settle_steps) == 0:
                self._settle_robot()
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
            
        # current_joint_positions = self.get_joint_positions()
        if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
            return True
        else:
            return False
    
    def _lift_arm(self, arm=0):
        """
        Lift the arm in the environment.
        Args:
            None.
        Returns:
            None.
        """
        cur_joint_positions = self.get_joint_positions()
        target_joint_positions = cur_joint_positions.clone()
        if arm == 0:
            target_joint_positions[10] -= 0.4
        elif arm == 1:
            target_joint_positions[11] -= 0.4
            
        action = self.robot.q_to_action(holonomic_base_command_in_world_frame(self.robot, target_joint_positions))
        for _ in range(20):
            _ = self.step(action)
            current_joint_positions = self.robot.get_joint_positions()
            if torch.allclose(current_joint_positions, target_joint_positions, atol=0.005):
                break
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get the joint positions of the robot.
        Returns:
            env.robot.joints.keys()
            dict_keys(['base_footprint_x_joint', 'base_footprint_y_joint', 'base_footprint_z_joint', 'base_footprint_rx_joint', 'base_footprint_ry_joint', 'base_footprint_rz_joint', 'torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1', 'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4', 'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6', 'left_arm_joint7', 'right_arm_joint7', 'left_gripper_finger_joint1', 'left_gripper_finger_joint2', 'right_gripper_finger_joint1', 'right_gripper_finger_joint2'])
        """
        return self.env.robots[0].get_joint_positions()
    
    
    def get_robot_eef_pose(self, arm=0) -> dict[str, Any]:
        """
        Get the end-effector pose of the robot.
        Args:
            arm: Arm to get the end-effector pose of.
        Returns:
            End-effector pose of the robot.
        """
        arm = self.robot.arm_names[arm]
        return self.robot.get_eef_pose(arm=arm)
    
    def get_robot_base_pose(self) -> dict[str, Any]:
        """
        Get the base pose of the robot. A list of 3 and 4
        Returns:
            Base pose of the robot.
        """
        return self.robot.get_position_orientation()

    def get_toggle_button_geom(self, obj) -> dict[str, Any]:
        """
        Get the toggle button of an object in the environment.
        Args:
            obj: Object to get the toggle button of.
        Returns:
            Toggle button object.
        """
        return obj.states[ToggledOn].visual_marker
    
    def task_completed(self) -> bool:
        """Whether the task is done, as of the last step. Never advances the sim."""
        if self.task_name == "turning_on_radio":
            return self.pick_up_radio_reward() == 1
        if 'trash' in self.task_name:
            return self.pick_up_trash_reward() == 1
        return self._last_success
    
    def enable_video_capture(self, enabled: bool = True, *, clear: bool = True) -> None:
        self._record_frames = enabled
        if clear:
            self._frame_buffer.clear()
        if enabled:
            self._record_frame()

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        if len(self._frame_buffer) == 0:
            self._record_frame()
        keys = list(self._frame_buffer[0].keys())
        frames = {key: [frame[key].copy() for frame in self._frame_buffer] for key in keys}
        if clear:
            self._frame_buffer.clear()
        return frames
    
    def _record_frame(self) -> None:
        if not self._record_frames:
            return
        obs = self.get_observation()
        obs_keys = list(obs.keys())
        robot_key = [key for key in obs_keys if 'robot' in key][0]
        frame ={
            'rgb': self.env.external_sensors['external_camera'].get_obs()[0]['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "ego": obs[f'{robot_key}'][f'{robot_key}:zed_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "left_wrist": obs[f'{robot_key}'][f'{robot_key}:left_realsense_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy(),
            "right_wrist": obs[f'{robot_key}'][f'{robot_key}:right_realsense_link:Camera:0']['rgb'][:,:,:3].detach().cpu().numpy().copy()
        }
        self._frame_buffer.append(frame)  # Flip vertically
    
__all__ = ["R1ProBehaviourLowLevel"]
