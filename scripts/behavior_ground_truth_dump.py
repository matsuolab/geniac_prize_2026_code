"""Privileged ground-truth dump for the picking_up_trash BEHAVIOR scene.

Runs INSIDE a capx-env-behavior container (own Isaac process — never point it
at the GPU serving a live backend unless there is VRAM headroom):

    docker run --rm --gpus '"device=2"' \
      -e NVIDIA_DRIVER_CAPABILITIES=all \
      -e OMNIGIBSON_DATA_PATH=/data/omnigibson \
      -e OMNIGIBSON_APPDATA_PATH=/data/omnigibson-appdata \
      -v cap-x-geniac_behavior-assets:/data/omnigibson \
      -v cap-x-geniac_behavior-appdata:/data/omnigibson-appdata \
      -v "$PWD/scripts/behavior_ground_truth_dump.py:/workspace/capx/gtdump.py:ro" \
      capx-env-behavior:local capx/third_party/b1k/.venv/bin/python /workspace/capx/gtdump.py

Prints (JSON lines, GT| prefix):
- robot base_footprint world pose, root prim (base_footprint_x) world pose,
  virtual base joint values q28[0:6]
- world pose of every scene object whose name contains "can"
- ego camera world pose; eef_link vs gripper_link world poses (tool offset)
- eef_to_fingertip_lengths (the legacy avg_finger_offset source)
"""

from __future__ import annotations

import json

import numpy as np
import torch


def _j(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _j(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_j(v) for v in value]
    return value


def emit(kind: str, **fields):
    fields["kind"] = kind
    print("GT|" + json.dumps(_j(fields)), flush=True)


def main() -> None:
    import capx.envs.simulators  # noqa: F401 — registers env factories
    from capx.envs.base import get_env

    base = get_env("r1pro_b1k_low_level", privileged=False, enable_render=True)
    base.reset(options={"trial": 0})

    robot = base.robot
    scene = base.env.scene

    base_pos, base_quat = robot.get_position_orientation()
    emit("base_footprint_world", pos=base_pos, quat_xyzw=base_quat)
    try:
        root_pos, root_quat = robot.root_link.get_position_orientation()
        emit("root_link_world", name=robot.root_link.prim_path, pos=root_pos, quat_xyzw=root_quat)
    except Exception as e:  # noqa: BLE001
        emit("root_link_error", error=str(e))

    q = base.get_joint_positions()
    emit("q28", q28=q, base6=q[0:6])

    for obj in scene.objects:
        name = getattr(obj, "name", "")
        if "can" in name.lower() or "trash" in name.lower():
            pos, quat = obj.get_position_orientation()
            try:
                aabb = obj.aabb
                extent = (aabb[1] - aabb[0])
            except Exception:  # noqa: BLE001
                extent = None
            emit("object", name=name, pos=pos, quat_xyzw=quat, aabb_extent=extent)

    robot_key = None
    for key in robot.sensors:
        if "zed_link" in key:
            robot_key = key
            cam_pos, cam_quat = robot.sensors[key].get_position_orientation()
            emit("ego_camera_world", sensor=key, pos=cam_pos, quat_xyzw=cam_quat)
    emit("sensors", keys=list(robot.sensors.keys()), zed=robot_key)

    for arm in robot.arm_names:
        eef_link = robot.eef_links[arm]
        eef_pos, eef_quat = eef_link.get_position_orientation()
        emit("eef_link_world", arm=arm, link=eef_link.prim_path, pos=eef_pos, quat_xyzw=eef_quat)
        for link_name, link in robot.links.items():
            if link_name == f"{arm}_gripper_link":
                gl_pos, gl_quat = link.get_position_orientation()
                emit("gripper_link_world", arm=arm, pos=gl_pos, quat_xyzw=gl_quat)
        rel_pos, rel_quat = robot.get_relative_eef_pose(arm=arm)
        emit("relative_eef_pose", arm=arm, pos=rel_pos, quat_xyzw=rel_quat)

    try:
        emit("eef_to_fingertip_lengths", value=robot.eef_to_fingertip_lengths)
    except Exception as e:  # noqa: BLE001
        emit("eef_to_fingertip_lengths_error", error=str(e))

    emit("done")


if __name__ == "__main__":
    main()
