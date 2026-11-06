"""Observation tiers and the allowlist ``ObsFilter``.

- proprio tier (every ``step``): robot self-state and ``sim_time`` only, cheap
  enough to poll every control cycle.
- full tier (``get_observation``): the proprio tier plus RGB-D camera views.

``ObsFilter`` rebuilds each observation from a fixed set of known-safe fields
rather than stripping known-bad ones, because the raw sim dict carries privileged
ground truth (e.g. ``cube_poses``) and a new task may add more. Ground-truth
``segmentation`` is never copied either.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Dataclasses (data plane — carried over framed msgpack, not pydantic)
# ---------------------------------------------------------------------------


@dataclass
class CameraObs:
    """A single camera view. Arrays are plain numpy; the codec handles the wire."""

    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray | None  # (H, W) or (H, W, 1) float32, meters
    intrinsics: np.ndarray  # (3, 3) float
    pose: np.ndarray  # (7,) [x, y, z, qw, qx, qy, qz] camera-in-robot-base
    pose_mat: np.ndarray | None = None  # (4, 4) float, redundant with `pose`


@dataclass
class ArmProprio:
    """One arm's self-state, for embodiments that have more than one.

    ``base_pose`` is that arm's base link in the world frame, which a single-arm
    task never needs: its ``cartesian_pos`` is already expressed in the one base
    frame there is. With two arms the submission has to relate them, so each arm's
    base has to cross the wire. It is robot self-state, not object ground truth.

    ``cartesian_pos`` is in **arm 0's** base frame for every arm, not each arm's
    own — that is the convention Robosuite's two-arm envs already publish and the
    bimanual control APIs are written against.
    """

    joint_pos: np.ndarray  # (8,) 7 joints + gripper
    cartesian_pos: np.ndarray  # (8,) [xyz, qwxyz, gripper] end-effector in arm 0's base
    base_pose: np.ndarray  # (7,) [qwxyz, xyz] of this arm's base link, world frame


@dataclass
class ProprioObs:
    """Robot self-state. No object ground truth ever appears here.

    ``arms`` is empty for single-arm embodiments. For a bimanual one it carries a
    row per arm, and ``joint_pos`` / ``cartesian_pos`` mirror arm 0 so anything
    written against the single-arm shape keeps working.
    """

    joint_pos: np.ndarray  # (Nq,) arm joints [+ gripper as last element]
    cartesian_pos: np.ndarray  # (8,) [xyz, qwxyz, gripper] end-effector in base
    sim_time: float
    arms: list[ArmProprio] = field(default_factory=list)


@dataclass
class FullObs:
    """Full-tier observation: proprio + all allowed camera views."""

    proprio: ProprioObs
    cameras: dict[str, CameraObs] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ObsFilter — the single chokepoint that strips privileged state
# ---------------------------------------------------------------------------


@dataclass
class ObsFilter:
    """Reconstruct a safe observation from a raw low-level env obs dict.

    Configured per task by the gateway (from the task pack), never by the
    submission. ``camera_names`` fixes which views are exposed; anything not
    listed here or in the proprio allowlist is discarded. Ground-truth
    segmentation is never reconstructed, so it cannot cross the wire.

    Attributes:
        camera_names: Cameras to expose (e.g. ``["robot0_robotview"]``).
        include_depth: Whether depth maps are exposed.
    """

    camera_names: list[str]
    include_depth: bool = True

    def to_proprio(self, raw: dict[str, Any], *, sim_time: float) -> ProprioObs:
        """Extract the proprio tier from a raw low-level obs dict.

        ``arms`` is an allowlist of the same kind as the rest of this class: only
        the three self-state fields are reconstructed, so a bimanual adapter cannot
        smuggle anything else through by putting it on an arm.
        """
        joint = np.asarray(raw["robot_joint_pos"], dtype=np.float32)
        cart = np.asarray(raw["robot_cartesian_pos"], dtype=np.float32)
        arms = [
            ArmProprio(
                joint_pos=np.asarray(arm["robot_joint_pos"], dtype=np.float32),
                cartesian_pos=np.asarray(arm["robot_cartesian_pos"], dtype=np.float32),
                base_pose=np.asarray(arm["base_pose"], dtype=np.float32),
            )
            for arm in raw.get("arms", ())
        ]
        return ProprioObs(
            joint_pos=joint, cartesian_pos=cart, sim_time=float(sim_time), arms=arms
        )

    def to_full(self, raw: dict[str, Any], *, sim_time: float) -> FullObs:
        """Extract the full tier (proprio + allowed cameras) from raw obs."""
        cameras: dict[str, CameraObs] = {}
        for name in self.camera_names:
            cam = raw.get(name)
            if not isinstance(cam, dict):
                continue
            images = cam.get("images", {})
            depth = None
            if self.include_depth and "depth" in images:
                depth = np.asarray(images["depth"], dtype=np.float32)
            cameras[name] = CameraObs(
                name=name,
                rgb=np.asarray(images["rgb"], dtype=np.uint8),
                depth=depth,
                intrinsics=np.asarray(cam["intrinsics"], dtype=np.float32),
                pose=np.asarray(cam["pose"], dtype=np.float32),
                pose_mat=(
                    np.asarray(cam["pose_mat"], dtype=np.float32)
                    if "pose_mat" in cam
                    else None
                ),
            )
        return FullObs(proprio=self.to_proprio(raw, sim_time=sim_time), cameras=cameras)


# ---------------------------------------------------------------------------
# Wire codec — plain dicts of numpy, ready for ``capx.protocol.framing``
# ---------------------------------------------------------------------------


def _jpeg_encode(rgb: np.ndarray) -> bytes:
    """JPEG-compress an (H, W, 3) uint8 image for the wire."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="jpeg", quality=90)
    return buf.getvalue()


def _jpeg_decode(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.uint8)


def encode_obs(obs: FullObs | ProprioObs, *, jpeg_rgb: bool = True) -> dict[str, Any]:
    """Serialize an observation to a dict that ``capx.protocol.framing`` can send.

    RGB is JPEG-compressed by default (``jpeg_rgb``); depth stays float.
    """
    if isinstance(obs, ProprioObs):
        encoded: dict[str, Any] = {
            "tier": "proprio",
            "joint_pos": obs.joint_pos,
            "cartesian_pos": obs.cartesian_pos,
            "sim_time": obs.sim_time,
        }
        # Omitted entirely when single-arm, so the single-arm wire is unchanged.
        if obs.arms:
            encoded["arms"] = [
                {
                    "joint_pos": arm.joint_pos,
                    "cartesian_pos": arm.cartesian_pos,
                    "base_pose": arm.base_pose,
                }
                for arm in obs.arms
            ]
        return encoded
    cams: dict[str, Any] = {}
    for name, cam in obs.cameras.items():
        cams[name] = {
            "rgb_jpeg" if jpeg_rgb else "rgb": (
                _jpeg_encode(cam.rgb) if jpeg_rgb else cam.rgb
            ),
            "depth": cam.depth,
            "intrinsics": cam.intrinsics,
            "pose": cam.pose,
            "pose_mat": cam.pose_mat,
        }
    return {
        "tier": "full",
        "proprio": encode_obs(obs.proprio, jpeg_rgb=jpeg_rgb),
        "cameras": cams,
    }


def decode_obs(msg: dict[str, Any]) -> FullObs | ProprioObs:
    """Inverse of :func:`encode_obs`."""
    if msg.get("tier") == "proprio":
        return ProprioObs(
            joint_pos=np.asarray(msg["joint_pos"]),
            cartesian_pos=np.asarray(msg["cartesian_pos"]),
            sim_time=float(msg["sim_time"]),
            arms=[
                ArmProprio(
                    joint_pos=np.asarray(arm["joint_pos"]),
                    cartesian_pos=np.asarray(arm["cartesian_pos"]),
                    base_pose=np.asarray(arm["base_pose"]),
                )
                for arm in msg.get("arms", ())
            ],
        )
    cams: dict[str, CameraObs] = {}
    for name, c in msg["cameras"].items():
        rgb = _jpeg_decode(c["rgb_jpeg"]) if "rgb_jpeg" in c else np.asarray(c["rgb"])
        cams[name] = CameraObs(
            name=name,
            rgb=rgb,
            depth=None if c.get("depth") is None else np.asarray(c["depth"]),
            intrinsics=np.asarray(c["intrinsics"]),
            pose=np.asarray(c["pose"]),
            pose_mat=None if c.get("pose_mat") is None else np.asarray(c["pose_mat"]),
        )
    proprio = decode_obs(msg["proprio"])
    assert isinstance(proprio, ProprioObs)
    return FullObs(proprio=proprio, cameras=cams)


__all__ = [
    "ArmProprio",
    "CameraObs",
    "ProprioObs",
    "FullObs",
    "ObsFilter",
    "encode_obs",
    "decode_obs",
]
