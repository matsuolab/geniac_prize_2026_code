"""A Robosuite task's initial state is a function of the seed the referee hands it.

The placement samplers draw from robosuite's own RNG, which a wrapper that only
reseeds its own leaves unseeded: a "fixed" seed policy then still gave every env
build a different scene. Needs robosuite, so it runs in the image.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("robosuite")
os.environ.setdefault("MUJOCO_GL", "egl")

from capx.envs.base import get_env  # noqa: E402
import capx.envs.simulators  # noqa: E402,F401

TARGETS = [
    "franka_robosuite_cube_lift_low_level",
    "franka_robosuite_cubes_low_level",
    "franka_robosuite_cubes_restack_low_level",
    "franka_robosuite_nut_assembly_low_level",
    "franka_robosuite_spill_wipe_low_level",
    "two_arm_lift_robosuite",
    "two_arm_handover_robosuite",
]


def _initial_qpos(target: str, seed: int) -> np.ndarray:
    env = get_env.__wrapped__(target, privileged=False, enable_render=False)
    try:
        env.reset(seed=seed)
        return np.array(env.robosuite_env.sim.data.qpos, copy=True)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


@pytest.mark.parametrize("target", TARGETS)
def test_the_same_seed_gives_the_same_scene_across_env_builds(target: str) -> None:
    np.testing.assert_allclose(_initial_qpos(target, 3), _initial_qpos(target, 3))


@pytest.mark.parametrize("target", TARGETS[:3])
def test_a_different_seed_gives_a_different_scene(target: str) -> None:
    assert not np.allclose(_initial_qpos(target, 3), _initial_qpos(target, 4))
