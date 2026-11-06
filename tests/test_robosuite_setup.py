"""Smoke test of the Robosuite install itself: build a scene and step it.

No renderer, so it runs on a CPU; skips where Robosuite is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_robosuite_scene_builds_and_steps() -> None:
    suite = pytest.importorskip("robosuite")
    env = suite.make(
        env_name="NutAssemblySquare",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
    )
    try:
        env.reset()
        low, high = env.action_spec
        rng = np.random.default_rng(0)
        for _ in range(10):
            obs, reward, done, _ = env.step(rng.uniform(low, high) * 0.1)
        assert obs["robot0_joint_pos"].shape == (7,)
        assert np.isfinite(reward)
    finally:
        env.close()
