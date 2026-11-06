"""Oracle smoke tests: each task env must reach reward 1.0 with its own oracle code.

They build real simulators with rendering on (so a GPU with EGL), and each skips
unless its simulator and the task-env stack are installed. Run one by hand,
optionally saving a video, with::

    uv run tests/test_environments.py --env-name franka_lift_code_env --video-path out.mp4
"""

from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


def run_oracle(env_name: str = "franka_pick_place_code_env", video_path: str | None = None) -> bool:
    """Reset ``env_name``, run its oracle code once, and report whether it succeeded."""
    from capx.envs.base import list_envs
    from capx.envs.tasks import get_config, get_exec_env, list_configs, list_exec_envs
    from capx.integrations.base_api import list_apis

    print("Available environments: ", list_envs())
    print("Available execution environments: ", list_exec_envs())
    print("Available APIs: ", list_apis())
    print("Available configurations: ", list_configs())
    env = get_exec_env(env_name)(get_config(env_name))
    if video_path is not None:
        env.enable_video_capture(True)
    start = time.time()
    obs, _ = env.reset()
    print("Time taken to reset: ", time.time() - start)
    print("Observation keys: ", list(obs.keys()))
    print("Prompt: ", obs["full_prompt"][1]["content"])
    start = time.time()
    _, reward, terminated, truncated, info_step = env.step(env.oracle_code)
    print("Time taken: ", time.time() - start)
    if video_path is not None:
        import imageio

        imageio.mimsave(video_path, env.get_video_frames(), fps=30)
        print(f"Video saved to {video_path}")
    if reward != 1.0:
        print("Reward: ", reward)
        print("Terminated: ", terminated)
        print("Truncated: ", truncated)
        print("Info: ", info_step)
        return False
    print("Success")
    return True


def _needs(simulator: str) -> None:
    pytest.importorskip(simulator)
    # The APIs the oracle code calls register from capx.integrations' subpackages.
    pytest.importorskip("capx.integrations.franka.control")
    pytest.importorskip("capx.envs.tasks")


def test_franka_pick_place_code_env() -> None:
    _needs("robosuite")
    assert run_oracle("franka_pick_place_code_env")


def test_franka_robosuite_pick_place_code_env() -> None:
    _needs("robosuite")
    assert run_oracle("franka_robosuite_pick_place_code_env")


def test_franka_lift_code_env() -> None:
    _needs("robosuite")
    assert run_oracle("franka_lift_code_env")


def test_franka_nut_assembly_code_env() -> None:
    _needs("robosuite")
    assert run_oracle("franka_nut_assembly_code_env")


def test_franka_pick_place_multi_code_env() -> None:
    _needs("robosuite")
    assert run_oracle("franka_pick_place_multi_code_env")


def test_r1pro_radio_code_env() -> None:
    _needs("omnigibson")
    assert run_oracle("r1pro_radio_code_env")


if __name__ == "__main__":
    import tyro

    tyro.cli(run_oracle)
