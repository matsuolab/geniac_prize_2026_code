from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class LiberoHandle:
    env: Any
    suite_name: str
    task_id: int
    task_language: str
    init_states: Any

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        self.env.seed(seed)
        obs = self.env.reset()
        if self.init_states is not None:
            obs = self.env.set_init_state(self.init_states[0])
        return obs, {}

    def step(self, action: list[float]) -> tuple[Any, float, bool, dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        return obs, float(reward), bool(done), info


_VENDORED_INIT_STATES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "third_party", "LIBERO-PRO", "libero", "libero", "init_files",
)


def load_init_states(task_suite: Any, task_id: int) -> Any:
    """A task's init states, from where LIBERO's ``get_task_init_states`` looks.

    They are pickled numpy arrays shipped with the benchmark, so this one trusted file
    is loaded with ``weights_only=False``: torch>=2.6 refuses them by default, and
    LIBERO layouts other than LIBERO-PRO call ``torch.load`` without the argument.
    Falls back to the vendored LIBERO-PRO copy when the configured path is missing.
    """
    import torch
    from libero.utils import get_libero_path  # type: ignore[import-not-found]

    task = task_suite.get_task(task_id)
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    if not os.path.exists(path):
        fallback = os.path.normpath(
            os.path.join(_VENDORED_INIT_STATES, task.problem_folder, task.init_states_file)
        )
        if not os.path.exists(fallback):
            raise FileNotFoundError(f"init states not found at {path} or {fallback}")
        print(f"Found init states file at fallback path: {fallback}")
        path = fallback
    return torch.load(path, weights_only=False)


def _extract_language_from_bddl(bddl_path: str) -> str | None:
    try:
        with open(bddl_path) as f:
            content = f.read()
        match = re.search(r"\(:language\s+(.*?)\)", content, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Warning: Could not extract language from {bddl_path}: {e}")
    return None


def load_libero_task(
    suite_name: str,
    task_id: int,
    cam_w: int = 128,
    cam_h: int = 128,
    controller: str = "OSC_POSE",
    horizon: int = 1000,
    control_freq: int = 20,
    camera_depths: bool = True,
) -> LiberoHandle:
    """Load a LIBERO task using OffScreenRenderEnv.

    Reference: https://github.com/Lifelong-Robot-Learning/LIBERO
    """
    # Prefer vendored third_party/LIBERO if present, then fall back to installed package
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    vendor_root = os.path.normpath(os.path.join(here, "..", "..", "third_party", "LIBERO-PRO"))
    if os.path.isdir(vendor_root) and vendor_root not in sys.path:
        sys.path.append(vendor_root)
    try:
        from libero import benchmark  # type: ignore[import-not-found]
        from libero.envs import OffScreenRenderEnv  # type: ignore[import-not-found]
        from libero.utils import get_libero_path  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "LIBERO not available; add submodule or run `uv sync --extra libero`."
        ) from e

    # setting help=True will print the available benchmarks
    benchmark_dict = benchmark.get_benchmark_dict(help=False)
    task_suite = benchmark_dict[suite_name]()
    task = task_suite.get_task(task_id)

    bddl_file_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )

    if not os.path.exists(bddl_file_path):
        # Fallback: try to locate BDDL files relative to this file
        # This handles cases where get_libero_path returns incorrect relative paths
        # Path: capx/integrations/libero/../../third_party/LIBERO-PRO/libero/libero/bddl_files
        fallback_bddl_root = os.path.abspath(
            os.path.join(here, "..", "..", "third_party", "LIBERO-PRO", "libero", "libero", "bddl_files")
        )
        fallback_path = os.path.join(fallback_bddl_root, task.problem_folder, task.bddl_file)

        if os.path.exists(fallback_path):
            print(f"Found BDDL file at fallback path: {fallback_path}")
            bddl_file_path = fallback_path
        else:
            print(f"Error: BDDL file not found at {bddl_file_path} OR {fallback_path}")

    env_args = {
        "bddl_file_name": bddl_file_path,
        "camera_heights": cam_h,
        "camera_widths": cam_w,
        "controller": controller,
        "horizon": horizon,
        "control_freq": control_freq,
        "camera_depths": camera_depths,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)

    # Try to extract language from BDDL file directly
    task_language = _extract_language_from_bddl(bddl_file_path)
    if not task_language:
        task_language = task.language

    init_states = load_init_states(task_suite, task_id)
    print(f"Loaded init states for task {task_id} in suite {suite_name}")

    handle = LiberoHandle(
        env=env,
        suite_name=suite_name,
        task_id=task_id,
        task_language=task_language,
        init_states=init_states,
    )
    return handle
