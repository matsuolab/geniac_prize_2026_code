"""Portable startup helpers; importing this must not import numpy, gym or Isaac."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


def configure_cpu_threads() -> int:
    """Run before numpy/torch/Isaac imports; one backend is one lane."""
    count = int(os.environ.get("CAPX_LANE_CPUS", "1"))
    if count < 1:
        raise ValueError("CAPX_LANE_CPUS must be positive")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "PXR_WORK_THREAD_LIMIT"):
        os.environ[name] = str(count)
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
    os.environ["MAX_JOBS"] = str(count)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/cache/torch_inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", "/cache/triton")
    os.environ.setdefault("CUDA_CACHE_PATH", "/cache/cuda")
    return count


def configure_isaac_threads() -> int:
    # SimulationApp otherwise overwrites PXR/OPENBLAS with its own default of
    # 32 and creates 32 Carb/TBB workers in EVERY lane, regardless of VM quota.
    from isaacsim import SimulationApp
    count = max(4, int(os.environ.get("CAPX_LANE_CPUS", "1")))
    SimulationApp.DEFAULT_LAUNCHER_CONFIG["limit_cpu_threads"] = count
    args = [arg for arg in SimulationApp.DEFAULT_LAUNCHER_CONFIG.get("extra_args", [])
            if not arg.startswith("--/persistent/physics/numThreads=")]
    args.append("--/persistent/physics/numThreads=" + os.environ.get("CAPX_LANE_CPUS", "1"))
    SimulationApp.DEFAULT_LAUNCHER_CONFIG["extra_args"] = args
    return count


def resolve_task_scene(data_root, task, datasets=("2025-challenge-task-instances",)):
    """A task's actual house, never the default can-task house or a None filename."""
    if not task or Path(task).name != task:
        raise ValueError("Invalid activity name")
    for dataset in datasets:
        scenes = Path(data_root) / dataset / "scenes"
        matches = sorted(scenes.glob(f"*/json/*_task_{task}_0_0_template.json"))
        if len(matches) > 1:
            raise ValueError(f"Ambiguous scene for {task}: {matches}")
        if matches:
            path = matches[0]
            return path.parent.parent.name, str(path)
    raise FileNotFoundError(f"No scene template for {task} under {data_root}")


@contextmanager
def initialization_lock():
    """Serialize scene/reset initialization across lanes sharing this owned cache."""
    path = Path(os.environ.get("CAPX_INIT_LOCK", "/cache/behavior-initialization.lock"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_camera_intrinsics(read, render, *, timeout=30.0, attempts=120, sleep=time.sleep):
    """Wait for actual renderer calibration, without inventing a camera matrix."""
    import numpy as np
    deadline = time.monotonic() + timeout
    last_error = None
    for attempt in range(attempts):
        try:
            value = read()
            array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
            if (array.shape == (3, 3) and np.isfinite(array).all()
                    and array[0, 0] > 0 and array[1, 1] > 0
                    and abs(float(np.linalg.det(array))) > 1e-12):
                return value
            last_error = ValueError(f"Renderer returned invalid intrinsic matrix: {array}")
        except AssertionError as exc:
            if "intrinsic matrix" not in str(exc) or "degenerate" not in str(exc):
                raise
            last_error = exc
        if attempt + 1 >= attempts or time.monotonic() >= deadline:
            break
        render()
        sleep(0.01)
    raise RuntimeError("Camera calibration did not become valid within bounded render warmup") from last_error


def install_camera_warmup():
    import omnigibson as og
    from omnigibson.sensors.vision_sensor import VisionSensor
    original = VisionSensor.intrinsic_matrix
    if getattr(original.fget, "_v1_warmup", False):
        return

    def calibrated(sensor):
        return read_camera_intrinsics(lambda: original.fget(sensor), lambda: og.sim.render())

    calibrated._v1_warmup = True
    VisionSensor.intrinsic_matrix = property(calibrated, doc=original.__doc__)
