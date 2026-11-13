"""Trial orchestration for CaP-X environments.

Handles batch execution, parallel worker dispatch, retry logic, and
wall-clock timeouts for running code-generation trials.  The actual
single-trial execution lives in :mod:`capx.envs.trial`.
"""

from __future__ import annotations

import functools
import os
import signal
import sys
from pathlib import Path
from typing import Any

from capx.envs.configs.instantiate import instantiate
from capx.envs.tasks.base import CodeExecutionEnvBase, TrialTimeout
from capx.envs.trial import (
    _annotate_code_blocks,
    _build_log_lines,
    _run_single_trial,
)
from capx.serving.api_servers import ApiServerSpec
from capx.utils.launch_utils import (
    TrialSummary,
    _print_and_save_summary,
    _save_trial_artifacts,
    run_server_proc,
)
from capx.utils.parallel_eval import run_parallel_with_setup
from capx.utils.unix_http import is_listening, resolve_socket

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRIAL_TIMEOUT_SECONDS = 1000
MAX_TRIAL_RETRIES = 3
# Once a trial's deadline has passed it is raised again this often until the trial
# unwinds, so code that swallowed it once cannot keep the trial running.
DEADLINE_REPEAT_SECONDS = 1.0


# ---------------------------------------------------------------------------
# API server helpers
# ---------------------------------------------------------------------------

def _start_api_servers(
    api_servers: list | None, wait_timeout: float = 120.0
) -> list:
    """Launch the API server sub-processes defined in the config.

    Skips servers whose socket already has a listener (e.g. started externally).
    After launching, waits until all servers are accepting connections.
    """
    import time

    raw_specs = list(api_servers or [])
    specs = [ApiServerSpec.model_validate(raw) for raw in raw_specs]
    procs = []
    to_wait: list[tuple[Path, Any]] = []
    try:
        for raw, spec in zip(raw_specs, specs, strict=True):
            path = resolve_socket(spec.socket)
            if is_listening(path):
                print(f"API server on {path} already running, skipping")
                continue
            proc = run_server_proc(raw)
            procs.append(proc)
            print(f"API server {raw} started")
            to_wait.append((path, proc))

        deadline = time.time() + wait_timeout
        for path, proc in to_wait:
            while time.time() < deadline:
                if is_listening(path):
                    print(f"API server on {path} is ready")
                    break
                if not proc.is_alive():
                    print(f"WARNING: API server for {path} exited before it was ready")
                    break
                time.sleep(1.0)
            else:
                print(f"WARNING: API server on {path} not ready after {wait_timeout}s")
    except BaseException:
        # The caller only gets the processes back on success, so a failed or
        # interrupted startup has to stop the ones it already launched.
        _stop_api_servers(procs)
        raise

    return procs


def _stop_api_servers(server_procs: list, grace_s: float = 5.0) -> None:
    """Terminate API server sub-processes, killing any that ignore SIGTERM."""
    for proc in server_procs:
        proc.terminate()
    for proc in server_procs:
        proc.join(timeout=grace_s)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=grace_s)


# ---------------------------------------------------------------------------
# Output directory setup
# ---------------------------------------------------------------------------

def _setup_output_dir(args, config: dict[str, Any]) -> None:
    """Create and normalize the output directory path.

    Inserts the model name into the output path so that results from
    different models are stored separately.
    """
    if args.use_oracle_code:
        args.model = "oracle"
    if config["output_dir"]:
        parts = config["output_dir"].split("/")
        parts.insert(-1, str(args.model).replace("/", "_"))
        new_out_dir = "/".join(parts)
        Path(new_out_dir).mkdir(parents=True, exist_ok=True)
        config["output_dir"] = new_out_dir


# ---------------------------------------------------------------------------
# Headless trial runner
# ---------------------------------------------------------------------------

def _run_headless_trials(
    args,
    env_factory: dict[str, Any],
    config: dict[str, Any],
    start_time: float,
) -> None:
    """Run all trials in headless CLI mode, then print a summary.

    With ``num_workers`` of 1 the trials run in this process, otherwise in workers.
    """
    if config["total_trials"] <= 0:
        print("No trials requested; exiting.")
        return

    if config["record_video"] and not config["output_dir"]:
        raise ValueError("record_video requires --output-dir to save generated videos")

    _setup_output_dir(args, config)

    # Determine which trials to run (supports resume)
    trial_ids = list(range(1, config["total_trials"] + 1))
    if config.get("resume_idx") is not None:
        trial_ids = list(range(config["resume_idx"], config["total_trials"] + 1))

    # One worker runs the trials in this process; either way a trial that raises is
    # recorded as failed and the rest still run.
    summaries = run_parallel_with_setup(
        trial_ids,
        num_workers=config["num_workers"],
        setup_fn=functools.partial(_worker_setup, env_factory=env_factory),
        trial_fn=functools.partial(_run_single_trial_worker, args=args, config=config),
        on_failure=_failed_trial_summary,
    )
    summaries.sort(key=lambda s: s.trial)
    _print_and_save_summary(summaries, args, config, start_time)

    # Write completion flag
    os.makedirs(os.path.join(config["output_dir"], "aaa_done_flag"), exist_ok=True)
    with open(os.path.join(config["output_dir"], "aaa_done_flag", "aaa_done_flag.txt"), "w") as f:
        f.write("1")


# ---------------------------------------------------------------------------
# Worker setup and trial dispatch
# ---------------------------------------------------------------------------

def _instantiate_env(env_factory: dict[str, Any]) -> CodeExecutionEnvBase:
    # OmniGibson / Isaac Sim reads sys.argv and can crash on unknown flags, and a
    # spawned worker inherits the parent's argv. Strip our CLI args while building.
    original_sys_argv = sys.argv[:]
    try:
        sys.argv = sys.argv[:1]
        return instantiate(env_factory)
    finally:
        sys.argv = original_sys_argv


def _worker_setup(*, env_factory: dict[str, Any]) -> tuple[Any, str | None]:
    """Create the environment once per worker, and reuse it for all its trials.

    Returns:
        Tuple of (environment, multi_turn_prompt).
    """
    env = _instantiate_env(env_factory)
    multi_turn_prompt = env_factory["cfg"].get("multi_turn_prompt", None)
    return (env, multi_turn_prompt)


def _run_single_trial_worker(
    state: tuple[Any, str | None],
    trial: int,
    *,
    args,
    config: dict[str, Any],
) -> TrialSummary:
    """Run a single trial using pre-initialized worker state."""
    env, multi_turn_prompt = state
    return _run_trial_with_retries(env, trial, args, config, multi_turn_prompt)


def _run_trial_with_retries(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
) -> TrialSummary:
    """Attempt a trial up to MAX_TRIAL_RETRIES times, retrying on timeout."""
    for attempt in range(MAX_TRIAL_RETRIES):
        try:
            is_last_attempt = attempt == MAX_TRIAL_RETRIES - 1
            return _run_single_trial_with_timeout(
                env=env,
                trial=trial,
                args=args,
                config=config,
                multi_turn_prompt=multi_turn_prompt,
                timeout_s=TRIAL_TIMEOUT_SECONDS,
                raise_on_timeout=not is_last_attempt,
            )
        except (TimeoutError, TrialTimeout):
            # TrialTimeout only gets here when a repeat of the deadline lands while
            # the trial is already being wound up; it is still that trial's timeout.
            print(f"Trial {trial} timed out (attempt {attempt + 1}/{MAX_TRIAL_RETRIES}). Retrying...")

    # All retries exhausted
    return TrialSummary(
        trial=trial,
        success=False,
        reward=0.0,
        terminated=False,
        truncated=True,
        sandbox_rc=1,
        log=f"Trial {trial} failed after {MAX_TRIAL_RETRIES} timeout retries",
        task_completed=False,
        code_path=None,
        num_regenerations=0,
        num_finishes=0,
        num_code_blocks=0,
    )


def _failed_trial_summary(trial: int, reason: str) -> TrialSummary:
    """A trial that produced no summary of its own still counts, as a failure."""
    return TrialSummary(
        trial=trial,
        success=False,
        reward=0.0,
        terminated=False,
        truncated=False,
        sandbox_rc=1,
        log=f"Trial {trial} failed without a result:\n{reason}",
        task_completed=False,
    )


# ---------------------------------------------------------------------------
# Timeout wrapper
# ---------------------------------------------------------------------------

def _run_single_trial_with_timeout(
    env: CodeExecutionEnvBase,
    trial: int,
    args,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    timeout_s: float,
    raise_on_timeout: bool = False,
) -> TrialSummary:
    """Run a single trial with a wall-clock timeout via SIGALRM."""
    timeout_seconds = max(1, int(timeout_s))
    timed_out = False

    def _timeout_handler(signum: int, frame) -> None:  # type: ignore[override]
        nonlocal timed_out
        timed_out = True
        # Re-armed here rather than with an interval: trial.py restarts the deadline
        # with a one-shot signal.alarm after the reset.
        signal.setitimer(signal.ITIMER_REAL, DEADLINE_REPEAT_SECONDS)
        raise TrialTimeout(f"Trial {trial} exceeded {timeout_seconds} seconds")

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    partial_artifacts: dict[str, Any] = {}
    cause: BaseException
    try:
        try:
            summary = _run_single_trial(
                env, trial, args, config, multi_turn_prompt, partial_artifacts=partial_artifacts
            )
        except BaseException as exc:
            if not (timed_out or _is_timeout(exc)):
                raise
            cause = exc
        else:
            if not timed_out:
                return summary
            # Something caught the deadline (a bare ``except:`` in generated code) and
            # the trial ran on; it still ran out of time.
            cause = TrialTimeout(f"Trial {trial} exceeded {timeout_seconds} seconds")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)

    if raise_on_timeout:
        raise TimeoutError(f"Trial {trial} timed out") from cause
    print(f"Trial {trial} timed out after {timeout_seconds} seconds")
    return _build_timeout_summary(trial, timeout_seconds, partial_artifacts, config, cause)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TrialTimeout, TimeoutError)):
        return True
    try:
        import requests
    except ImportError:
        return False
    return isinstance(exc, requests.exceptions.Timeout)


def _build_timeout_summary(
    trial: int,
    timeout_seconds: int,
    pa: dict[str, Any],
    config: dict[str, Any],
    exc: BaseException,
) -> TrialSummary:
    """Build a TrialSummary from partial artifacts after a timeout."""
    raw_code = pa.get("raw_code", "")
    code_blocks = pa.get("code_blocks", [])
    code_block_metadata = pa.get("code_block_metadata", [])
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)

    info_step = pa.get("info_step", {"sandbox_rc": 1, "stdout": "", "stderr": str(exc)})
    if info_step.get("stderr") == "":
        info_step["stderr"] = str(exc)
    else:
        info_step["stderr"] += f"\n\nTimeout Error: {exc}"

    reward = pa.get("reward", 0.0)
    terminated = pa.get("terminated", False)
    truncated = pa.get("truncated", False)
    num_regenerations = pa.get("num_regenerations", 0)
    num_finishes = pa.get("num_finishes", 0)
    num_code_blocks = pa.get("num_code_blocks", len(code_blocks))

    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        prefix=f"Trial {trial} timed out after {timeout_seconds} seconds.",
    )

    code_path = _save_trial_artifacts(
        config, trial,
        sandbox_rc=1,
        reward=reward,
        task_completed=info_step.get("task_completed", False),
        final_code=final_code,
        raw_code=raw_code,
        all_responses=pa.get("all_responses", []),
        log_lines=log_lines,
        visual_feedback_imgs=pa.get("visual_feedback_imgs", []),
        ensemble_data=pa.get("ensemble_data"),
        multiturn_ensemble_data=pa.get("multiturn_ensemble_data", []),
    )

    return TrialSummary(
        trial=trial,
        success=False,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=1,
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", False),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )