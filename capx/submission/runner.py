"""Run the reference agent's trials against the gateway.

Each manifest task resolves to a task family config. ``cfg.concurrency`` lanes, each
on its own gateway connection, pull (task, trial) items off one shared worklist and
run them through ``capx.envs.trial._run_single_trial`` directly, never through
``capx.envs.runner``'s retry wrapper, so a reset error does not silently spend
another official trial. A trial the gateway voids is refunded, so it goes back on the
worklist; a lane whose connection fails puts its trial back and stops.
"""

from __future__ import annotations

import contextlib
import dataclasses
import enum
import multiprocessing
import os
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capx.envs.launch import LaunchArgs
from capx.envs.runner import _start_api_servers, _stop_api_servers
from capx.envs.trial import _run_single_trial
from capx.protocol import EnvClient, ErrorCode, GatewayError
from capx.protocol.taskpack import Embodiment, ManifestEntry, Simulator
from capx.submission.code_env import GatewayCodeExecEnv, build_code_env
from capx.submission.interfaces import (
    GATEWAY_CONFIG_DIR,
    SubmissionConfig,
    TaskRunConfig,
    load_task_run_config,
)
from capx.submission.shims import (
    BimanualFrankaGatewayShim,
    FrankaGatewayShim,
    GatewayShimBase,
    R1ProGatewayShim,
)
from capx.submission.transport import build_env_clients
from capx.submission.worklist import Worklist
from capx.utils.launch_utils import TrialSummary

_DEFAULT_ROBOSUITE_CONFIG = str(GATEWAY_CONFIG_DIR / "franka_robosuite.yaml")
_DEFAULT_LIBERO_CONFIG = str(GATEWAY_CONFIG_DIR / "franka_libero.yaml")
_DEFAULT_R1PRO_CONFIG = str(GATEWAY_CONFIG_DIR / "r1pro.yaml")
_DEFAULT_BIMANUAL_CONFIG = str(GATEWAY_CONFIG_DIR / "franka_two_arm_lift.yaml")

# The manifest has no simulator field; LIBERO tasks are the Franka ones with this camera.
_LIBERO_CAMERA_MARKER = "agentview"


# How often one trial may be run again after the gateway voided it or the lane
# running it lost its connection. A refund costs no trial, only job time.
_MAX_RETRIES = 2


class _Outcome(enum.Enum):
    DONE = enum.auto()
    VOIDED = enum.auto()
    TASK_EXHAUSTED = enum.auto()
    RUN_OVER = enum.auto()
    DISCONNECTED = enum.auto()


_OUTCOME_BY_CODE = {
    ErrorCode.ENV_CRASH: _Outcome.VOIDED,
    ErrorCode.TRIALS_EXHAUSTED: _Outcome.TASK_EXHAUSTED,
    ErrorCode.JOB_TIMEOUT: _Outcome.RUN_OVER,
    ErrorCode.JOB_FINISHED: _Outcome.RUN_OVER,
}


def simulator_of(entry: ManifestEntry) -> Simulator:
    """Which simulator runs the task, told from its embodiment and cameras."""
    if entry.embodiment is Embodiment.R1PRO:
        return Simulator.BEHAVIOR
    # Before the camera check: a two-arm task renders "agentview" too.
    if entry.embodiment is Embodiment.FRANKA_BIMANUAL:
        return Simulator.ROBOSUITE
    if entry.embodiment is Embodiment.FRANKA:
        if _LIBERO_CAMERA_MARKER in entry.camera_names:
            return Simulator.LIBERO
        return Simulator.ROBOSUITE
    raise ValueError(f"no simulator known for embodiment {entry.embodiment.value!r}")


def resolve_task_config_path(entry: ManifestEntry, cfg: SubmissionConfig) -> str:
    """The ``task_map`` entry for this task, else a config chosen by embodiment and cameras."""
    for mapping in cfg.task_map:
        if mapping.task_id == entry.task_id:
            return mapping.config
    if entry.embodiment is Embodiment.FRANKA_BIMANUAL:
        return _DEFAULT_BIMANUAL_CONFIG
    return {
        Simulator.BEHAVIOR: _DEFAULT_R1PRO_CONFIG,
        Simulator.LIBERO: _DEFAULT_LIBERO_CONFIG,
        Simulator.ROBOSUITE: _DEFAULT_ROBOSUITE_CONFIG,
    }[simulator_of(entry)]


def _build_shim(client: EnvClient, entry: ManifestEntry, task_cfg: TaskRunConfig) -> GatewayShimBase:
    camera_names = list(entry.camera_names)
    if entry.embodiment is Embodiment.R1PRO:
        return R1ProGatewayShim(
            client,
            entry.task_id,
            camera_names=camera_names,
            step_chunk=task_cfg.step_chunk,
            knobs=task_cfg.r1pro,
        )
    if entry.embodiment is Embodiment.FRANKA_BIMANUAL:
        shim_cls: type[GatewayShimBase] = BimanualFrankaGatewayShim
    elif entry.embodiment is Embodiment.FRANKA:
        shim_cls = FrankaGatewayShim
    else:
        raise ValueError(f"no shim for embodiment {entry.embodiment.value!r}")
    return shim_cls(client, entry.task_id, camera_names=camera_names, step_chunk=task_cfg.step_chunk)


def _build_launch_args(cfg: SubmissionConfig, config_path: str) -> LaunchArgs:
    return LaunchArgs(
        config_path=config_path,
        server_url=cfg.server_url,
        model=cfg.model,
        # The differencing model defaults to hosted Gemini; a submission has one model
        # and no network, so point it at that one.
        visual_differencing_model=cfg.model,
        visual_differencing_model_server_url=cfg.server_url,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        use_visual_feedback=cfg.use_visual_feedback,
        use_img_differencing=cfg.use_img_differencing,
        use_video_differencing=cfg.use_video_differencing,
        use_wrist_camera=cfg.use_wrist_camera,
        debug=False,
    )


def _build_config_dict(
    cfg: SubmissionConfig, task_cfg: TaskRunConfig, task_id: str, output_dir: str
) -> dict[str, Any]:
    """Every ``config`` key ``capx.envs.trial._run_single_trial`` reads."""
    return {
        "output_dir": output_dir,
        "record_video": task_cfg.record_video,
        "use_visual_feedback": cfg.use_visual_feedback,
        "use_img_differencing": cfg.use_img_differencing,
        "use_video_differencing": cfg.use_video_differencing,
        "use_wrist_camera": cfg.use_wrist_camera,
        # Oracle code is ground truth, and multi-model ensembling queries hosted
        # models a submission has no network to reach; neither may be on.
        "use_oracle_code": False,
        "use_parallel_ensemble": cfg.use_parallel_ensemble,
        "use_multimodel": False,
        "save_multiturn_prompts": False,
        "task_name": task_id,
    }


def _attempt_trial(
    env: GatewayCodeExecEnv,
    shim: GatewayShimBase,
    trial: int,
    launch_args: LaunchArgs,
    config_dict: dict[str, Any],
    multi_turn_prompt: str | None,
) -> tuple[_Outcome, TrialSummary | None]:
    """Run one trial and say what became of it.

    ``env.reset()`` raises the gateway's verdicts as ``GatewayError``. Mid-episode the
    shim turns them into ``RuntimeError`` for the agent's code, which may catch them,
    and records an ``env_crash`` or a failed connection on itself for this check.
    """
    try:
        summary = _run_single_trial(env, trial, launch_args, config_dict, multi_turn_prompt)
    except GatewayError as exc:
        outcome = _OUTCOME_BY_CODE.get(exc.code)
        if outcome is None:
            raise
        return outcome, None
    except ConnectionError:
        # Only the reset talks to the gateway outside the shim; a disk error writing
        # artifacts is the trial's failure, not the lane's.
        return _Outcome.DISCONNECTED, None
    except Exception:
        if not (shim.voided or shim.disconnected):
            raise
        summary = None
    if shim.disconnected:
        return _Outcome.DISCONNECTED, summary
    if shim.voided:
        return _Outcome.VOIDED, summary
    return _Outcome.DONE, summary


def _rss_gib() -> float:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        pass
    return 0.0


@dataclass(frozen=True)
class _Work:
    """One trial. Lanes pull these from a shared :class:`Worklist` rather than own a
    fixed slice: trial lengths vary by more than a factor of two, and a slice leaves
    the lane that drew the long trials running after the rest have gone idle."""

    entry: ManifestEntry
    config_path: str
    task_cfg: TaskRunConfig
    trial: int
    simulator: Simulator
    retries: int = 0


class _RunState:
    """Verdicts every lane must honour once any lane hears them."""

    def __init__(self) -> None:
        self.stop_run = threading.Event()
        self.exhausted: set[str] = set()
        self.lock = threading.Lock()

    def note(self, outcome: _Outcome, task_id: str) -> None:
        if outcome is _Outcome.RUN_OVER:
            self.stop_run.set()
        elif outcome is _Outcome.TASK_EXHAUSTED:
            with self.lock:
                self.exhausted.add(task_id)

    def skip(self, work: _Work) -> bool:
        """Whether a queued trial is not worth a reset.

        A trial queued again is always tried: if it was voided, its refund may have
        come after another lane found the shared budget spent, and its own reset
        then tells.
        """
        if self.stop_run.is_set():
            return True
        if work.retries:
            return False
        with self.lock:
            return work.entry.task_id in self.exhausted


def _worklist(manifest: list[ManifestEntry], cfg: SubmissionConfig) -> list[_Work]:
    task_cfg_cache: dict[str, TaskRunConfig] = {}
    work: list[_Work] = []
    for entry in manifest:
        config_path = resolve_task_config_path(entry, cfg)
        task_cfg = task_cfg_cache.get(config_path)
        if task_cfg is None:
            task_cfg = load_task_run_config(config_path)
            task_cfg_cache[config_path] = task_cfg
        trials = cfg.trials_per_task or entry.n_trials
        apis = ", ".join(task_cfg.apis)
        print(f"[runner] {entry.task_id}: {trials} trial(s), {config_path} ({apis})")
        simulator = simulator_of(entry)
        work.extend(
            _Work(entry, config_path, task_cfg, trial, simulator) for trial in range(trials)
        )
    return work


def _server_specs(task_cfg: TaskRunConfig) -> list[dict[str, Any]]:
    """``api_servers``, each hub ``model_name`` swapped for its local copy where one exists."""
    specs = []
    for spec in task_cfg.api_servers:
        local = task_cfg.local_models.get(spec.get("model_name", ""))
        specs.append({**spec, "model_name": local} if local and Path(local).is_dir() else spec)
    return specs


class _ApiServers:
    """The perception/IK servers, started once per config; they listen on fixed ports
    shared by every lane, so two lanes must not start them at once."""

    def __init__(self) -> None:
        self._started: set[str] = set()
        self._lock = threading.Lock()
        self.procs: list[multiprocessing.Process] = []

    def ensure(self, config_path: str, task_cfg: TaskRunConfig) -> None:
        with self._lock:
            if config_path in self._started:
                return
            self._started.add(config_path)
            self.procs.extend(_start_api_servers(_server_specs(task_cfg)))


def _run_lane(
    lane: int,
    client: EnvClient,
    pending: Worklist[_Work, Simulator],
    cfg: SubmissionConfig,
    state: _RunState,
    servers: _ApiServers,
) -> bool:
    """Run trials until none are left or in flight; False if the lane lost its connection.

    The lane keeps only the env of the task it is on: an env's shim buffers video and
    observations, so keeping every task's env would grow with the pack.
    """
    try:
        # The gateway refuses a connection past its lane cap on the first call.
        client.manifest()
    except (OSError, RuntimeError) as exc:
        print(f"[runner] lane {lane} could not claim a gateway lane ({exc}); running without it")
        return False
    shim: GatewayShimBase | None = None
    env: GatewayCodeExecEnv | None = None
    env_task: str | None = None
    idle = False
    while True:
        work = pending.take(lane)
        if work is None:
            if not idle:
                # A lane holds a simulator backend until it lets go, and backends are
                # capped: waiting on the others while holding one could starve them.
                _close_quietly(client)
                pending.release(lane)
                idle = True
            if pending.wait():
                continue
            return True
        idle = False
        requeue: _Work | None = None
        try:
            task_id = work.entry.task_id
            if state.skip(work):
                continue
            servers.ensure(work.config_path, work.task_cfg)
            if shim is None or env is None or env_task != task_id:
                shim = _build_shim(client, work.entry, work.task_cfg)
                env = build_code_env(shim, work.task_cfg)
                env_task = task_id
            output_dir = os.path.join(cfg.output_dir, task_id)
            os.makedirs(output_dir, exist_ok=True)
            label = f"lane {lane} task {task_id!r} trial {work.trial}"
            try:
                outcome, summary = _attempt_trial(
                    env,
                    shim,
                    work.trial,
                    _build_launch_args(cfg, work.config_path),
                    _build_config_dict(cfg, work.task_cfg, task_id, output_dir),
                    work.task_cfg.multi_turn_prompt,
                )
            except Exception:
                # One bad trial (e.g. a reply the code extractor cannot parse) must not
                # end the run.
                traceback.print_exc()
                print(f"[runner] {label} raised; moving on")
                continue
            if summary is not None:
                print(
                    f"[runner] {label}: sandbox_rc={summary.sandbox_rc} "
                    f"truncated={summary.truncated} rss={_rss_gib():.1f}GiB"
                )
            if outcome in (_Outcome.VOIDED, _Outcome.DISCONNECTED):
                name = outcome.name.lower()
                if work.retries < _MAX_RETRIES:
                    requeue = dataclasses.replace(work, retries=work.retries + 1)
                    print(f"[runner] {label}: {name}; queued to run again")
                else:
                    print(f"[runner] {label}: {name} {work.retries + 1} times; giving up")
                if outcome is _Outcome.DISCONNECTED:
                    print(f"[runner] lane {lane} lost its gateway connection; retiring it")
                    return False
                continue
            state.note(outcome, task_id)
        finally:
            pending.settle(requeue)


def run_submission(cfg: SubmissionConfig) -> None:
    lanes = max(1, cfg.concurrency)
    clients = build_env_clients(cfg, lanes)
    servers = _ApiServers()
    state = _RunState()

    try:
        manifest = [ManifestEntry.model_validate(entry) for entry in clients[0].manifest()]
        if cfg.task_filter is not None:
            allowed = set(cfg.task_filter)
            manifest = [entry for entry in manifest if entry.task_id in allowed]

        work = _worklist(manifest, cfg)
        print(
            f"[runner] {len(work)} trial(s) over {len(manifest)} task(s) across {lanes} "
            f"lane(s), lane caps {({sim.value: cap for sim, cap in cfg.simulator_lanes.items()})}"
        )
        pending = Worklist(work, lambda item: item.simulator, cfg.simulator_lanes)

        def run_lane(i: int, client: EnvClient) -> bool:
            try:
                return _run_lane(i, client, pending, cfg, state, servers)
            finally:
                _close_quietly(client)
                pending.release(i)

        with ThreadPoolExecutor(max_workers=lanes) as pool:
            futures = [pool.submit(run_lane, i, client) for i, client in enumerate(clients)]
            connected = [
                client for client, future in zip(clients, futures, strict=True) if future.result()
            ]
        left = [work for work in pending.drain() if not state.skip(work)]
        if left or not connected:
            # finish() would fix the score with these trials unrun; without it the job
            # runs on to its budget, and the organizer sees the failure.
            raise RuntimeError(
                f"lanes lost their gateway connection with {len(left)} trial(s) unrun; "
                "not finishing the job"
            )
        # Stops the job clock and fixes the score; the organizer then stops the container.
        connected[0].finish()
        print("[runner] all trials done; finished the job")
    finally:
        _stop_api_servers(servers.procs)
        for client in clients:
            _close_quietly(client)


def _close_quietly(client: EnvClient) -> None:
    with contextlib.suppress(Exception):
        client.close()


__all__ = ["run_submission", "resolve_task_config_path"]
