"""The gateway: the refereed ``EnvProtocol`` that hosts a task pack's low-level envs.

- :class:`Referee` allocates the hidden seed, charges trials, enforces the job
  budget and ends the job on ``finish``; a lane ends a trial at the task's
  ``trial_steplimit``;
- :class:`Scoring` keeps reward and success internal;
- an :class:`EnvFactory` builds the simulator for a task. A sim exception, while
  stepping or while building an observation, becomes ``env_crash``: the episode is
  voided (trial refunded, score dropped) and the env discarded so the next reset
  rebuilds it.

Only ``terminated`` / ``truncated`` cross back, never reward, success or the seed.

Concurrency is expressed as lanes (:class:`GatewaySession`), each holding at most one
live episode and its env. N lanes run N episodes at once against N env backends
while sharing one referee, which keeps trial caps, the job budget and seed
allocation team-wide however wide the run is.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

from capx.config import EvalConfig, load_eval_config
from capx.gateway.low_level import LowLevelEnv, Recordable, StepOutcome
from capx.gateway.recording import Recorder
from capx.gateway.referee import EnforcementMode, Episode, Referee
from capx.gateway.scoring import EpisodeScore, Scoring
from capx.gateway.timing import Timings
from capx.protocol.actions import (
    BimanualFrankaRawAction,
    FrankaRawAction,
    InvalidActionError,
    R1ProRawAction,
    RawAction,
    check_bimanual_franka_bounds,
    check_franka_bounds,
)
from capx.protocol.env_protocol import (
    ErrorCode,
    GatewayError,
    RequestError,
    ResetResult,
    StepResult,
)
from capx.protocol.observations import FullObs, ObsFilter, ProprioObs
from capx.protocol.taskpack import Embodiment, TaskSpec


class EnvFactory(Protocol):
    """Builds a fresh low-level env for a task.

    ``supports_parallel`` declares whether ``build`` hands out independent envs that
    are safe to drive from concurrent lanes; ``Gateway`` refuses more than one lane
    on a factory that does not.
    """

    supports_parallel: bool

    def build(self, spec: TaskSpec) -> LowLevelEnv: ...


def full_obs(env: LowLevelEnv, obs_filter: ObsFilter) -> FullObs:
    return obs_filter.to_full(env.get_observation(), sim_time=env.sim_time)


def proprio_obs(env: LowLevelEnv, obs_filter: ObsFilter) -> ProprioObs:
    return obs_filter.to_proprio(env.get_proprio(), sim_time=env.sim_time)


_ObsT = TypeVar("_ObsT", ProprioObs, FullObs)


@dataclass
class _BuiltEnv:
    """The env a lane holds, reused across the trials of one task."""

    task_id: str
    env: LowLevelEnv
    obs_filter: ObsFilter


@dataclass
class _Active:
    episode: Episode
    spec: TaskSpec
    env: LowLevelEnv
    obs_filter: ObsFilter
    cycles: int = 0
    recording: bool = False


class _InFlight:
    """Counts the lane calls in progress, so the gateway can wait for them to record
    their scores before the final results dump."""

    def __init__(self) -> None:
        self._count = 0
        self._idle = threading.Condition()

    @contextlib.contextmanager
    def call(self) -> Iterator[None]:
        with self._idle:
            self._count += 1
        try:
            yield
        finally:
            with self._idle:
                self._count -= 1
                self._idle.notify_all()

    def wait_idle(self, timeout_s: float) -> bool:
        with self._idle:
            return self._idle.wait_for(lambda: self._count == 0, timeout_s)


@dataclass(frozen=True)
class _Shared:
    """What every lane shares, so a wide run charges the ledger a serial one would."""

    tasks: dict[str, TaskSpec]
    factory: EnvFactory
    config: EvalConfig
    referee: Referee
    scoring: Scoring
    timing: Timings
    clock: Callable[[], float]
    in_flight: _InFlight
    recorder: Recorder | None


_logger = logging.getLogger(__name__)

# What robosuite raises when a step arrives after its horizon
# (third_party/robosuite/robosuite/environments/base.py: `if self.done: raise
# ValueError(...)`, and the same line in LIBERO's vendored copy). Matched on the
# text because that is the only signal it gives; `capx/submission/interfaces.py`
# keys off the same string on the other side of the wire, for the same reason.
_HORIZON_REACHED = "executing action in terminated episode"


def _is_horizon_reached(exc: Exception) -> bool:
    return _HORIZON_REACHED in str(exc)


class GatewaySession:
    """One lane of a :class:`Gateway`: at most one live episode and its env.

    A new ``reset`` supersedes this lane's previous episode, and the env is reused
    across trials of the same task.
    """

    def __init__(self, shared: _Shared) -> None:
        self._shared = shared
        self._built: _BuiltEnv | None = None
        self._active: _Active | None = None
        self._lock = threading.RLock()

    def manifest(self) -> list[dict[str, Any]]:
        return [
            spec.to_manifest_entry().model_dump(mode="json")
            for spec in self._shared.tasks.values()
        ]

    def reset(self, task_id: str) -> ResetResult:
        shared = self._shared
        spec = shared.tasks.get(task_id)
        if spec is None:
            raise RequestError(f"unknown task_id: {task_id!r}")
        with self._lock, shared.in_flight.call():
            episode = shared.referee.begin_episode(task_id)
            # Retire the superseded episode before any env work, so the referee's
            # live set stays bounded even when the build or reset below crashes.
            self._end_episode()
            shared.timing.begin(episode.episode_id, spec.task_id)
            built = self._env_for(spec, episode)
            started = shared.clock()
            try:
                built.env.reset(seed=episode.seed)
            except Exception as exc:
                shared.timing.add_sim(episode.episode_id, shared.clock() - started, steps=0)
                raise self._crash(episode.episode_id, exc) from exc
            shared.timing.add_sim(episode.episode_id, shared.clock() - started, steps=0)
            active = self._active = _Active(episode, spec, built.env, built.obs_filter)
            active.recording = self._start_recording(active)
            # An unstepped trial still counts as a failure; env_crash later drops it.
            shared.scoring.record(
                episode.episode_id, task_id=spec.task_id, reward=0.0, success=False
            )
            return ResetResult(
                episode.episode_id, self._observe(active, full_obs), spec.instruction
            )

    def step(self, episode_id: str, action: RawAction) -> StepResult:
        shared = self._shared
        with self._lock, shared.in_flight.call():
            shared.referee.check_job_budget()
            active = self._require(episode_id)
            # An action outside the published bounds is refused before it reaches
            # the simulator; the trial goes on and the submission may send another.
            try:
                self._check_bounds(active.spec, action)
            except InvalidActionError as exc:
                raise GatewayError(ErrorCode.INVALID_ACTION, str(exc)) from exc
            limit = active.spec.trial_steplimit
            # Counted in control cycles, so `repeat` cannot buy steps past the limit.
            cycles = min(action.repeat, limit - active.cycles)
            if cycles <= 0:
                return StepResult(
                    obs=self._step_obs(active, action), terminated=False, truncated=True
                )
            started, cycles_before = shared.clock(), active.cycles
            # Set when the job ends partway through this step (its budget, or a finish
            # from another lane). The cycles already run are still scored.
            stopped: GatewayError | None = None
            outcome = StepOutcome()
            try:
                for i in range(cycles):
                    # A single high-repeat step must not ride past the end of the job.
                    if i:
                        try:
                            shared.referee.check_job_budget()
                        except GatewayError as exc:
                            stopped = exc
                            break
                    outcome = active.env.apply_raw_action(action)
                    active.cycles += 1
                    # Checked after every cycle, so a success reached partway through a
                    # `repeat` counts even if a later cycle undoes it, and recorded at
                    # once: a later cycle may still be running when the job ends.
                    if outcome.success:
                        shared.scoring.record_success(episode_id, active.spec.task_id)
                        shared.timing.succeed(episode_id, active.cycles)
                    if outcome.terminated or outcome.truncated:
                        break
                reward = active.env.compute_reward()
            except GatewayError:
                raise
            except Exception as exc:
                if not _is_horizon_reached(exc):
                    raise self._crash(episode_id, exc) from exc
                # env_build sets the env's horizon well past the steplimit, so this
                # is an env that ignored it. It still ends the episode rather than
                # voiding it: voiding refunds the trial, and a losing episode run to
                # the horizon would get the attempt back.
                _logger.warning(
                    "episode %s hit the simulator's horizon before trial_steplimit", episode_id
                )
                outcome = StepOutcome(truncated=True)
                try:
                    reward = active.env.compute_reward()
                except Exception as scoring_exc:  # the sim really is broken
                    raise self._crash(episode_id, scoring_exc) from scoring_exc
            finally:
                shared.timing.add_sim(
                    episode_id, shared.clock() - started, steps=active.cycles - cycles_before
                )
            if outcome.terminated or outcome.truncated:
                # The env ended the episode itself; a later step must not reach it.
                active.cycles = limit
            if active.cycles >= limit:
                shared.timing.reach_limit(episode_id)
            shared.scoring.record(episode_id, task_id=active.spec.task_id, reward=reward)
            if stopped is not None:
                raise stopped
            return StepResult(
                obs=self._step_obs(active, action),
                terminated=outcome.terminated,
                truncated=active.cycles >= limit,
            )

    def get_observation(self, episode_id: str) -> FullObs:
        shared = self._shared
        with self._lock, shared.in_flight.call():
            shared.referee.check_not_finished()
            return self._observe(self._require(episode_id), full_obs)

    def finish(self) -> None:
        """End the whole job, whichever lane asks. An episode in flight on another lane
        stops at its next control cycle and keeps what it scored."""
        self._shared.referee.finish()

    def close(self) -> None:
        """Retire this lane: end its episode and give its env back."""
        with self._lock:
            self._end_episode()
            self._discard_env()

    def _end_episode(self) -> None:
        """End the lane's episode, keeping its charged trial and score."""
        if self._active is not None:
            self._save_recording(self._active)
            episode_id = self._active.episode.episode_id
            # A lane is also closed when the submission is stopped at the end of the job.
            how = "reset" if self._shared.referee.job_open() else "cut"
            self._shared.referee.end_episode(episode_id)
            self._shared.timing.end(episode_id, how)
            self._active = None

    def _start_recording(self, active: _Active) -> bool:
        recorder = self._shared.recorder
        episode = active.episode
        if recorder is None or not recorder.wants(episode.task_id, episode.trial_index):
            return False
        if not isinstance(active.env, Recordable):
            return False
        try:
            active.env.start_recording(recorder.fps)
        except Exception:
            _logger.exception("could not start recording episode %s", episode.episode_id)
            return False
        return True

    def _save_recording(self, active: _Active) -> None:
        recorder = self._shared.recorder
        if recorder is None or not active.recording or not isinstance(active.env, Recordable):
            return
        episode = active.episode
        try:
            frames = active.env.stop_recording()
        except Exception:
            _logger.exception("could not fetch the recording of episode %s", episode.episode_id)
            return
        success = self._shared.scoring.succeeded(episode.episode_id)
        try:
            recorder.save(active.spec, episode.trial_index, success, frames)
        except Exception:
            _logger.exception("could not save the recording of episode %s", episode.episode_id)

    def _step_obs(self, active: _Active, action: RawAction) -> ProprioObs | FullObs:
        if action.include_cameras:
            return self._observe(active, full_obs)
        return self._observe(active, proprio_obs)

    def _observe(self, active: _Active, build: Callable[[LowLevelEnv, ObsFilter], _ObsT]) -> _ObsT:
        """Build an observation; a sim failure here is an env crash like any other."""
        shared = self._shared
        episode_id = active.episode.episode_id
        started = shared.clock()
        try:
            obs = build(active.env, active.obs_filter)
        except Exception as exc:
            raise self._crash(episode_id, exc) from exc
        shared.timing.add_obs(episode_id, shared.clock() - started)
        return obs

    def _check_bounds(self, spec: TaskSpec, action: RawAction) -> None:
        if spec.embodiment is Embodiment.FRANKA:
            if not isinstance(action, FrankaRawAction):
                raise InvalidActionError("expected a franka action for a franka task")
            franka = self._shared.config.embodiments.franka
            check_franka_bounds(
                action,
                franka.joint_lower,
                franka.joint_upper,
                franka.gripper_min,
                franka.gripper_max,
            )
        elif spec.embodiment is Embodiment.FRANKA_BIMANUAL:
            if not isinstance(action, BimanualFrankaRawAction):
                raise InvalidActionError(
                    "expected a franka_bimanual action for a franka_bimanual task"
                )
            franka = self._shared.config.embodiments.franka
            check_bimanual_franka_bounds(
                action,
                franka.joint_lower,
                franka.joint_upper,
                franka.gripper_min,
                franka.gripper_max,
            )
        elif spec.embodiment is Embodiment.R1PRO:
            if not isinstance(action, R1ProRawAction):
                raise InvalidActionError("expected an r1pro action for an r1pro task")
            action_dim = self._shared.config.embodiments.r1pro.action_dim
            got = np.asarray(action.action).shape[0]
            if action_dim is not None and got != action_dim:
                raise InvalidActionError(f"r1pro.action must have dim {action_dim}, got {got}")

    def _env_for(self, spec: TaskSpec, episode: Episode) -> _BuiltEnv:
        """Reuse the lane's env across trials of the same task; rebuild it otherwise.

        The lane owns a new env as soon as it is built, so a crash in the reset that
        follows closes it and gives its backend back.
        """
        if self._built is not None and self._built.task_id == spec.task_id:
            return self._built
        self._discard_env()
        try:
            env = self._shared.factory.build(spec)
            built = self._built = _BuiltEnv(
                spec.task_id, env, ObsFilter(camera_names=spec.camera_names)
            )
            if env.embodiment != spec.embodiment.value:
                raise RuntimeError(
                    f"task {spec.task_id!r} is {spec.embodiment.value}, "
                    f"but its env is {env.embodiment}"
                )
        except Exception as exc:
            raise self._crash(episode.episode_id, exc) from exc
        return built

    def _crash(self, episode_id: str, cause: Exception) -> GatewayError:
        """Roll back a crashed episode: refund the trial, drop its score, discard the env.

        The sim's exception text is logged gateway-side only — it can carry
        internals (paths, object names, state) and must never reach the client,
        so the wire detail is a fixed string.
        """
        _logger.error("env_crash in episode %s: %s: %s", episode_id, type(cause).__name__, cause)
        self._shared.scoring.drop(episode_id)
        self._shared.timing.drop(episode_id)
        self._shared.referee.void_episode(episode_id)
        self._discard_env()
        return GatewayError(ErrorCode.ENV_CRASH, "environment crashed; episode voided")

    def _discard_env(self) -> None:
        self._active = None
        if self._built is not None:
            with contextlib.suppress(Exception):
                self._built.env.close()
            self._built = None

    def _require(self, episode_id: str) -> _Active:
        if self._active is None or self._active.episode.episode_id != episode_id:
            raise GatewayError(ErrorCode.CLIENT_TIMEOUT, f"unknown episode {episode_id!r}")
        return self._active


class Gateway:
    """The refereed task pack, served through up to ``max_sessions`` lanes."""

    def __init__(
        self,
        tasks: dict[str, TaskSpec],
        factory: EnvFactory,
        mode: EnforcementMode,
        *,
        config: EvalConfig | None = None,
        job_timeout_s: float | None = None,
        master_seed: int = 0,
        clock: Callable[[], float] = time.monotonic,
        max_sessions: int = 1,
        on_clock_start: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        if mode is EnforcementMode.OFFICIAL:
            leaky = sorted(t for t, s in tasks.items() if s.privileged)
            if leaky:
                raise ValueError(f"privileged tasks are forbidden in official mode: {leaky}")
        if max_sessions > 1 and not factory.supports_parallel:
            raise ValueError(
                f"max_sessions={max_sessions} needs a factory that hands out isolated "
                f"envs, but {type(factory).__name__} is not parallel-safe (it shares one "
                "cached sim across lanes). Use max_sessions=1 or a process-isolated factory."
            )
        config = config or load_eval_config()
        self._shared = _Shared(
            tasks=tasks,
            factory=factory,
            config=config,
            referee=Referee(
                tasks,
                mode,
                job_timeout_s=job_timeout_s,
                master_seed=master_seed,
                clock=clock,
                on_clock_start=on_clock_start,
                on_finish=on_finish,
            ),
            scoring=Scoring(),
            timing=Timings(clock),
            clock=clock,
            in_flight=_InFlight(),
            recorder=recorder,
        )
        self._max_sessions = max(1, max_sessions)
        self._sessions_lock = threading.Lock()
        self._sessions: list[GatewaySession] = []
        self._free: list[GatewaySession] = []

    def __enter__(self) -> Gateway:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def acquire(self) -> GatewaySession:
        """Claim a lane, up to ``max_sessions``.

        Raises rather than queueing: a submission asking for more concurrency than
        the organizer provisioned backends for should be told so, not silently
        serialized behind a lane it believes is its own.
        """
        self._shared.referee.start_clock()
        with self._sessions_lock:
            if self._free:
                return self._free.pop()
            if len(self._sessions) >= self._max_sessions:
                raise ValueError(
                    f"gateway allows {self._max_sessions} concurrent lane(s); "
                    f"all {len(self._sessions)} are in use"
                )
            session = GatewaySession(self._shared)
            self._sessions.append(session)
            return session

    def release(self, session: GatewaySession) -> None:
        """Close a lane and hand it back (a submission disconnected)."""
        session.close()
        with self._sessions_lock:
            if session in self._sessions and session not in self._free:
                self._free.append(session)

    def close(self) -> None:
        """Close every lane."""
        with self._sessions_lock:
            sessions = list(self._sessions)
            self._sessions = []
            self._free = []
        for session in sessions:
            session.close()

    # -- referee/scoring views (server-internal, never over the wire) --------

    def aggregate(self) -> dict[str, dict[str, float]]:
        """Per-task results over the pack, not over what the submission attempted.

        A task with no episode is a zero, not an absence: a denominator taken from
        the episodes would be one the submission picks by choosing what to reset.
        In official mode a task is likewise divided by its trial budget, not by the
        trials that ran.
        """
        return self._shared.scoring.aggregate(self._shared.referee.scoring_trials())

    def scores(self) -> list[EpisodeScore]:
        return self._shared.scoring.episodes()

    def timings(self) -> dict[str, object]:
        return self._shared.timing.report()

    def trials_used(self, task_id: str) -> int:
        return self._shared.referee.trials_used(task_id)

    def wait_idle(self, timeout_s: float) -> bool:
        """Wait for the lane calls in progress to finish, so a step that the end of the
        job cut short has recorded its score. False if some are still running."""
        return self._shared.in_flight.wait_idle(timeout_s)

    def finished_at(self) -> float | None:
        """When the submission's ``finish()`` was taken (epoch seconds), if it was."""
        return self._shared.referee.finished_at

    def end_job(self) -> None:
        """End the job from the organizer's side (``Referee.end``): a step in flight stops
        at its next control cycle, and every later call is refused."""
        self._shared.referee.end()

    def live_episodes(self) -> list[str]:
        """Episodes the referee still holds open: at most one per lane in use."""
        return self._shared.referee.live_episodes()


__all__ = ["EnvFactory", "Gateway", "GatewaySession", "full_obs", "proprio_obs"]
