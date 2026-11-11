"""The referee: the fairness accounting a submission must not control.

It owns which seed an episode gets, how many trials a task has left, the
whole-job budget and whether the submission has finished the job; the per-trial
step limit is the gateway lane's. Every gateway lane
shares one referee, so all mutations take an internal lock.

Seeds are a pure function of ``(master_seed, task_id, trial_index)``. A reset takes
the smallest trial index of its task that is not handed out, and a voided
(``env_crash``) episode gives its index back. Its retry therefore runs the same task
instance whichever lane asks, so inducing a crash cannot reroll it, and however many
lanes run, the indices handed out are the ones a serial run would use.
"""

from __future__ import annotations

import enum
import heapq
import threading
import time
import uuid
import zlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from capx.protocol.env_protocol import ErrorCode, GatewayError
from capx.protocol.taskpack import TaskSpec


class EnforcementMode(str, enum.Enum):
    DEV = "dev"  # relaxed: unlimited trials
    OFFICIAL = "official"  # strict: trials capped


@dataclass
class Episode:
    episode_id: str
    task_id: str
    trial_index: int
    seed: int


class Referee:
    def __init__(
        self,
        tasks: dict[str, TaskSpec],
        mode: EnforcementMode,
        *,
        job_timeout_s: float | None = None,
        master_seed: int = 0,
        clock: Callable[[], float] = time.monotonic,
        on_clock_start: Callable[[], None] | None = None,
        on_finish: Callable[[], None] | None = None,
    ) -> None:
        self._tasks = tasks
        self._mode = mode
        self._job_timeout_s = job_timeout_s
        self._clock = clock
        self._trials_used = dict.fromkeys(tasks, 0)
        self._next_trial_index = dict.fromkeys(tasks, 0)
        # Indices given back by voided episodes; always below the task's next index.
        self._returned_indices: dict[str, list[int]] = {task_id: [] for task_id in tasks}
        self._master_seed = master_seed
        self._episodes: dict[str, Episode] = {}
        self._job_start: float | None = None
        self._on_clock_start = on_clock_start
        self._finished = False
        self.finished_at: float | None = None
        """When :meth:`finish` was taken (epoch seconds); None until then, and after
        :meth:`end`."""
        self._on_finish = on_finish
        self._lock = threading.Lock()

    @property
    def strict(self) -> bool:
        return self._mode is EnforcementMode.OFFICIAL

    def n_trials(self, task_id: str) -> int:
        return self._tasks[task_id].n_trials

    def scoring_trials(self) -> dict[str, int]:
        """Each task's minimum score denominator: its trial budget when trials are
        capped, else 0 so dev runs are scored over what they ran."""
        return {
            task_id: self.n_trials(task_id) if self.strict else 0 for task_id in self._tasks
        }

    def start_clock(self) -> None:
        """Start the job budget, once: at the submission's first contact with the
        gateway, so loading its model is not charged to it."""
        with self._lock:
            if self._job_start is not None:
                return
            self._job_start = self._clock()
        if self._on_clock_start is not None:
            self._on_clock_start()

    def finish(self) -> None:
        """End the job for good, once: the submission says its score is final. Every
        later call but ``finish`` and ``close`` is refused."""
        if self._end(finished=True) and self._on_finish is not None:
            self._on_finish()

    def end(self) -> None:
        """End the job as :meth:`finish` does, for the organizer: the submission is
        being stopped, and nothing it does from here is scored. ``on_finish`` does not
        run, since the submission did not finish."""
        self._end()

    def _end(self, *, finished: bool = False) -> bool:
        with self._lock:
            if self._finished:
                return False
            self._finished = True
            if finished:
                self.finished_at = time.time()
            return True

    def check_not_finished(self) -> None:
        if self._finished:
            raise GatewayError(ErrorCode.JOB_FINISHED, "the job was finished")

    def begin_episode(self, task_id: str) -> Episode:
        """Charge a trial and allocate a hidden seed, or refuse. Called from ``reset``."""
        self.check_job_budget()
        spec = self._tasks[task_id]
        with self._lock:
            used = self._trials_used[task_id]
            if self.strict and used >= self.n_trials(task_id):
                raise GatewayError(ErrorCode.TRIALS_EXHAUSTED, task_id)
            self._trials_used[task_id] = used + 1

            index = self._take_trial_index(task_id)
            episode = Episode(
                uuid.uuid4().hex, task_id, index, self._allocate_seed(spec, trial=index)
            )
            self._episodes[episode.episode_id] = episode
            return episode

    def _take_trial_index(self, task_id: str) -> int:
        returned = self._returned_indices[task_id]
        if returned:
            return heapq.heappop(returned)
        index = self._next_trial_index[task_id]
        self._next_trial_index[task_id] = index + 1
        return index

    def live_episodes(self) -> list[str]:
        """Episodes the referee still considers open, in no particular order."""
        with self._lock:
            return list(self._episodes)

    def void_episode(self, episode_id: str) -> None:
        """Roll an episode back on ``env_crash``: refund its trial, forget it.

        Its trial index goes back to the task, so the next reset of the task replays
        the same seed, and the job budget keeps ticking through crashes: voiding
        cannot be used to shop for task instances.
        """
        with self._lock:
            episode = self._episodes.pop(episode_id, None)
            if episode is not None:
                self._trials_used[episode.task_id] -= 1
                heapq.heappush(self._returned_indices[episode.task_id], episode.trial_index)

    def end_episode(self, episode_id: str) -> None:
        with self._lock:
            self._episodes.pop(episode_id, None)

    def trials_used(self, task_id: str) -> int:
        return self._trials_used[task_id]

    def _allocate_seed(self, spec: TaskSpec, *, trial: int) -> int:
        policy = spec.seed_policy
        if policy.mode == "fixed" and policy.seeds:
            return policy.seeds[trial % len(policy.seeds)]
        rng = np.random.default_rng(
            (self._master_seed, zlib.crc32(spec.task_id.encode()), trial)
        )
        return int(rng.integers(0, 2**31 - 1))

    def job_open(self) -> bool:
        """Whether the submission may still act: not finished, ended or out of budget."""
        try:
            self.check_job_budget()
        except GatewayError:
            return False
        return True

    def check_job_budget(self) -> None:
        self.check_not_finished()
        if self._job_timeout_s is None or self._job_start is None:
            return
        if self._clock() - self._job_start > self._job_timeout_s:
            raise GatewayError(ErrorCode.JOB_TIMEOUT, "job budget exhausted")


__all__ = ["EnforcementMode", "Episode", "Referee"]
