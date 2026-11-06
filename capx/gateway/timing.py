"""Where an episode's wall clock goes, measured gateway-side:

    sim_s     inside ``apply_raw_action`` / reward / success
    obs_s     building observations (where rendering lands)
    client_s  everything else: the submission's own work plus the RPC round trip

Organizer-side only; it reaches an operator through the results dump.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class EpisodeTiming:
    episode_id: str
    task_id: str
    started_at: float
    steps: int = 0
    sim_s: float = 0.0
    observations: int = 0
    obs_s: float = 0.0
    ended_at: float | None = None

    def wall_s(self, now: float) -> float:
        return (self.ended_at if self.ended_at is not None else now) - self.started_at

    def client_s(self, now: float) -> float:
        """Wall clock the simulator did not account for, floored at zero."""
        return max(0.0, self.wall_s(now) - self.sim_s - self.obs_s)

    def as_dict(self, now: float) -> dict[str, float | int | str]:
        wall = self.wall_s(now)
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "wall_s": round(wall, 3),
            "sim_s": round(self.sim_s, 3),
            "obs_s": round(self.obs_s, 3),
            "client_s": round(self.client_s(now), 3),
            "steps": self.steps,
            "observations": self.observations,
        }


class Timings:
    """Per-episode timing, in start order."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._episodes: dict[str, EpisodeTiming] = {}
        self._lock = threading.Lock()

    def begin(self, episode_id: str, task_id: str) -> None:
        with self._lock:
            self._episodes[episode_id] = EpisodeTiming(
                episode_id=episode_id, task_id=task_id, started_at=self._clock()
            )

    def end(self, episode_id: str) -> None:
        with self._lock:
            timing = self._episodes.get(episode_id)
            if timing is not None and timing.ended_at is None:
                timing.ended_at = self._clock()

    def drop(self, episode_id: str) -> None:
        """Forget a voided episode, matching ``Scoring.drop``."""
        with self._lock:
            self._episodes.pop(episode_id, None)

    def add_sim(self, episode_id: str, seconds: float, steps: int = 1) -> None:
        with self._lock:
            timing = self._episodes.get(episode_id)
            if timing is not None:
                timing.sim_s += seconds
                timing.steps += steps

    def add_obs(self, episode_id: str, seconds: float) -> None:
        with self._lock:
            timing = self._episodes.get(episode_id)
            if timing is not None:
                timing.obs_s += seconds
                timing.observations += 1

    def episodes(self) -> list[EpisodeTiming]:
        with self._lock:
            return list(self._episodes.values())

    def report(self) -> dict[str, object]:
        """Per-episode rows plus a per-task roll-up, ready for the results dump."""
        now = self._clock()
        with self._lock:
            episodes = list(self._episodes.values())

        by_task: dict[str, dict[str, float]] = {}
        for timing in episodes:
            task = by_task.setdefault(
                timing.task_id,
                {"episodes": 0, "wall_s": 0.0, "sim_s": 0.0, "obs_s": 0.0, "client_s": 0.0},
            )
            task["episodes"] += 1
            task["wall_s"] += timing.wall_s(now)
            task["sim_s"] += timing.sim_s
            task["obs_s"] += timing.obs_s
            task["client_s"] += timing.client_s(now)

        for task in by_task.values():
            episodes_run = task["episodes"] or 1
            for key in ("wall_s", "sim_s", "obs_s", "client_s"):
                task[f"mean_{key}"] = round(task[key] / episodes_run, 3)
                task[key] = round(task[key], 3)

        return {
            "episodes": [timing.as_dict(now) for timing in episodes],
            "by_task": by_task,
        }


__all__ = ["EpisodeTiming", "Timings"]
