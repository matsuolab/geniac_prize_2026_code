"""Internal scoring and the leaderboard built from it. Never crosses the RPC boundary.

Success is sticky within an episode (solved-then-moved-on still counts), and a voided
episode (``env_crash``) is dropped. ``reward`` is the env's reward at the last
executed step, not a sum or peak; ``mean_reward`` is a diagnostic. The official score
is built from these by the eval queue (``capx.evalqueue.leaderboard.score_result``),
which can weigh each simulator's success rate with the pack's ``score_weights``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from capx.protocol.taskpack import Visibility


@dataclass
class EpisodeScore:
    task_id: str
    reward: float = 0.0
    success: bool = False


class Scoring:
    def __init__(self) -> None:
        self._scores: dict[str, EpisodeScore] = {}

    def record(
        self, episode_id: str, task_id: str, *, reward: float, success: bool = False
    ) -> None:
        score = self._score(episode_id, task_id)
        score.reward = reward
        score.success = score.success or success  # sticky

    def record_success(self, episode_id: str, task_id: str) -> None:
        self._score(episode_id, task_id).success = True

    def _score(self, episode_id: str, task_id: str) -> EpisodeScore:
        return self._scores.setdefault(episode_id, EpisodeScore(task_id))

    def succeeded(self, episode_id: str) -> bool:
        score = self._scores.get(episode_id)
        return score is not None and score.success

    def drop(self, episode_id: str) -> None:
        self._scores.pop(episode_id, None)

    def episodes(self) -> list[EpisodeScore]:
        return list(self._scores.values())

    def aggregate(self, trials: Mapping[str, int]) -> dict[str, dict[str, float]]:
        """Per-task trials, successes, success rate and mean reward over every task in
        ``trials``.

        ``trials`` maps each task to the episodes it is divided by at the least: its
        trial budget in official mode, 0 in dev. A trial the submission never ran
        counts as a failure, so stopping after one success does not score 1.0."""
        by_task: dict[str, list[EpisodeScore]] = {task_id: [] for task_id in trials}
        # A copy: lanes record scores while the results dump aggregates.
        for score in list(self._scores.values()):
            by_task.setdefault(score.task_id, []).append(score)
        aggregate: dict[str, dict[str, float]] = {}
        for task_id, group in by_task.items():
            denominator = max(len(group), trials.get(task_id, 0), 1)
            successes = sum(s.success for s in group)
            aggregate[task_id] = {
                "episodes": len(group),
                "trials": denominator,
                "successes": successes,
                "success_rate": successes / denominator,
                "mean_reward": sum(s.reward for s in group) / denominator,
            }
        return aggregate


def build_leaderboard(
    aggregate: dict[str, dict[str, float]], visibility: Visibility
) -> dict[str, object]:
    """The leaderboard view of ``aggregate``: successful trials over every trial of the
    pack, so a task weighs as many trials as it has. A private pack exposes only each
    task's success rate, hiding reward shaping and per-task counts; the totals stay,
    because the eval queue adds them up across subsets."""
    if visibility is Visibility.PRIVATE:
        tasks: dict[str, object] = {
            task_id: {"success_rate": metrics["success_rate"]}
            for task_id, metrics in aggregate.items()
        }
    else:
        tasks = dict(aggregate)

    successes = int(sum(m["successes"] for m in aggregate.values()))
    trials = int(sum(m["trials"] for m in aggregate.values()))
    return {
        "visibility": visibility.value,
        "overall_success_rate": successes / trials if trials else 0.0,
        "successes": successes,
        "trials": trials,
        "tasks": tasks,
    }


__all__ = ["EpisodeScore", "Scoring", "build_leaderboard"]
