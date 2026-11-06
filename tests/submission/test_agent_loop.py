"""``capx.agent_client.agent_loop.run_eval``, the loop ``docker/run_submission.py`` runs."""

from __future__ import annotations

from typing import Any

from capx.agent_client import HoldPolicy, run_eval
from capx.agent_client.agent_loop import MAX_VOID_RETRIES
from capx.protocol import ErrorCode, GatewayError, StepResult
from tests.submission.fakes import FakeEnvClient, _default_manifest


def _one_task(n_trials: int) -> list[dict[str, Any]]:
    entry = next(e for e in _default_manifest() if e["task_id"] == "stack_cubes")
    return [{**entry, "n_trials": n_trials}]


def _hold(_embodiment: str) -> HoldPolicy:
    return HoldPolicy()


class _CrashingClient(FakeEnvClient):
    def step(self, episode_id: str, action: Any) -> StepResult:
        raise GatewayError(ErrorCode.ENV_CRASH, "environment crashed; episode voided")


def test_a_voided_trial_is_run_again_and_the_job_finished() -> None:
    """The gateway refunds a voided trial; the loop used to count it as done anyway."""
    client = FakeEnvClient(_one_task(2))
    client.fail_at(1, GatewayError(ErrorCode.ENV_CRASH, "environment crashed; episode voided"))
    run_eval(client, _hold, max_steps=2)
    assert client._trials_used["stack_cubes"] == 3
    assert client.finished


def test_a_trial_that_keeps_crashing_is_given_up() -> None:
    client = _CrashingClient(_one_task(2))
    run_eval(client, _hold, max_steps=2)
    assert client._trials_used["stack_cubes"] == 2 * (1 + MAX_VOID_RETRIES)
    assert client.finished


def test_a_job_timeout_ends_the_run_without_finishing() -> None:
    client = FakeEnvClient(_one_task(3))
    client.fail_at(0, GatewayError(ErrorCode.JOB_TIMEOUT, "job budget exhausted"))
    run_eval(client, _hold, max_steps=2)
    assert client._trials_used == {}
    assert not client.finished
