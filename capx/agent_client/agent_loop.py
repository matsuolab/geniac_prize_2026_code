"""The trial loop: attempt each task's trials over an ``EnvClient``.

A gateway verdict mid-trial (``client_timeout``, ``invalid_action``) ends that one
trial and the loop moves on. ``env_crash`` voids and refunds the trial, so it is run
again; ``job_timeout`` and ``job_finished`` stop the run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np

from capx.protocol import (
    BimanualFrankaRawAction,
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    ProprioObs,
    R1ProRawAction,
    RawAction,
    ResetResult,
)


class Policy(Protocol):
    def reset(self, reset: ResetResult) -> None: ...
    def act(self, obs: ProprioObs | FullObs) -> RawAction: ...


# Builds a fresh Policy for one trial; the argument is the task's embodiment id
# from the manifest ("franka" | "r1pro" | ...), so one factory can serve a pack
# that mixes embodiments.
PolicyFactory = Callable[[str], Policy]


class HoldPolicy:
    """Trivial baseline: hold the reset pose. Exercises the loop without solving anything."""

    def __init__(self) -> None:
        self._joints = np.zeros(7)

    def reset(self, reset: ResetResult) -> None:
        self._joints = np.asarray(reset.obs.proprio.joint_pos[:7])

    def act(self, obs: ProprioObs | FullObs) -> RawAction:
        return FrankaRawAction(joints=self._joints, gripper=1.0)


class BimanualHoldPolicy:
    """``HoldPolicy`` for two Franka arms: hold both reset poses, grippers open."""

    def __init__(self) -> None:
        self._joints = np.zeros((2, 7))

    def reset(self, reset: ResetResult) -> None:
        arms = reset.obs.proprio.arms
        if len(arms) != 2:
            raise ValueError(f"a bimanual reset reports {len(arms)} arms, not 2")
        self._joints = np.stack([np.asarray(arm.joint_pos[:7]) for arm in arms])

    def act(self, obs: ProprioObs | FullObs) -> RawAction:
        return BimanualFrankaRawAction(joints=self._joints, grippers=np.ones(2))


class R1ProZeroPolicy:
    """Trivial BEHAVIOR baseline: an all-zero r1pro control vector every step.

    ``action_dim`` comes from ``EvalConfig.embodiments.r1pro``; unlike Franka's 7
    joints it is not a constant of the protocol.
    """

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim

    def reset(self, reset: ResetResult) -> None:
        pass

    def act(self, obs: ProprioObs | FullObs) -> RawAction:
        return R1ProRawAction(action=np.zeros(self._action_dim, dtype=np.float32))


def run_episode(client: EnvClient, reset: ResetResult, policy: Policy, *, max_steps: int) -> None:
    policy.reset(reset)
    obs: ProprioObs | FullObs = reset.obs
    for _ in range(max_steps):
        result = client.step(reset.episode_id, policy.act(obs))
        obs = result.obs
        if result.terminated or result.truncated:
            break


_logger = logging.getLogger(__name__)

# How often one trial is run again after the gateway voided it. A refund costs no
# trial, only job time; the bound keeps a reliably crashing task from eating it all.
MAX_VOID_RETRIES = 2


def _run_trial(client: EnvClient, task_id: str, policy: Policy, *, max_steps: int) -> ErrorCode | None:
    """One trial; the gateway's verdict if it ended the trial early."""
    try:
        run_episode(client, client.reset(task_id), policy, max_steps=max_steps)
    except GatewayError as exc:
        return exc.code
    return None


def run_eval(
    client: EnvClient,
    policy_factory: PolicyFactory,
    *,
    max_trials_per_task: int | None = None,
    max_steps: int | None = None,
) -> None:
    """Attempt every manifest task, then ``finish()`` the job.

    ``None`` takes the manifest's ``n_trials`` and ``trial_steplimit``; a number caps
    the run below them, e.g. for a quick check.
    """
    for entry in client.manifest():
        task_id = entry["task_id"]
        steps = max_steps or int(entry["trial_steplimit"])
        trials = max_trials_per_task or int(entry["n_trials"])
        done = retries = 0
        while done < trials:
            code = _run_trial(client, task_id, policy_factory(entry["embodiment"]), max_steps=steps)
            if code in (ErrorCode.JOB_TIMEOUT, ErrorCode.JOB_FINISHED):
                return
            if code is ErrorCode.TRIALS_EXHAUSTED:
                break
            if code is ErrorCode.ENV_CRASH and retries < MAX_VOID_RETRIES:
                retries += 1
                _logger.warning("trial of %s voided by env_crash; running it again", task_id)
                continue
            if code is not None:
                _logger.warning("trial of %s ended by the gateway (%s)", task_id, code.value)
            done += 1
            retries = 0
    client.finish()


def baseline_policy_factory(
    embodiments: set[str], *, r1pro_action_dim: int | None
) -> PolicyFactory:
    """The shipped baseline policy for each embodiment in ``embodiments``.

    The whole set is validated upfront, so a run fails at startup rather than at
    the first trial of an unsupported task.
    """
    unsupported = embodiments - {"franka", "franka_bimanual", "r1pro"}
    if unsupported:
        raise ValueError(f"no baseline policy for embodiment(s) {sorted(unsupported)}")
    if "r1pro" in embodiments and not r1pro_action_dim:
        raise ValueError("r1pro tasks need embodiments.r1pro.action_dim configured")

    def make(embodiment: str) -> Policy:
        if embodiment == "franka":
            return HoldPolicy()
        if embodiment == "franka_bimanual":
            return BimanualHoldPolicy()
        if embodiment == "r1pro":
            assert r1pro_action_dim is not None
            return R1ProZeroPolicy(r1pro_action_dim)
        raise ValueError(f"no baseline policy for embodiment {embodiment!r}")

    return make


__all__ = [
    "MAX_VOID_RETRIES",
    "Policy",
    "PolicyFactory",
    "HoldPolicy",
    "R1ProZeroPolicy",
    "run_episode",
    "run_eval",
    "baseline_policy_factory",
]
