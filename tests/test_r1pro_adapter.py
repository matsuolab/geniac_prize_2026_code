"""R1Pro adapter: every sim call runs on one thread, whichever thread asks, and a
control cycle advances the BEHAVIOR sim exactly once."""

from __future__ import annotations

import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from test_r1pro_low_level import _env, r1pro_b1k  # noqa: F401  — fakes of OmniGibson

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.r1pro_adapter import R1ProLowLevelEnv
from capx.protocol import EnvClient, InProcessTransport, TaskSpec
from capx.protocol.actions import R1ProRawAction
from capx.protocol.taskpack import SeedPolicy


class ThreadRecordingBase:
    def __init__(self) -> None:
        self.threads: list[int] = []
        self.calls: list[str] = []

    def _record(self, call: str) -> None:
        self.threads.append(threading.get_ident())
        self.calls.append(call)

    def reset(self, options: dict[str, int]) -> None:
        self._record("reset")

    def step(self, command: object) -> None:
        self._record("step")

    def compute_reward(self) -> float:
        self._record("compute_reward")
        return 0.5

    def task_completed(self) -> bool:
        self._record("task_completed")
        return True


def drive(env: R1ProLowLevelEnv, seed: int) -> None:
    env.reset(seed=seed)
    env.compute_reward()


def test_calls_from_different_connections_share_one_sim_thread() -> None:
    # EnvBackendServer handles each gateway connection on its own thread, while Kit
    # only answers the thread that launched it.
    base = ThreadRecordingBase()
    env = R1ProLowLevelEnv(base)

    with ThreadPoolExecutor(max_workers=1) as first, ThreadPoolExecutor(max_workers=1) as second:
        first.submit(drive, env, 0).result()
        second.submit(drive, env, 1).result()

    assert len(base.threads) == 4
    assert len(set(base.threads)) == 1
    assert base.threads[0] != threading.get_ident()


def test_results_come_back_from_the_sim_thread() -> None:
    env = R1ProLowLevelEnv(ThreadRecordingBase())

    assert env.compute_reward() == 0.5


def test_a_cycle_checks_success_right_after_its_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Success is part of the cycle, on the sim thread, so the gateway sees a success
    reached partway through a ``repeat``."""
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(as_tensor=lambda x: x))
    base = ThreadRecordingBase()
    env = R1ProLowLevelEnv(base)

    outcome = env.apply_raw_action(R1ProRawAction(action=np.zeros(25)))

    assert outcome.success is True
    assert base.calls == ["step", "task_completed"]
    assert len(set(base.threads)) == 1
    assert base.threads[0] != threading.get_ident()


# --- the real BEHAVIOR env under the adapter -----------------------------------


class ObservedR1Pro(R1ProLowLevelEnv):
    """The real adapter's cycles over the real env; only the observations stand in,
    since they read OmniGibson's robot and cameras."""

    sim_time = 0.0

    def get_proprio(self) -> dict[str, np.ndarray]:
        return {
            "robot_joint_pos": np.zeros(28, dtype=np.float32),
            "robot_cartesian_pos": np.zeros(14, dtype=np.float32),
        }

    def get_observation(self) -> dict[str, np.ndarray]:
        return self.get_proprio()


def _behavior_gateway(env: object) -> Gateway:
    class Factory:
        supports_parallel = False

        def build(self, spec: TaskSpec) -> ObservedR1Pro:
            return ObservedR1Pro(env)

    task = TaskSpec(
        task_id="t",
        embodiment="r1pro",
        simulator="behavior",
        instruction="do it",
        env_target="fake",
        n_trials=1,
        trial_steplimit=100,
        seed_policy=SeedPolicy(mode="fixed", seeds=[0]),
    )
    return Gateway({"t": task}, Factory(), EnforcementMode.OFFICIAL)


def test_a_behavior_cycle_advances_the_sim_exactly_once(r1pro_b1k) -> None:  # noqa: F811
    """Success is checked after every cycle, so checking it must not step the sim:
    repeat=5 is five simulator steps, and a success reached at the second one
    counts although the third undoes it."""
    env = _env(r1pro_b1k)
    og_env, og_step = env.env, env.env.step

    def step(action: object) -> object:
        og_env.next_success = og_env.steps + 1 == 2
        return og_step(action)

    og_env.step = step
    gateway = _behavior_gateway(env)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    stepped_by_reset = og_env.steps

    client.step(reset.episode_id, R1ProRawAction(action=np.zeros(25), repeat=5))

    assert og_env.steps - stepped_by_reset == 5
    assert gateway.aggregate()["t"]["successes"] == 1
