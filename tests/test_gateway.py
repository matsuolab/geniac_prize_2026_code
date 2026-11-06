"""Referee, scoring, gateway, pack loader, leaderboard and the trial loop.

All GPU-free: a fake factory stands in for the real Robosuite env, so the fairness
logic (trials, seeds, wall-clock, crash rollback) and the internal scoring are
exercised without a simulator.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from capx.agent_client import HoldPolicy, R1ProZeroPolicy, baseline_policy_factory, run_eval
from capx.gateway import EnforcementMode, Gateway, Referee, Scoring, StepOutcome
from capx.gateway.scoring import build_leaderboard
from capx.protocol import (
    EnvClient,
    ErrorCode,
    FrankaRawAction,
    FullObs,
    GatewayError,
    InProcessTransport,
    ProprioObs,
)
from capx.protocol.taskpack import (
    SeedPolicy,
    Simulator,
    TaskSpec,
    Visibility,
    import_pack_modules,
    load_pack,
    load_task_packs,
    pack_dirs,
)

CAMERA = "robot0_robotview"
# A joint vector inside the published Panda limits (index 3 must be in [-3.07, -0.07]).
VALID_JOINTS = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
PACK_LIMITS = "limits:\n  trial_time_s: 60\n  n_trials: 1\n  trial_steplimit: 10\n"


def hold_factory(_embodiment: str) -> HoldPolicy:
    """PolicyFactory for the fake (Franka) task sets used across the test files."""
    return HoldPolicy()


class FakeEnv:
    embodiment = "franka"

    def __init__(
        self,
        *,
        crash_on_reset=False,
        crash_at_cycle=None,
        solve_at_cycle=None,
        horizon=None,
    ) -> None:
        self.sim_time = 0.0
        self.closed = False
        self.last_seed: int | None = None
        self.seeds: list[int] = []
        self._cycles = 0
        self._crash_on_reset = crash_on_reset
        self._crash_at = crash_at_cycle
        self._solve_at = solve_at_cycle
        # Robosuite refuses to step past its horizon, with this exact text.
        self._horizon = horizon
        self._joints = np.zeros(7)
        self._gripper = 1.0

    def reset(self, *, seed: int) -> None:
        if self._crash_on_reset:
            raise RuntimeError("boom-reset")
        self.last_seed = seed
        self.seeds.append(seed)
        self._cycles = 0
        self.sim_time = 0.0
        self._joints = VALID_JOINTS.copy()  # a valid home pose, like the real sim

    def apply_raw_action(self, action) -> StepOutcome:
        self._cycles += 1
        self.sim_time += 0.05
        if self._crash_at is not None and self._cycles >= self._crash_at:
            raise RuntimeError("boom-step")
        if self._horizon is not None and self._cycles > self._horizon:
            raise ValueError("executing action in terminated episode")
        self._joints = np.asarray(action.joints)
        self._gripper = float(action.gripper)
        return StepOutcome(success=self._solved())

    def get_proprio(self):
        return {
            "robot_joint_pos": np.concatenate([self._joints, [self._gripper]]).astype(np.float32),
            "robot_cartesian_pos": np.zeros(8, dtype=np.float32),
        }

    def get_observation(self):
        obs = self.get_proprio()
        obs["cube_poses"] = {"primary": np.zeros(7, dtype=np.float32)}  # privileged
        obs[CAMERA] = {
            "images": {"rgb": np.zeros((4, 4, 3), np.uint8), "depth": np.zeros((4, 4), np.float32)},
            "intrinsics": np.eye(3, dtype=np.float32),
            "pose": np.zeros(7, dtype=np.float32),
        }
        return obs

    def compute_reward(self) -> float:
        return 1.0 if self._solved() else 0.1

    def _solved(self) -> bool:
        return self._solve_at is not None and self._cycles >= self._solve_at

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    # Each build() returns a fresh, independent FakeEnv, so concurrent workers
    # don't share state — safe to run with workers>1 (unlike RobosuiteEnvFactory).
    supports_parallel = True

    def __init__(self, **env_kwargs) -> None:
        self._kwargs = env_kwargs
        self.built: list[FakeEnv] = []

    def build(self, spec) -> FakeEnv:
        env = FakeEnv(**self._kwargs)
        self.built.append(env)
        return env


def make_tasks(**overrides) -> dict[str, TaskSpec]:
    fields: dict = dict(
        task_id="t",
        embodiment="franka",
        simulator="robosuite",
        instruction="do it",
        camera_names=[CAMERA],
        env_target="fake",
        n_trials=2,
        trial_steplimit=100,
        seed_policy=SeedPolicy(mode="fixed", seeds=[7, 8]),
    )
    fields.update(overrides)
    return {"t": TaskSpec(**fields)}


def act() -> FrankaRawAction:
    return FrankaRawAction(joints=VALID_JOINTS, gripper=1.0)


# --- Referee ---------------------------------------------------------------


def test_referee_dev_is_unlimited_but_official_caps_trials() -> None:
    tasks = make_tasks()
    dev = Referee(tasks, EnforcementMode.DEV)
    for _ in range(5):
        dev.begin_episode("t")  # never refused

    official = Referee(tasks, EnforcementMode.OFFICIAL)
    official.begin_episode("t")
    official.begin_episode("t")
    with pytest.raises(GatewayError) as exc:
        official.begin_episode("t")
    assert exc.value.code is ErrorCode.TRIALS_EXHAUSTED


def test_referee_uses_fixed_seeds_in_order() -> None:
    ref = Referee(make_tasks(), EnforcementMode.OFFICIAL)
    assert ref.begin_episode("t").seed == 7
    assert ref.begin_episode("t").seed == 8


def test_referee_void_refunds_the_trial() -> None:
    ref = Referee(make_tasks(), EnforcementMode.OFFICIAL)
    ep = ref.begin_episode("t")
    assert ref.trials_used("t") == 1
    ref.void_episode(ep.episode_id)
    assert ref.trials_used("t") == 0


def test_referee_job_timeout() -> None:
    clock = [0.0]
    ref = Referee(make_tasks(), EnforcementMode.DEV, job_timeout_s=1.0, clock=lambda: clock[0])
    ref.start_clock()
    ref.begin_episode("t")
    clock[0] = 2.0
    with pytest.raises(GatewayError) as exc:
        ref.begin_episode("t")
    assert exc.value.code is ErrorCode.JOB_TIMEOUT


def test_the_job_clock_starts_at_first_contact_not_at_startup() -> None:
    """Loading a model before connecting is not charged to the job budget."""
    clock = [0.0]
    started: list[float] = []
    gateway = Gateway(
        make_tasks(),
        FakeFactory(),
        EnforcementMode.OFFICIAL,
        job_timeout_s=1.0,
        clock=lambda: clock[0],
        on_clock_start=lambda: started.append(clock[0]),
    )
    clock[0] = 100.0  # the submission's model server was coming up
    client = EnvClient(InProcessTransport(gateway.acquire()))
    client.manifest()
    client.reset("t")
    assert started == [100.0]  # once, at the first contact

    clock[0] = 102.0
    with pytest.raises(GatewayError) as exc:
        client.reset("t")
    assert exc.value.code is ErrorCode.JOB_TIMEOUT


def test_opening_a_lane_starts_the_job_clock() -> None:
    started: list[bool] = []
    gateway = Gateway(
        make_tasks(),
        FakeFactory(),
        EnforcementMode.DEV,
        max_sessions=2,
        on_clock_start=lambda: started.append(True),
    )
    gateway.acquire()
    gateway.acquire()
    assert started == [True]


# --- Scoring ---------------------------------------------------------------


def test_scoring_success_is_sticky_and_aggregates() -> None:
    scoring = Scoring()
    scoring.record("e1", "t", reward=0.2, success=False)
    scoring.record("e1", "t", reward=1.0, success=True)
    scoring.record("e1", "t", reward=0.3, success=False)  # still counts as success
    scoring.record("e2", "t", reward=0.0, success=False)
    agg = scoring.aggregate({"t": 0})
    assert agg["t"]["episodes"] == 2
    assert agg["t"]["success_rate"] == 0.5


def test_scoring_divides_by_the_trial_budget_when_it_is_larger() -> None:
    scoring = Scoring()
    scoring.record("e1", "t", reward=1.0, success=True)
    agg = scoring.aggregate({"t": 3, "u": 3})
    assert agg["t"] == {
        "episodes": 1,
        "trials": 3,
        "successes": 1,
        "success_rate": pytest.approx(1 / 3),
        "mean_reward": pytest.approx(1 / 3),
    }
    assert agg["u"] == {
        "episodes": 0, "trials": 3, "successes": 0, "success_rate": 0.0, "mean_reward": 0.0
    }


# --- Gateway (via EnvClient) -----------------------------------------------


def client_for(tasks, factory, mode=EnforcementMode.DEV, **kw):
    gateway = Gateway(tasks, factory, mode, **kw)
    return EnvClient(InProcessTransport(gateway.acquire())), gateway


def test_gateway_reset_step_hides_privileged_state() -> None:
    lane = Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV).acquire()
    client = EnvClient(InProcessTransport(lane))
    reset = client.reset("t")
    assert isinstance(reset.obs, FullObs)

    result = client.step(reset.episode_id, act())
    assert isinstance(result.obs, ProprioObs)
    np.testing.assert_allclose(result.obs.joint_pos[:7], VALID_JOINTS, atol=1e-5)

    wire = InProcessTransport(lane).call("reset", {"task_id": "t"})
    assert "cube_poses" not in _flatten_keys(wire)


def test_gateway_reuses_env_across_trials_of_a_task() -> None:
    factory = FakeFactory()
    client, _ = client_for(make_tasks(), factory)
    client.reset("t")
    client.reset("t")
    assert len(factory.built) == 1  # env reused, not rebuilt


def test_gateway_official_refuses_after_trials_exhausted() -> None:
    client, _ = client_for(make_tasks(), FakeFactory(), EnforcementMode.OFFICIAL)
    client.reset("t")
    client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.reset("t")
    assert exc.value.code is ErrorCode.TRIALS_EXHAUSTED


def test_gateway_env_crash_rolls_back_and_rebuilds() -> None:
    factory = FakeFactory(crash_at_cycle=1)
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, act())
    assert exc.value.code is ErrorCode.ENV_CRASH
    assert gateway.trials_used("t") == 0  # refunded
    # The score is dropped, but the task stays in the aggregate at zero: the
    # denominator is the pack, not the set of tasks that produced an episode.
    assert gateway.aggregate() == {
        "t": {"episodes": 0, "trials": 2, "successes": 0, "success_rate": 0.0, "mean_reward": 0.0}
    }

    client.reset("t")  # a fresh env is built after the crash
    assert len(factory.built) == 2


def test_the_steplimit_counts_control_cycles_and_ends_the_trial() -> None:
    factory = FakeFactory()
    client, gateway = client_for(make_tasks(trial_steplimit=5), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    assert client.step(reset.episode_id, act()).truncated is False

    held = FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=10)
    result = client.step(reset.episode_id, held)  # 10 asked, 4 left

    assert result.truncated is True
    assert factory.built[0]._cycles == 5
    assert gateway.trials_used("t") == 1
    assert gateway.aggregate()["t"]["episodes"] == 1


def test_a_step_past_the_steplimit_does_not_reach_the_simulator() -> None:
    factory = FakeFactory()
    client, _ = client_for(make_tasks(trial_steplimit=1), factory)
    reset = client.reset("t")
    client.step(reset.episode_id, act())

    result = client.step(reset.episode_id, act())

    assert result.truncated is True
    assert factory.built[0]._cycles == 1


def test_the_steplimit_restarts_with_each_trial() -> None:
    client, _ = client_for(make_tasks(trial_steplimit=1), FakeFactory())
    first = client.reset("t")
    assert client.step(first.episode_id, act()).truncated is True
    second = client.reset("t")  # same env, reused across trials
    assert client.step(second.episode_id, act()).truncated is True


def test_the_manifest_tells_the_submission_its_limits() -> None:
    client, _ = client_for(make_tasks(trial_steplimit=5), FakeFactory())
    (entry,) = client.manifest()
    assert entry["n_trials"] == 2
    assert entry["trial_steplimit"] == 5


def test_gateway_scores_success_internally() -> None:
    client, gateway = client_for(make_tasks(), FakeFactory(solve_at_cycle=2))
    reset = client.reset("t")
    for _ in range(3):
        result = client.step(reset.episode_id, act())
    assert gateway.aggregate()["t"]["success_rate"] == 1.0
    # success never leaks: terminated stays False even after solving
    assert result.terminated is False


class FleetingSuccessEnv(FakeEnv):
    """The cube is lifted at cycle 2 and slips at cycle 3."""

    def _solved(self) -> bool:
        return self._cycles == 2


class FleetingSuccessFactory(FakeFactory):
    def build(self, spec) -> FakeEnv:
        env = FleetingSuccessEnv()
        self.built.append(env)
        return env


def test_a_success_inside_a_repeat_counts() -> None:
    """Achieving the task once during the trial is a success (PRELIM_RULEBOOK.md §4.5),
    so one step of repeat=3 must score what three steps of repeat=1 would."""
    client, gateway = client_for(make_tasks(), FleetingSuccessFactory(), EnforcementMode.OFFICIAL)
    reset = client.reset("t")

    client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=3))

    assert gateway.aggregate()["t"]["successes"] == 1


class HangsAfterSuccessEnv(FakeEnv):
    """Succeeds at cycle 2; cycle 3 then hangs (a stuck backend call) until released."""

    def __init__(self) -> None:
        super().__init__(solve_at_cycle=2)
        self.hanging = threading.Event()
        self.release = threading.Event()

    def apply_raw_action(self, action) -> StepOutcome:
        if self._cycles == 2:
            self.hanging.set()
            self.release.wait(5)
        return super().apply_raw_action(action)


def test_a_success_is_scored_before_the_rest_of_its_step_runs(tmp_path) -> None:
    """The final dump waits only so long for a step in flight; a success reached
    before the step's later cycles hung must already be in it."""
    import json

    from capx.gateway.results_dump import dump_once

    env = HangsAfterSuccessEnv()

    class Factory(FakeFactory):
        def build(self, spec) -> FakeEnv:
            return env

    client, gateway = client_for(make_tasks(), Factory(), EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    stepping = threading.Thread(
        target=client.step,
        args=(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=5)),
    )
    stepping.start()
    try:
        assert env.hanging.wait(5)
        assert not gateway.wait_idle(0.05)  # what the final dump gives up on

        path = tmp_path / "results.json"
        dump_once(gateway, Visibility.PUBLIC, str(path))
        assert json.loads(path.read_text())["leaderboard"]["successes"] == 1
    finally:
        env.release.set()
        stepping.join(5)


class EndsAtCycleEnv(FakeEnv):
    def apply_raw_action(self, action) -> StepOutcome:
        super().apply_raw_action(action)
        return StepOutcome(truncated=self._cycles >= 2)


class EndsAtCycleFactory(FakeFactory):
    def build(self, spec) -> FakeEnv:
        env = EndsAtCycleEnv()
        self.built.append(env)
        return env


def test_an_env_that_ends_the_episode_stops_the_repeat() -> None:
    factory = EndsAtCycleFactory()
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")

    result = client.step(
        reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=5)
    )
    after = client.step(reset.episode_id, act())

    assert result.truncated is True
    assert after.truncated is True
    assert factory.built[0]._cycles == 2  # neither the repeat nor the next step went on
    assert gateway.trials_used("t") == 1


class RenderFailsEnv(FakeEnv):
    """Steps fine, but building an observation fails once ``broken`` (a lost render
    context, or a backend that died)."""

    broken = False

    def get_observation(self):
        if self.broken:
            raise RuntimeError("EGL context lost")
        return super().get_observation()

    def get_proprio(self):
        if self.broken:
            raise RuntimeError("EGL context lost")
        return super().get_proprio()


class RenderFailsFactory(FakeFactory):
    def __init__(self, *, broken: bool = False, **env_kwargs) -> None:
        super().__init__(**env_kwargs)
        self._broken = broken

    def build(self, spec) -> FakeEnv:
        env = RenderFailsEnv(**self._kwargs)
        env.broken = self._broken
        self.built.append(env)
        return env


def test_a_failed_reset_observation_is_an_env_crash() -> None:
    factory = RenderFailsFactory(broken=True)
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)

    with pytest.raises(GatewayError) as exc:
        client.reset("t")

    assert exc.value.code is ErrorCode.ENV_CRASH
    assert "EGL" not in exc.value.detail
    assert gateway.trials_used("t") == 0  # refunded
    assert gateway.aggregate()["t"]["episodes"] == 0  # its failure score dropped
    assert gateway.live_episodes() == []
    assert factory.built[0].closed is True  # discarded, so the next reset rebuilds


def test_a_failed_step_observation_is_an_env_crash() -> None:
    factory = RenderFailsFactory(solve_at_cycle=1)
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    factory.built[0].broken = True

    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, act())

    assert exc.value.code is ErrorCode.ENV_CRASH
    assert gateway.trials_used("t") == 0
    assert gateway.aggregate()["t"]["successes"] == 0  # the voided episode's score is gone


def test_a_failed_get_observation_is_an_env_crash() -> None:
    factory = RenderFailsFactory()
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    factory.built[0].broken = True

    with pytest.raises(GatewayError) as exc:
        client.get_observation(reset.episode_id)

    assert exc.value.code is ErrorCode.ENV_CRASH
    assert gateway.trials_used("t") == 0


class BimanualEnvFactory(FakeFactory):
    def build(self, spec) -> FakeEnv:
        env = FakeEnv()
        env.embodiment = "franka_bimanual"
        self.built.append(env)
        return env


def test_an_env_of_the_wrong_embodiment_is_refused_at_build() -> None:
    """A task routed to a simulator whose envs are for another robot (an r1pro task on
    the LIBERO backend, say) is the organizers' mistake; it must not reach the first
    step as an assertion deep in an adapter."""
    factory = BimanualEnvFactory()
    client, gateway = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)

    with pytest.raises(GatewayError) as exc:
        client.reset("t")

    assert exc.value.code is ErrorCode.ENV_CRASH
    assert factory.built[0].closed is True
    assert gateway.trials_used("t") == 0


def test_closing_a_lane_ends_its_episode() -> None:
    """Reconnecting must not pile up open episodes in the referee."""
    gateway = Gateway(make_tasks(), FakeFactory(), EnforcementMode.DEV)
    for _ in range(3):
        lane = gateway.acquire()
        lane.reset("t")
        gateway.release(lane)

    assert gateway.live_episodes() == []
    assert gateway.aggregate()["t"]["episodes"] == 3  # ended, not voided


def test_env_crash_detail_never_carries_sim_internals() -> None:
    client, _ = client_for(make_tasks(), FakeFactory(crash_at_cycle=1), EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, act())
    assert exc.value.code is ErrorCode.ENV_CRASH
    assert "boom-step" not in exc.value.detail  # sim exception text is log-only


def test_reset_crash_detail_never_carries_sim_internals() -> None:
    client, _ = client_for(make_tasks(), FakeFactory(crash_on_reset=True))
    with pytest.raises(GatewayError) as exc:
        client.reset("t")
    assert exc.value.code is ErrorCode.ENV_CRASH
    assert "boom-reset" not in exc.value.detail


def test_crash_refund_replays_the_same_seed_under_random_policy() -> None:
    # A deliberately induced crash must not reroll the hidden seed: the refunded
    # trial index reproduces the identical seed, so seed-shopping is pointless.
    factory = FakeFactory(crash_at_cycle=1)
    client, _ = client_for(
        make_tasks(seed_policy=SeedPolicy(mode="random")), factory, EnforcementMode.OFFICIAL
    )
    reset = client.reset("t")
    first_seed = factory.built[0].last_seed
    with pytest.raises(GatewayError):
        client.step(reset.episode_id, act())
    client.reset("t")  # retry of the refunded trial
    assert factory.built[1].last_seed == first_seed


def test_job_budget_is_enforced_mid_repeat() -> None:
    clock = [0.0]

    class SlowEnv(FakeEnv):
        def apply_raw_action(self, action) -> StepOutcome:
            clock[0] += 5.0  # each cycle burns past the 1.0s job budget
            return super().apply_raw_action(action)

    class SlowFactory(FakeFactory):
        def build(self, spec) -> FakeEnv:
            env = SlowEnv()
            self.built.append(env)
            return env

    factory = SlowFactory()
    client, _ = client_for(
        make_tasks(), factory, EnforcementMode.OFFICIAL, job_timeout_s=1.0, clock=lambda: clock[0]
    )
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=50))
    assert exc.value.code is ErrorCode.JOB_TIMEOUT
    assert factory.built[0]._cycles == 1  # not all 50 cycles ran to completion


def test_franka_bounds_violation_is_invalid_action() -> None:
    client, _ = client_for(make_tasks(), FakeFactory())
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        # joints=0 violates joint 4's published range [-3.07, -0.07]
        client.step(reset.episode_id, FrankaRawAction(joints=np.zeros(7), gripper=1.0))
    assert exc.value.code is ErrorCode.INVALID_ACTION


def test_official_mode_rejects_privileged_tasks() -> None:
    with pytest.raises(ValueError, match="privileged"):
        Gateway(make_tasks(privileged=True), FakeFactory(), EnforcementMode.OFFICIAL)


def test_reset_crash_keeps_the_episode_ledger_bounded() -> None:
    class CrashOnSecondResetEnv(FakeEnv):
        def reset(self, *, seed: int) -> None:
            if self.seeds:
                raise RuntimeError("boom-second-reset")
            super().reset(seed=seed)

    class Factory(FakeFactory):
        def build(self, spec) -> FakeEnv:
            env = CrashOnSecondResetEnv()
            self.built.append(env)
            return env

    client, gateway = client_for(make_tasks(), Factory())
    client.reset("t")
    with pytest.raises(GatewayError):
        client.reset("t")  # prior episode must be retired even on this crash path
    assert gateway.live_episodes() == []


# --- finish ----------------------------------------------------------------


def test_finish_from_one_lane_ends_the_job_for_every_lane() -> None:
    finished: list[bool] = []
    gateway = Gateway(
        make_tasks(),
        FakeFactory(),
        EnforcementMode.OFFICIAL,
        max_sessions=2,
        on_finish=lambda: finished.append(True),
    )
    one, other = (EnvClient(InProcessTransport(gateway.acquire())) for _ in range(2))
    reset = one.reset("t")
    assert gateway.finished_at() is None
    before = time.time()
    other.finish()
    taken = gateway.finished_at()
    other.finish()  # the second one does nothing
    assert taken is not None and before <= taken <= time.time()
    assert gateway.finished_at() == taken
    calls = (
        lambda: one.reset("t"),
        lambda: one.step(reset.episode_id, act()),
        lambda: one.get_observation(reset.episode_id),
    )
    for call in calls:
        with pytest.raises(GatewayError) as exc:
            call()
        assert exc.value.code is ErrorCode.JOB_FINISHED
    assert finished == [True]
    one.close()  # a lane can still disconnect


def test_finish_keeps_the_score_and_fails_the_trials_never_run() -> None:
    client, gateway = client_for(
        make_tasks(), FakeFactory(solve_at_cycle=1), EnforcementMode.OFFICIAL
    )
    reset = client.reset("t")
    client.step(reset.episode_id, act())
    client.finish()
    task = gateway.aggregate()["t"]
    assert (task["successes"], task["trials"]) == (1, 2)


def test_a_step_cut_short_by_finish_scores_the_cycles_it_ran() -> None:
    factory = FakeFactory(solve_at_cycle=2)
    gateway = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL, max_sessions=2)
    client, other = (EnvClient(InProcessTransport(gateway.acquire())) for _ in range(2))
    reset = client.reset("t")
    env = factory.built[0]
    apply = env.apply_raw_action

    def apply_then_finish(action):
        outcome = apply(action)
        if env._cycles == 2:
            other.finish()
        return outcome

    env.apply_raw_action = apply_then_finish
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, repeat=10))
    assert exc.value.code is ErrorCode.JOB_FINISHED
    assert env._cycles == 2
    assert gateway.aggregate()["t"]["successes"] == 1
    assert gateway.timings()["episodes"][0]["steps"] == 2


def test_the_organizer_ending_the_job_scores_nothing_after_it() -> None:
    """How the eval queue stops a submission over its storage limit (the gateway ends
    the job on SIGTERM): a step in flight stops at its next control cycle, a success it
    would have reached later is not scored, and later calls are refused. It is not the
    submission's finish(), so on_finish does not run."""
    finished: list[bool] = []
    factory = FakeFactory(solve_at_cycle=3)
    gateway = Gateway(
        make_tasks(),
        factory,
        EnforcementMode.OFFICIAL,
        on_finish=lambda: finished.append(True),
    )
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    env = factory.built[0]
    apply = env.apply_raw_action

    def apply_then_end(action):
        outcome = apply(action)
        if env._cycles == 2:
            gateway.end_job()
        return outcome

    env.apply_raw_action = apply_then_end
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, repeat=10))
    assert exc.value.code is ErrorCode.JOB_FINISHED
    assert env._cycles == 2
    assert gateway.aggregate()["t"]["successes"] == 0
    with pytest.raises(GatewayError) as exc:
        client.reset("t")
    assert exc.value.code is ErrorCode.JOB_FINISHED
    assert finished == [] and gateway.finished_at() is None


def test_the_gateway_can_wait_for_a_step_in_flight_to_record_its_score() -> None:
    factory = FakeFactory(solve_at_cycle=1)
    gateway = Gateway(make_tasks(), factory, EnforcementMode.OFFICIAL)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    env = factory.built[0]
    apply, inside, release = env.apply_raw_action, threading.Event(), threading.Event()

    def slow_apply(action):
        inside.set()
        release.wait(5)
        return apply(action)

    env.apply_raw_action = slow_apply
    stepping = threading.Thread(target=client.step, args=(reset.episode_id, act()))
    stepping.start()
    assert inside.wait(5)
    assert not gateway.wait_idle(0.05)
    release.set()
    assert gateway.wait_idle(5)
    assert gateway.aggregate()["t"]["successes"] == 1
    stepping.join()


# --- Pack loader -----------------------------------------------------------


def test_load_task_packs(tmp_path) -> None:
    (tmp_path / "pack.yaml").write_text(
        f"pack_id: p\nvisibility: public\n{PACK_LIMITS}tasks:\n"
        "  - task_id: a\n    embodiment: franka\n    simulator: robosuite\n"
        "    instruction: go\n    env_target: fake\n"
        "  - task_id: b\n    embodiment: franka\n    simulator: robosuite\n"
        "    instruction: go\n    env_target: fake\n    trial_steplimit: 3\n"
    )
    tasks = load_task_packs([str(tmp_path)])
    assert set(tasks) == {"a", "b"}
    assert tasks["a"].embodiment.value == "franka"
    assert (tasks["a"].n_trials, tasks["a"].trial_steplimit) == (1, 10)  # the pack's
    assert tasks["b"].trial_steplimit == 3  # its own


def test_a_pack_must_set_its_limits(tmp_path) -> None:
    (tmp_path / "pack.yaml").write_text("pack_id: p\ntasks: []\n")
    with pytest.raises(ValueError, match="limits"):
        load_task_packs([str(tmp_path)])


def test_a_task_cannot_set_its_own_horizon() -> None:
    with pytest.raises(ValueError, match="trial_steplimit"):
        make_tasks(env_kwargs={"max_steps": 9})


def test_a_pack_cannot_repeat_a_task_id(tmp_path) -> None:
    task = "  - task_id: a\n    embodiment: franka\n    simulator: robosuite\n    instruction: go\n    env_target: fake\n"
    (tmp_path / "pack.yaml").write_text(f"pack_id: p\n{PACK_LIMITS}tasks:\n{task}{task}")
    with pytest.raises(ValueError, match="repeats task ids"):
        load_task_packs([str(tmp_path)])


def test_load_task_packs_rejects_duplicate_ids(tmp_path) -> None:
    body = (
        "pack_id: {pid}\n" + PACK_LIMITS + "tasks:\n  - task_id: dup\n    embodiment: franka\n"
        "    simulator: robosuite\n    instruction: go\n    env_target: fake\n"
    )
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / "pack.yaml").write_text(body.format(pid="one"))
    (tmp_path / "two" / "pack.yaml").write_text(body.format(pid="two"))
    with pytest.raises(ValueError, match="duplicate"):
        load_task_packs([str(tmp_path / "one"), str(tmp_path / "two")])


def test_load_task_packs_rejects_unknown_keys(tmp_path) -> None:
    """A key the schema does not know would otherwise be dropped without a word."""
    (tmp_path / "pack.yaml").write_text(
        f"pack_id: p\n{PACK_LIMITS}tasks:\n"
        "  - task_id: a\n    embodiment: franka\n    simulator: robosuite\n"
        "    instruction: go\n    env_target: fake\n    success_fn: mine.check\n"
    )
    with pytest.raises(ValueError, match="success_fn"):
        load_task_packs([str(tmp_path)])


def _pack_with_env_module(root, module_name: str) -> None:
    (root / f"{module_name}.py").write_text(
        "from capx.envs.base import register_env\n"
        "BUILT = []\n"
        "def _make(**kwargs):\n"
        "    BUILT.append(kwargs)\n"
        "    return kwargs\n"
        f"register_env({module_name!r}, _make)\n"
    )
    (root / "pack.yaml").write_text(
        f"pack_id: p\n{PACK_LIMITS}register_modules: {{robosuite: [{module_name}]}}\ntasks:\n"
        "  - task_id: a\n    embodiment: franka\n    simulator: robosuite\n"
        f"    instruction: go\n    env_target: {module_name}\n"
        "    env_kwargs: {variation: swap}\n"
    )


def test_pack_env_code_is_imported_only_where_envs_are_built(tmp_path) -> None:
    """The gateway image has no simulator, so reading a pack must not import its code."""
    import sys

    from capx.envs.base import list_envs
    from capx.gateway.env_build import build_low_level_env

    _pack_with_env_module(tmp_path, "pack_env_under_test")
    try:
        tasks = load_task_packs([str(tmp_path)])
        assert "pack_env_under_test" not in sys.modules
        assert "pack_env_under_test" not in list_envs()

        import_pack_modules(tmp_path, Simulator.LIBERO)
        assert "pack_env_under_test" not in list_envs()

        import_pack_modules(tmp_path, Simulator.ROBOSUITE)
        built = build_low_level_env(tasks["a"], enable_render=False)
        assert built["variation"] == "swap"
        assert built["enable_render"] is False
    finally:
        from capx.envs import base

        base._ENV_FACTORIES.pop("pack_env_under_test", None)
        sys.modules.pop("pack_env_under_test", None)
        sys.path.remove(str(tmp_path.resolve()))


class SettlingEnv(FakeEnv):
    """Like the Robosuite and LIBERO envs: reset steps the simulator to let the scene
    settle, and those steps count against its horizon."""

    SETTLE_STEPS = 3

    def reset(self, *, seed: int) -> None:
        super().reset(seed=seed)
        self._cycles = self.SETTLE_STEPS


def test_settling_steps_in_reset_do_not_shorten_the_trial() -> None:
    """The published trial_steplimit is the trial's length, whatever the env's reset
    spent of its horizon. Before the fix, nut assembly's 1000-step trials ended at 900."""
    from capx.envs import base
    from capx.gateway.env_build import build_low_level_env

    def make(*, max_steps: int = 1500, **kwargs) -> SettlingEnv:
        return SettlingEnv(horizon=max_steps)

    class EnvBuildFactory:
        supports_parallel = False

        def build(self, spec):
            return build_low_level_env(spec)

    base.register_env("settling_env_under_test", make)
    try:
        client, gateway = client_for(
            make_tasks(env_target="settling_env_under_test", trial_steplimit=5),
            EnvBuildFactory(),
            EnforcementMode.OFFICIAL,
        )
        reset = client.reset("t")
        result = client.step(
            reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=5)
        )
    finally:
        base._ENV_FACTORIES.pop("settling_env_under_test", None)

    assert result.truncated is True
    assert gateway.timings()["episodes"][0]["steps"] == 5  # the whole steplimit ran
    assert gateway.trials_used("t") == 1


def _extending_pack(
    root, task_ids: list[str], extra: str = "", limits: str = PACK_LIMITS
) -> None:
    (root / "middle").mkdir()
    (root / "middle" / "pack.yaml").write_text(
        f"pack_id: middle\n{limits}extends: final\ntask_ids: {task_ids}\n{extra}"
    )


def test_a_pack_can_select_tasks_from_a_sibling_pack(tmp_path) -> None:
    import sys

    from capx.envs import base

    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_extended")
    final = (tmp_path / "final" / "pack.yaml").read_text()
    second = final[final.index("  - task_id: a") :].replace("task_id: a", "task_id: b")
    (tmp_path / "final" / "pack.yaml").write_text(
        final.replace("pack_id: p\n", "pack_id: final\nvisibility: private\n") + second
    )
    _extending_pack(tmp_path, ["b"])
    try:
        manifest, code_paths = load_pack(tmp_path / "middle")
        assert [t.task_id for t in manifest.tasks] == ["b"]
        assert manifest.visibility is Visibility.PRIVATE
        assert code_paths == [(tmp_path / "final").resolve()]
        assert pack_dirs(tmp_path / "middle") == [
            (tmp_path / "middle").resolve(),
            (tmp_path / "final").resolve(),
        ]

        import_pack_modules(tmp_path / "middle", Simulator.ROBOSUITE)
        assert "pack_env_extended" in base.list_envs()
    finally:
        base._ENV_FACTORIES.pop("pack_env_extended", None)
        sys.modules.pop("pack_env_extended", None)
        sys.path.remove(str((tmp_path / "final").resolve()))


def test_a_pack_imports_env_code_from_its_code_dirs(tmp_path) -> None:
    import sys

    from capx.envs import base

    (tmp_path / "shared").mkdir()
    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_shared")
    (tmp_path / "final" / "pack_env_shared.py").rename(tmp_path / "shared" / "pack_env_shared.py")
    pack = tmp_path / "final" / "pack.yaml"
    pack.write_text(pack.read_text().replace("tasks:\n", "code_dirs: [shared]\ntasks:\n", 1))
    _extending_pack(tmp_path, ["a"])
    try:
        assert pack_dirs(tmp_path / "middle") == [
            (tmp_path / "middle").resolve(),
            (tmp_path / "final").resolve(),
            (tmp_path / "shared").resolve(),
        ]
        import_pack_modules(tmp_path / "middle", Simulator.ROBOSUITE)
        assert "pack_env_shared" in base.list_envs()
    finally:
        base._ENV_FACTORIES.pop("pack_env_shared", None)
        sys.modules.pop("pack_env_shared", None)
        for name in ("final", "shared"):
            sys.path.remove(str((tmp_path / name).resolve()))


def test_an_extending_pack_must_name_tasks_its_base_has(tmp_path) -> None:
    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_unused")
    _extending_pack(tmp_path, ["nope"])
    with pytest.raises(ValueError, match="nope"):
        load_pack(tmp_path / "middle")


def test_an_extending_pack_applies_its_own_limits(tmp_path) -> None:
    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_unused")
    pack = tmp_path / "final" / "pack.yaml"
    second = pack.read_text().split("tasks:\n", 1)[1].replace("task_id: a", "task_id: b")
    pack.write_text(pack.read_text() + second + "    trial_steplimit: 3\n")
    _extending_pack(
        tmp_path,
        ["a", "b"],
        limits="limits:\n  trial_time_s: 5\n  n_trials: 2\n  trial_steplimit: 20\n",
    )
    manifest, _ = load_pack(tmp_path / "middle")
    assert manifest.limits.trial_time_s == 5
    a, b = manifest.tasks
    assert (a.n_trials, a.trial_steplimit) == (2, 20)  # the extending pack's defaults
    assert b.trial_steplimit == 3  # the task's own value travels with it


def test_an_extending_pack_must_set_its_limits(tmp_path) -> None:
    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_unused")
    _extending_pack(tmp_path, ["a"], limits="")
    with pytest.raises(ValueError, match="limits"):
        load_pack(tmp_path / "middle")


def test_an_extending_pack_cannot_add_tasks_of_its_own(tmp_path) -> None:
    (tmp_path / "final").mkdir()
    _pack_with_env_module(tmp_path / "final", "pack_env_unused")
    _extending_pack(
        tmp_path,
        ["a"],
        "tasks:\n  - task_id: c\n    embodiment: franka\n    simulator: robosuite\n"
        "    instruction: go\n    env_target: fake\n",
    )
    with pytest.raises(ValueError, match="task_ids only"):
        load_pack(tmp_path / "middle")


def test_the_original_pack_loads() -> None:
    assert load_task_packs(["taskpacks/original"])


def test_a_pack_may_route_tasks_to_a_simulator_id_of_its_own() -> None:
    assert make_tasks(simulator="libero_variant")["t"].simulator == "libero_variant"
    with pytest.raises(ValueError):
        make_tasks(simulator="Bad Id")


# --- Leaderboard -----------------------------------------------------------


def test_leaderboard_private_hides_reward_detail() -> None:
    agg = {"t": {"episodes": 2, "trials": 2, "successes": 1, "success_rate": 0.5, "mean_reward": 0.7}}
    public = build_leaderboard(agg, Visibility.PUBLIC)
    private = build_leaderboard(agg, Visibility.PRIVATE)
    assert public["tasks"]["t"] == agg["t"]
    assert private["tasks"]["t"] == {"success_rate": 0.5}
    assert (private["overall_success_rate"], private["successes"], private["trials"]) == (0.5, 1, 2)


def test_leaderboard_weighs_each_task_by_its_trials() -> None:
    """Successes over every trial: one solved trial of four is 0.25, where the mean of
    the per-task rates would be (1.0 + 0.0) / 2."""
    scoring = Scoring()
    scoring.record("e1", "t", reward=1.0, success=True)
    board = build_leaderboard(scoring.aggregate({"t": 1, "u": 3}), Visibility.PRIVATE)
    assert (board["overall_success_rate"], board["successes"], board["trials"]) == (0.25, 1, 4)


# --- Agent loop end to end -------------------------------------------------


def test_a_submission_cannot_shrink_its_own_denominator() -> None:
    """Attempting one task out of three must not score like a perfect run.

    Which tasks get reset is the submission's choice, so if the denominator came
    from the episodes, picking the easy task would be the winning strategy.
    """
    tasks = make_tasks()
    for extra in ("u", "v"):
        tasks[extra] = tasks["t"].model_copy(update={"task_id": extra})
    client, gateway = client_for(tasks, FakeFactory(solve_at_cycle=1))

    reset = client.reset("t")
    client.step(reset.episode_id, act())

    agg = gateway.aggregate()
    assert set(agg) == {"t", "u", "v"}
    assert agg["t"]["success_rate"] == 1.0
    assert agg["u"] == {
        "episodes": 0, "trials": 1, "successes": 0, "success_rate": 0.0, "mean_reward": 0.0
    }
    board = build_leaderboard(agg, Visibility.PUBLIC)
    assert board["overall_success_rate"] == pytest.approx(1 / 3)


def test_official_trials_left_unrun_count_as_failures() -> None:
    """Stopping after the first success must not score like solving every trial."""
    client, gateway = client_for(
        make_tasks(), FakeFactory(solve_at_cycle=1), EnforcementMode.OFFICIAL
    )
    reset = client.reset("t")  # one of n_trials=2
    client.step(reset.episode_id, act())

    agg = gateway.aggregate()
    assert agg["t"]["episodes"] == 1
    assert agg["t"]["success_rate"] == 0.5


def test_dev_runs_are_scored_over_the_trials_that_ran() -> None:
    client, gateway = client_for(make_tasks(), FakeFactory(solve_at_cycle=1))
    reset = client.reset("t")
    client.step(reset.episode_id, act())
    assert gateway.aggregate()["t"]["success_rate"] == 1.0


def test_run_eval_drives_the_gateway() -> None:
    client, gateway = client_for(make_tasks(), FakeFactory(solve_at_cycle=1))
    run_eval(client, hold_factory, max_trials_per_task=2, max_steps=3)
    assert gateway.aggregate()["t"]["episodes"] == 2


def test_run_eval_absorbs_env_crash_and_finishes() -> None:
    # A mid-episode sim crash voids one trial; the loop must move on, not abort.
    client, gateway = client_for(
        make_tasks(), FakeFactory(crash_at_cycle=1), EnforcementMode.OFFICIAL
    )
    run_eval(client, hold_factory, max_trials_per_task=2, max_steps=3)
    # Every trial voided, so nothing was scored -- and the task is still counted,
    # at zero. An env that crashes for everyone shows up as a column of zeros
    # rather than quietly leaving the leaderboard.
    assert gateway.aggregate() == {
        "t": {"episodes": 0, "trials": 2, "successes": 0, "success_rate": 0.0, "mean_reward": 0.0}
    }
    assert gateway.trials_used("t") == 0


def test_baseline_policy_factory_dispatches_per_embodiment() -> None:
    # A mixed pack (e.g. the public robosuite+libero+behavior pack) gets the
    # baseline matching each task's embodiment.
    factory = baseline_policy_factory({"franka", "r1pro"}, r1pro_action_dim=25)
    assert isinstance(factory("franka"), HoldPolicy)
    assert isinstance(factory("r1pro"), R1ProZeroPolicy)


def test_baseline_policy_factory_validates_upfront() -> None:
    with pytest.raises(ValueError, match="no baseline policy"):
        baseline_policy_factory({"franka", "alien"}, r1pro_action_dim=25)
    with pytest.raises(ValueError, match="action_dim"):
        baseline_policy_factory({"franka", "r1pro"}, r1pro_action_dim=None)


def _flatten_keys(obj: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(str(k))
            keys |= _flatten_keys(v)
    return keys


def test_the_simulator_horizon_ends_the_episode_without_refunding_it() -> None:
    """Robosuite raises when a step arrives past its horizon. The horizon is the
    steplimit, so only an env that ignored it gets here — and it is still the
    episode ending, not the sim breaking.

    Voiding it would refund the trial, so a submission that let a losing episode
    run to the horizon would get the attempt back. Measured before the fix: 73
    episodes of one 1200-episode round vanished this way, 45 of them from a single
    task, and the leaderboard simply had a smaller denominator for it.
    """
    client, gateway = client_for(
        make_tasks(), FakeFactory(horizon=2), EnforcementMode.OFFICIAL
    )
    reset = client.reset("t")
    client.step(reset.episode_id, act())
    client.step(reset.episode_id, act())

    result = client.step(reset.episode_id, act())  # one past the horizon

    assert result.truncated is True
    assert gateway.trials_used("t") == 1  # spent, not refunded
    assert gateway.aggregate()["t"]["episodes"] == 1  # scored, not dropped


def test_a_step_after_the_simulator_horizon_does_not_reach_it() -> None:
    factory = FakeFactory(horizon=1)
    client, _ = client_for(make_tasks(), factory, EnforcementMode.OFFICIAL)
    reset = client.reset("t")
    client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=1.0, repeat=3))
    reached = factory.built[0]._cycles

    result = client.step(reset.episode_id, act())

    assert result.truncated is True
    assert factory.built[0]._cycles == reached


def test_a_real_sim_crash_still_voids_the_episode() -> None:
    client, gateway = client_for(
        make_tasks(), FakeFactory(crash_at_cycle=1), EnforcementMode.OFFICIAL
    )
    reset = client.reset("t")
    with pytest.raises(GatewayError) as exc:
        client.step(reset.episode_id, act())
    assert exc.value.code is ErrorCode.ENV_CRASH
    assert gateway.trials_used("t") == 0


def test_the_job_budget_is_five_minutes_a_trial_run_one_after_another(tmp_path) -> None:
    task = "    embodiment: franka\n    simulator: {sim}\n    instruction: go\n    env_target: fake\n"
    (tmp_path / "pack.yaml").write_text(
        f"pack_id: p\n{PACK_LIMITS.replace('trial_time_s: 60', 'trial_time_s: 300')}tasks:\n"
        f"  - task_id: a\n{task.format(sim='robosuite')}    n_trials: 10\n"
        f"  - task_id: b\n{task.format(sim='libero')}"
    )
    manifest, _ = load_pack(tmp_path)
    assert manifest.job_timeout_s() == 300 * 11
    assert manifest.job_timeout_s({"libero"}) == 300 * 1
