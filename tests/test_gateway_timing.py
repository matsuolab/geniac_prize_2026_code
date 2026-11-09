"""Where an episode's wall clock is attributed.

The whole point of these numbers is deciding what to spend money on — more
simulators, more concurrent episodes, a faster model server — so the split has to
be right. A fake clock that only advances inside the simulator's own methods makes
the expected attribution exact rather than approximate.
"""

from __future__ import annotations

from test_http_transport import CAMERA, VALID_JOINTS, FakeEnv, make_tasks

from capx.gateway import EnforcementMode, Gateway
from capx.gateway.low_level import StepOutcome
from capx.protocol import EnvClient, FrankaRawAction, InProcessTransport


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TimedEnv(FakeEnv):
    """A simulator that costs a known amount of the clock."""

    def __init__(self, clock: FakeClock, *, step_s: float, render_s: float) -> None:
        super().__init__()
        self._clock = clock
        self._step_s = step_s
        self._render_s = render_s

    def apply_raw_action(self, action):  # noqa: ANN001 - matches LowLevelEnv
        self._clock.advance(self._step_s)
        return super().apply_raw_action(action)

    def get_observation(self):
        self._clock.advance(self._render_s)
        return super().get_observation()

    def get_proprio(self):
        return super().get_proprio()


class TimedFactory:
    def __init__(self, clock: FakeClock, *, step_s: float = 0.1, render_s: float = 1.0) -> None:
        self._clock = clock
        self._step_s = step_s
        self._render_s = render_s

    def build(self, spec) -> TimedEnv:  # noqa: ANN001
        return TimedEnv(self._clock, step_s=self._step_s, render_s=self._render_s)


def drive(clock: FakeClock, factory: TimedFactory, *, steps: int, think_s: float):
    gateway = Gateway(make_tasks(), factory, EnforcementMode.DEV, clock=clock)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    for _ in range(steps):
        clock.advance(think_s)  # the submission generating, perceiving, waiting
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))
    return gateway, reset.episode_id


def test_simulator_time_and_client_time_are_told_apart() -> None:
    clock = FakeClock()
    gateway, episode_id = drive(
        clock, TimedFactory(clock, step_s=0.1, render_s=1.0), steps=10, think_s=5.0
    )

    episode = next(e for e in gateway.timings()["episodes"] if e["episode_id"] == episode_id)
    assert episode["steps"] == 10
    assert episode["sim_s"] == 1.0  # 10 steps x 0.1
    assert episode["obs_s"] == 1.0  # the reset's full obs; steps return proprio
    assert episode["client_s"] == 50.0  # 10 x 5 s of the submission's own work
    assert episode["wall_s"] == 52.0


def test_rendering_is_attributed_to_observations_not_the_step() -> None:
    """Cameras are the expensive tier, and a plan to add simulators rests on knowing
    how much of the clock they actually take."""
    clock = FakeClock()
    gateway = Gateway(make_tasks(), TimedFactory(clock, render_s=2.0), EnforcementMode.DEV,
                      clock=clock)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    client.get_observation(reset.episode_id)

    episode = gateway.timings()["episodes"][0]
    assert episode["observations"] == 2  # reset + the explicit get_observation
    assert episode["obs_s"] == 4.0
    assert episode["sim_s"] == 0.0


def test_an_idle_submission_shows_up_as_client_time() -> None:
    """The 40-minute trial that motivated this: the agent slept in its own retry
    backoff, and nothing in the simulator accounts for it."""
    clock = FakeClock()
    gateway, episode_id = drive(
        clock, TimedFactory(clock, step_s=0.01, render_s=0.01), steps=1, think_s=2400.0
    )

    episode = next(e for e in gateway.timings()["episodes"] if e["episode_id"] == episode_id)
    assert episode["client_s"] > 2399.0
    assert episode["sim_s"] + episode["obs_s"] < 1.0


def test_the_roll_up_averages_over_a_task_s_episodes() -> None:
    clock = FakeClock()
    factory = TimedFactory(clock, step_s=0.1, render_s=1.0)
    gateway = Gateway(make_tasks(), factory, EnforcementMode.DEV, clock=clock)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    for _ in range(2):
        reset = client.reset("t")
        clock.advance(4.0)
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))

    by_task = gateway.timings()["by_task"]["t"]
    assert by_task["episodes"] == 2
    assert by_task["mean_client_s"] == 4.0
    assert by_task["mean_sim_s"] == 0.1


def test_a_voided_episode_leaves_no_timing_behind() -> None:
    """env_crash drops the score, so it has to drop the timing too — otherwise the
    per-task mean is diluted by an episode that does not count."""
    clock = FakeClock()

    class CrashingFactory(TimedFactory):
        def build(self, spec):  # noqa: ANN001
            env = super().build(spec)

            def boom(action):  # noqa: ANN001
                raise RuntimeError("sim died")

            env.apply_raw_action = boom
            return env

    gateway = Gateway(make_tasks(), CrashingFactory(clock), EnforcementMode.DEV, clock=clock)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    try:
        client.step(reset.episode_id, FrankaRawAction(joints=VALID_JOINTS, gripper=0.0))
    except Exception:
        pass

    assert gateway.timings()["episodes"] == []
    assert CAMERA  # the fake env is the one the rest of the suite uses


class SucceedingEnv(FakeEnv):
    """Achieves the task on its ``at``-th control cycle and stays achieved."""

    def __init__(self, clock: FakeClock, at: int | None) -> None:
        super().__init__()
        self._clock, self._at, self._cycles = clock, at, 0

    def apply_raw_action(self, action):  # noqa: ANN001 - matches LowLevelEnv
        self._clock.advance(0.1)
        super().apply_raw_action(action)
        self._cycles += 1
        return StepOutcome(success=self._at is not None and self._cycles >= self._at)


class SucceedingFactory:
    def __init__(self, clock: FakeClock, at: int | None) -> None:
        self._clock, self._at = clock, at

    def build(self, spec) -> SucceedingEnv:  # noqa: ANN001
        return SucceedingEnv(self._clock, self._at)


def _episode_after(at: int | None, repeats: list[int]) -> dict:
    clock = FakeClock()
    gateway = Gateway(make_tasks(), SucceedingFactory(clock, at), EnforcementMode.DEV, clock=clock)
    client = EnvClient(InProcessTransport(gateway.acquire()))
    reset = client.reset("t")
    for repeat in repeats:
        clock.advance(1.0)
        action = FrankaRawAction(joints=VALID_JOINTS, gripper=0.0, repeat=repeat)
        client.step(reset.episode_id, action)
    return next(e for e in gateway.timings()["episodes"] if e["episode_id"] == reset.episode_id)


def test_the_first_success_is_logged_with_its_cycle_count_and_wall_time() -> None:
    episode = _episode_after(at=3, repeats=[1] * 6)
    assert episode["steps"] == 6  # the episode went on after succeeding
    assert episode["steps_to_success"] == 3
    assert episode["wall_to_success_s"] == 3.3  # 3 x (1.0 s of the submission + one 0.1 s cycle)


def test_a_success_partway_through_a_repeat_counts_the_cycles_up_to_it() -> None:
    episode = _episode_after(at=7, repeats=[5, 5])
    assert episode["steps"] == 10 and episode["steps_to_success"] == 7


def test_an_episode_that_never_succeeds_logs_no_success() -> None:
    episode = _episode_after(at=None, repeats=[1] * 4)
    assert episode["steps_to_success"] is None and episode["wall_to_success_s"] is None
