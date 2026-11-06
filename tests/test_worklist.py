"""Handing trials to lanes within each simulator's lane cap (capx/submission/worklist.py)."""

from __future__ import annotations

import threading

from capx.submission.worklist import Worklist


def _worklist(items: list[str], caps: dict[str, int]) -> Worklist[str, str]:
    """Items are named by their simulator's first letter: b1 runs on "b"."""
    return Worklist(items, lambda item: item[0], caps)


def test_capped_trials_go_out_first_and_order_is_kept_otherwise() -> None:
    work = _worklist(["r1", "l1", "b1", "r2", "b2"], {"b": 8})
    assert [work.take(lane) for lane in range(5)] == ["b1", "b2", "r1", "l1", "r2"]
    assert work.take(0) is None


def test_a_lane_is_not_handed_a_trial_of_a_simulator_at_its_cap() -> None:
    work = _worklist(["b1", "b2", "b3", "r1"], {"b": 2})
    assert (work.take(0), work.take(1)) == ("b1", "b2")
    assert work.take(2) == "r1"  # b3 would wait for a backend


def test_a_lane_that_holds_the_simulator_keeps_going_on_it() -> None:
    work = _worklist(["b1", "b2", "b3"], {"b": 1})
    assert work.take(0) == "b1"
    assert work.take(0) == "b2"
    assert work.take(0) == "b3"


def test_a_lane_waits_for_room_and_gets_it_when_a_lane_closes() -> None:
    work = _worklist(["b1", "b2"], {"b": 1})
    assert work.take(0) == "b1"
    got: list[str | None] = []
    waiting = threading.Thread(target=lambda: got.append(work.take(1)))
    waiting.start()
    waiting.join(0.2)
    assert waiting.is_alive()
    work.release(0)
    waiting.join(5)
    assert got == ["b2"]


def test_a_waiting_lane_ends_when_the_work_runs_out() -> None:
    work = _worklist(["b1", "b2"], {"b": 1})
    assert work.take(0) == "b1"
    got: list[str | None] = []
    waiting = threading.Thread(target=lambda: got.append(work.take(1)))
    waiting.start()
    assert work.take(0) == "b2"  # the lane holding BEHAVIOR takes the last one
    assert work.take(0) is None
    work.release(0)
    waiting.join(5)
    assert got == [None]


def test_a_settled_trial_can_come_back_to_the_front() -> None:
    work = _worklist(["r1", "r2"], {})
    first = work.take(0)
    assert first == "r1"
    work.settle(first)
    assert work.take(1) == "r1"


def test_wait_holds_an_idle_lane_until_trials_in_flight_settle() -> None:
    work = _worklist(["r1"], {})
    taken = work.take(0)
    assert work.take(1) is None
    result: list[bool] = []
    waiting = threading.Thread(target=lambda: result.append(work.wait()))
    waiting.start()
    waiting.join(0.2)
    assert waiting.is_alive()
    work.settle(taken)
    waiting.join(5)
    assert result == [True]
    assert work.take(1) == "r1"
    work.settle(None)
    assert work.wait() is False


def test_drain_returns_what_is_left() -> None:
    work = _worklist(["b1", "r1", "r2"], {"b": 1})
    assert work.take(0) == "b1"
    assert work.drain() == ["r1", "r2"]
    assert work.take(1) is None
