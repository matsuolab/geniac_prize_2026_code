"""Hand a run's trials to its lanes, keeping each simulator within its lane cap.

The gateway caps how many lanes may run one simulator at once (PRELIM_RULEBOOK.md
§4.4). A lane over the cap waits inside ``reset`` until a backend comes free, doing
nothing meanwhile. So a lane is handed only a trial it can start: one of the simulator
it already holds, or of a simulator under its cap. A lane holds its simulator's
backend until it resets a task of another one, or closes its connection.

Trials of a capped simulator are handed out first: they cannot spread over every
lane, so started last they would run on a few lanes while the rest sit idle.

A trial that is taken counts as in flight until the lane settles it. One in flight
can still come back (a void, a lost lane), so the list is not done until nothing is
in flight either.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Generic, TypeVar

T = TypeVar("T")
S = TypeVar("S", bound=Hashable)


class Worklist(Generic[T, S]):  # noqa: UP046  # requires-python includes 3.10, which lacks `class C[T]`
    """Trials of type ``T``, each run by a simulator of type ``S``."""

    def __init__(
        self, items: Iterable[T], simulator_of: Callable[[T], S], caps: Mapping[S, int]
    ) -> None:
        self._simulator_of = simulator_of
        self._caps = dict(caps)
        listed = list(items)
        self._items = [item for item in listed if simulator_of(item) in self._caps] + [
            item for item in listed if simulator_of(item) not in self._caps
        ]
        self._holding: dict[int, S] = {}
        self._in_flight = 0
        self._changed = threading.Condition()

    def take(self, lane: int) -> T | None:
        """The next trial ``lane`` can start, waiting while everything left belongs to a
        simulator at its cap. None if there is nothing left to take right now."""
        with self._changed:
            while self._items:
                for index, item in enumerate(self._items):
                    simulator = self._simulator_of(item)
                    if self._holding.get(lane) == simulator or self._has_room(simulator):
                        del self._items[index]
                        self._holding[lane] = simulator
                        self._in_flight += 1
                        self._changed.notify_all()
                        return item
                self._changed.wait()
            return None

    def wait(self) -> bool:
        """Block while there is nothing to take but trials are in flight; False once
        there is nothing left at all."""
        with self._changed:
            while not self._items and self._in_flight:
                self._changed.wait()
            return bool(self._items)

    def settle(self, requeue: T | None) -> None:
        """A taken trial is over; ``requeue`` goes back to the front of the list."""
        with self._changed:
            self._in_flight -= 1
            if requeue is not None:
                self._items.insert(0, requeue)
            self._changed.notify_all()

    def release(self, lane: int) -> None:
        """``lane`` has closed its connection, so it holds no simulator any more."""
        with self._changed:
            self._holding.pop(lane, None)
            self._changed.notify_all()

    def drain(self) -> list[T]:
        with self._changed:
            items = self._items
            self._items = []
            return items

    def _has_room(self, simulator: S) -> bool:
        cap = self._caps.get(simulator)
        return cap is None or sum(held == simulator for held in self._holding.values()) < cap


__all__ = ["Worklist"]
