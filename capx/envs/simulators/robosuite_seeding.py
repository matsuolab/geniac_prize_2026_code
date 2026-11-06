"""Seeding a robosuite env so that a reset is a function of its seed."""

from __future__ import annotations

import random
from typing import Any

import numpy as np


def reseed(robosuite_env: Any, seed: int) -> None:
    """Pin every RNG a robosuite reset draws from to `seed`.

    The placement samplers keep their own reference to the env's rng, set when the
    model was loaded, and the robot's initial joint noise comes from numpy's global
    RNG, so replacing the wrapper's own rng alone leaves the scene unseeded.
    """
    robosuite_env.rng = np.random.default_rng(seed)
    for attr in ("placement_initializer", "_distractor_sampler"):
        sampler = getattr(robosuite_env, attr, None)
        if sampler is None:
            continue
        sampler.rng = robosuite_env.rng
        for sub in getattr(sampler, "samplers", {}).values():
            sub.rng = robosuite_env.rng
    np.random.seed(seed)
    random.seed(seed)


__all__ = ["reseed"]
