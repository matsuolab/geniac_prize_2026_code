"""Build a low-level env for a task without ``get_env``'s process-wide cache.

``capx.envs.base.get_env`` is ``@lru_cache``'d. A gateway lane already reuses its env
across trials of a task, and a pack with many distinct scenes would otherwise keep
every sim and render context it ever built alive for the whole run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from capx.protocol.taskpack import TaskSpec

if TYPE_CHECKING:
    from capx.envs.base import BaseEnv


# Resets that let the scene settle step the simulator, and those steps count against
# its horizon (100 of them in nut assembly). The gateway counts a trial's cycles
# itself, so the horizon only has to stay out of the way.
HORIZON_MARGIN_STEPS = 10_000


def build_low_level_env(spec: TaskSpec, *, enable_render: bool = True) -> BaseEnv:
    """Construct ``spec.env_target`` fresh through the function ``get_env`` memoizes.

    An env that takes ``max_steps`` gets a horizon of the task's ``trial_steplimit``
    plus ``HORIZON_MARGIN_STEPS``: the gateway ends the trial at the steplimit, and
    the simulator must not end it earlier.
    """
    from capx.envs.base import env_parameters, get_env

    kwargs = dict(spec.env_kwargs)
    if "max_steps" in env_parameters(spec.env_target):
        kwargs["max_steps"] = spec.trial_steplimit + HORIZON_MARGIN_STEPS
    return get_env.__wrapped__(
        spec.env_target,
        privileged=spec.privileged,
        enable_render=enable_render,
        **kwargs,
    )


__all__ = ["HORIZON_MARGIN_STEPS", "build_low_level_env"]
