"""``GatewayCodeExecEnv``: the ``CodeExecutionEnvBase`` subclass driven by a gateway shim."""

from __future__ import annotations

import copy
from typing import Any

from capx.envs.base import ObsType
from capx.envs.tasks.base import CodeExecEnvConfig, CodeExecutionEnvBase
from capx.submission.interfaces import TaskRunConfig
from capx.submission.shims.base import GatewayShimBase

_INSTRUCTION_PLACEHOLDER = "{task_instruction}"


class GatewayCodeExecEnv(CodeExecutionEnvBase):
    """Injects each episode's instruction, which comes from the gateway, into the prompt."""

    prompt = ""

    def __init__(self, cfg: CodeExecEnvConfig) -> None:
        super().__init__(cfg)
        self._prompt_template: list[dict[str, Any]] = copy.deepcopy(self._full_prompt)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Restore the prompt template, reset, and substitute the new instruction.

        ``_full_prompt`` is updated in place because ``obs["full_prompt"]`` is that
        same list. ``str.replace``, not ``str.format``: the API docs in the prompt
        contain literal braces.
        """
        self._full_prompt[:] = copy.deepcopy(self._prompt_template)
        obs, info = super().reset(seed=seed, options=options)

        instruction = getattr(self.low_level_env, "last_instruction", None)
        if instruction is not None:
            content = self._full_prompt[-1]["content"][0]
            content["text"] = content["text"].replace(_INSTRUCTION_PLACEHOLDER, instruction)
        return obs, info


def build_code_env(shim: GatewayShimBase, task_cfg: TaskRunConfig) -> GatewayCodeExecEnv:
    cfg = CodeExecEnvConfig(
        # CodeExecutionEnvBase only duck-types the low-level env; a shim is not an Env.
        low_level=shim,  # type: ignore[arg-type]
        apis=task_cfg.apis,
        prompt=task_cfg.prompt,
    )
    return GatewayCodeExecEnv(cfg)


__all__ = ["GatewayCodeExecEnv", "build_code_env"]
