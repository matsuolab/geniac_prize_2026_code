"""Evaluation settings that are not a round's: image limits and embodiment bounds.

A round's job budget, trial count and step limit are the task pack's
(``capx.protocol.taskpack.PackLimits``). Organizers override ``defaults.yaml`` by
passing their own YAML.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

_DEFAULTS = Path(__file__).with_name("defaults.yaml")


class SubmissionLimits(BaseModel):
    max_image_gb: float = 80.0


class FrankaLimits(BaseModel):
    joint_lower: list[float]
    joint_upper: list[float]
    gripper_min: float = 0.0
    gripper_max: float = 1.0

    @model_validator(mode="after")
    def _bounds_are_sane(self) -> FrankaLimits:
        # Fail at load, not as a numpy broadcast error inside the bounds check.
        if len(self.joint_lower) != 7 or len(self.joint_upper) != 7:
            raise ValueError("franka joint limits must have exactly 7 entries")
        if any(lo >= hi for lo, hi in zip(self.joint_lower, self.joint_upper, strict=True)):
            raise ValueError("franka joint_lower must be strictly below joint_upper")
        if self.gripper_min >= self.gripper_max:
            raise ValueError("franka gripper_min must be strictly below gripper_max")
        return self


class R1ProLimits(BaseModel):
    action_dim: int | None = None


class EmbodimentLimits(BaseModel):
    franka: FrankaLimits
    r1pro: R1ProLimits = Field(default_factory=R1ProLimits)


class EvalConfig(BaseModel):
    submission: SubmissionLimits = Field(default_factory=SubmissionLimits)
    embodiments: EmbodimentLimits


def load_eval_config(path: str | None = None) -> EvalConfig:
    data = yaml.safe_load(Path(path or _DEFAULTS).read_text())
    return EvalConfig.model_validate(data)


__all__ = [
    "SubmissionLimits",
    "FrankaLimits",
    "R1ProLimits",
    "EmbodimentLimits",
    "EvalConfig",
    "load_eval_config",
]
