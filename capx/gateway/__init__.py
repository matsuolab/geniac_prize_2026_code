"""CapX gateway: the environment-side referee that implements ``EnvProtocol``."""

from __future__ import annotations

from capx.gateway.gateway import EnvFactory, Gateway, GatewaySession, full_obs, proprio_obs
from capx.gateway.low_level import LowLevelEnv, StepOutcome
from capx.gateway.referee import EnforcementMode, Episode, Referee
from capx.gateway.scoring import EpisodeScore, Scoring

__all__ = [
    "LowLevelEnv",
    "StepOutcome",
    "full_obs",
    "proprio_obs",
    "Gateway",
    "GatewaySession",
    "EnvFactory",
    "Referee",
    "Episode",
    "EnforcementMode",
    "Scoring",
    "EpisodeScore",
]
