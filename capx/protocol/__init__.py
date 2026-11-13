"""CapX evaluation protocol: the only surface between a submission and the gateway.

    manifest()                   -> [ManifestEntry]
    reset(task_id)               -> {episode_id, obs (full tier), instruction}
    step(episode_id, raw_action) -> {obs (proprio tier, full on request), terminated, truncated}
    get_observation(episode_id)  -> {obs (full tier: cameras + proprio)}
    finish()                     -> ends the job; the score is final

Everything else (code execution, perception, IK, the control-primitive library)
lives on the submission side. The gateway never runs untrusted code and never
exposes reward, success, oracles or seeds.

- ``observations``: obs tiers + the allowlist ``ObsFilter`` + msgpack codec.
- ``actions``:      embodiment-specific raw actions + validation + msgpack codec.
- ``env_protocol``: ``EnvProtocol`` / ``Transport`` / ``EnvClient`` / error codes.
- ``taskpack``:     public/private task-pack schema + manifest.
"""

from __future__ import annotations

from capx.protocol.actions import (
    BimanualFrankaRawAction,
    FrankaRawAction,
    InvalidActionError,
    R1ProRawAction,
    RawAction,
    decode_action,
    encode_action,
)
from capx.protocol.env_protocol import (
    EnvClient,
    EnvProtocol,
    ErrorCode,
    GatewayError,
    InProcessTransport,
    ResetResult,
    StepResult,
    Transport,
    UnixTransport,
)
from capx.protocol.observations import (
    ArmProprio,
    CameraObs,
    FullObs,
    ObsFilter,
    ProprioObs,
    decode_obs,
    encode_obs,
)
from capx.protocol.taskpack import (
    Embodiment,
    ManifestEntry,
    TaskPackManifest,
    TaskSpec,
    Visibility,
)

__all__ = [
    # observations
    "CameraObs",
    "FullObs",
    "ArmProprio",
    "ProprioObs",
    "ObsFilter",
    "encode_obs",
    "decode_obs",
    # actions
    "RawAction",
    "FrankaRawAction",
    "BimanualFrankaRawAction",
    "R1ProRawAction",
    "InvalidActionError",
    "encode_action",
    "decode_action",
    # env protocol
    "EnvProtocol",
    "Transport",
    "InProcessTransport",
    "UnixTransport",
    "EnvClient",
    "ErrorCode",
    "GatewayError",
    "ResetResult",
    "StepResult",
    # taskpack
    "Embodiment",
    "Visibility",
    "TaskSpec",
    "ManifestEntry",
    "TaskPackManifest",
]
