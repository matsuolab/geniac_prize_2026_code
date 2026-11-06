"""Per-embodiment gateway shims: low-level envs that drive the gateway's ``EnvClient``.

Each shim is imported on first use, so using one does not pay for another's
dependencies (the R1Pro shim needs torch).
"""

from __future__ import annotations

from typing import Any

from capx.submission.shims.base import GatewayShimBase

_LAZY = {
    "FrankaGatewayShim": "capx.submission.shims.franka",
    "BimanualFrankaGatewayShim": "capx.submission.shims.bimanual",
    "R1ProGatewayShim": "capx.submission.shims.r1pro",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "GatewayShimBase",
    "FrankaGatewayShim",
    "BimanualFrankaGatewayShim",
    "R1ProGatewayShim",
]
