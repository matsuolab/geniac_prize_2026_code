"""``python -m capx.submission.run``: run the reference agent against a gateway."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

import tyro

from capx.submission.interfaces import (
    GATEWAY_CONFIG_DIR,
    SubmissionConfig,
    load_submission_config,
)
from capx.submission.runner import run_submission


@dataclass
class SubmissionArgs:
    """Overrides of the ``SubmissionConfig`` fields of the same name; unset keeps the YAML."""

    config_path: str = str(GATEWAY_CONFIG_DIR / "runner.yaml")
    gateway_socket: str | None = None
    model: str | None = None
    server_url: str | None = None
    trials_per_task: int | None = None
    concurrency: int | None = None
    """Capped at ``CAPX_MAX_LANES``, the most connections the gateway accepts."""
    task_filter: list[str] | None = None
    use_parallel_ensemble: bool | None = None
    """Unset falls back to ``CAPX_USE_PARALLEL_ENSEMBLE``, then the YAML."""
    output_dir: str | None = None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else None


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    raise SystemExit(f"{name} must be one of 1/0/true/false/yes/no, not {raw!r}")


def _apply_overrides(cfg: SubmissionConfig, args: SubmissionArgs) -> SubmissionConfig:
    """Layer the set CLI flags, then the organizer's environment, over the YAML.

    The container's environment wins over the YAML because the organizer, not the
    team's config, knows what it provisioned.
    """
    overrides = {
        field.name: getattr(args, field.name)
        for field in dataclasses.fields(args)
        if field.name != "config_path" and getattr(args, field.name) is not None
    }
    if args.use_parallel_ensemble is None:
        from_env = _env_bool("CAPX_USE_PARALLEL_ENSEMBLE")
        if from_env is not None:
            overrides["use_parallel_ensemble"] = from_env
    max_lanes = _env_int("CAPX_MAX_LANES")
    concurrency = overrides.get("concurrency", cfg.concurrency)
    if max_lanes is not None and concurrency > max_lanes:
        overrides["concurrency"] = max_lanes
    return SubmissionConfig.model_validate({**cfg.model_dump(), **overrides})


def main(args: SubmissionArgs) -> None:
    run_submission(_apply_overrides(load_submission_config(args.config_path), args))


if __name__ == "__main__":
    main(tyro.cli(SubmissionArgs))
