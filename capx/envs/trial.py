"""Single-trial execution for CaP-X environments.

This module handles single trial execution including code generation,
multi-turn decisions, and visual feedback. It contains the core trial
loop extracted from launch.py, covering:

- Initial code generation and oracle code handling
- Code block execution with multi-turn regeneration
- Visual feedback capture and image/video differencing
- Trial artifact saving (code, logs, per-turn videos, combined video)
"""

from __future__ import annotations

import base64
import copy
import gc
import io
import json
import os
import signal
import time
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from capx.envs.tasks.base import CodeExecutionEnvBase
from capx.llm.client import (
    ModelQueryArgs,
    query_model,
    query_model_ensemble,
    query_single_model_ensemble,
)
from capx.submission.interfaces import EPISODE_OVER_MESSAGE
from capx.utils import turn_timing
from capx.utils.launch_utils import (
    TrialSummary,
    _build_multi_turn_decision_prompt,
    _extract_code,
    _get_visual_feedback,
    _parse_multi_turn_decision,
    _save_trial_artifacts,
)
from capx.utils.video_utils import _encode_video_base64, _write_video

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs


MULTITURN_LIMIT = 10

# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------

def _annotate_code_blocks(
    code_blocks: list[str],
    code_block_metadata: list[dict[str, Any]],
) -> str:
    """Join code blocks into a single string with ``# Code block N`` headers."""
    annotated = []
    for i, (block, _metadata) in enumerate(zip(code_blocks, code_block_metadata, strict=False)):
        annotated.append(f"# Code block {i}\n{block}")
    return "\n\n".join(annotated)


# Per-million-token prices in USD. Output prices include reasoning ("thinking")
# tokens, which OpenAI bills at the output rate. Cached input is billed at the
# `cached_in` rate when the request hits the prompt cache.
#
# Sources (Jan 2026 snapshot of OpenAI public pricing pages):
#   - gpt-5 / gpt-5.1: $1.25 / $10 (cached input $0.125)
#     https://developers.openai.com/api/docs/models/gpt-5.1
#   - gpt-5-mini / gpt-5.1-mini: $0.25 / $2 (cached $0.025)
#   - gpt-5-nano / gpt-5.1-nano: $0.05 / $0.40 (cached $0.005)
#   - gpt-4o: $2.50 / $10 (cached $1.25)
#   - gpt-4o-mini: $0.15 / $0.60 (cached $0.075)
# When the model is unknown we return zeros so the eval still runs.
_MODEL_PRICING_PER_M: dict[str, dict[str, float]] = {
    # Reasoning / GPT-5 family
    "gpt-5.1":      {"in": 1.25, "cached_in": 0.125, "out": 10.0},
    "gpt-5.1-mini": {"in": 0.25, "cached_in": 0.025, "out": 2.0},
    "gpt-5.1-nano": {"in": 0.05, "cached_in": 0.005, "out": 0.40},
    "gpt-5":        {"in": 1.25, "cached_in": 0.125, "out": 10.0},
    "gpt-5-mini":   {"in": 0.25, "cached_in": 0.025, "out": 2.0},
    "gpt-5-nano":   {"in": 0.05, "cached_in": 0.005, "out": 0.40},
    # o-series reasoning models (approximate, may drift)
    "o1":           {"in": 15.0, "cached_in": 7.5,   "out": 60.0},
    "o1-mini":      {"in": 1.10, "cached_in": 0.55,  "out": 4.40},
    "o3":           {"in": 2.0,  "cached_in": 0.50,  "out": 8.0},
    "o3-mini":      {"in": 1.10, "cached_in": 0.55,  "out": 4.40},
    "o4-mini":      {"in": 1.10, "cached_in": 0.275, "out": 4.40},
    # Chat models
    "gpt-4o":       {"in": 2.50, "cached_in": 1.25,  "out": 10.0},
    "gpt-4o-mini":  {"in": 0.15, "cached_in": 0.075, "out": 0.60},
}


def _model_pricing(model_name: str | None) -> dict[str, float]:
    """Return per-million-token prices for the given model.

    Strips an optional ``openai/`` prefix and falls back to the longest exact
    prefix match (e.g. ``gpt-5.1-2025-11-13`` -> ``gpt-5.1``). Unknown models
    return zeros so cost simply shows as $0 rather than crashing the run.
    """
    if not model_name:
        return {"in": 0.0, "cached_in": 0.0, "out": 0.0}
    m = model_name.lower()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    if m in _MODEL_PRICING_PER_M:
        return _MODEL_PRICING_PER_M[m]
    # longest-prefix fallback
    best = None
    for key in _MODEL_PRICING_PER_M:
        if m.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    if best is not None:
        return _MODEL_PRICING_PER_M[best]
    return {"in": 0.0, "cached_in": 0.0, "out": 0.0}


def _aggregate_usage(
    all_responses: list[dict[str, Any]] | None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Sum input / output / thinking tokens across all LLM calls in a trial.

    Each entry in ``all_responses`` may carry a ``usage`` dict produced by
    ``capx/llm/client.py:query_model`` with keys ``input_tokens``,
    ``output_tokens``, ``thinking_tokens`` and optionally
    ``cached_prompt_tokens``. Missing values are treated as 0.

    Also computes USD cost using the per-model pricing table. Reasoning
    ("thinking") tokens are billed at the output rate; cached input tokens
    receive the discounted ``cached_in`` rate. Returns a dict with token
    totals plus ``cost_usd`` and ``model``.
    """
    totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "thinking_tokens": 0,
        "calls": 0,
    }
    if all_responses:
        for resp in all_responses:
            usage = resp.get("usage") if isinstance(resp, dict) else None
            if not usage:
                continue
            totals["calls"] += 1
            for key, src in (
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("thinking_tokens", "thinking_tokens"),
                ("cached_input_tokens", "cached_prompt_tokens"),
            ):
                value = usage.get(src)
                if isinstance(value, (int, float)) and value is not None:
                    totals[key] += int(value)

    price = _model_pricing(model_name)
    non_cached_input = max(totals["input_tokens"] - totals["cached_input_tokens"], 0)
    cost = (
        non_cached_input          / 1_000_000 * price["in"]
        + totals["cached_input_tokens"] / 1_000_000 * price["cached_in"]
        # OpenAI counts reasoning_tokens inside completion_tokens already,
        # so we charge output_tokens (which already includes thinking) at
        # the output rate. Do NOT double-count thinking_tokens here.
        + totals["output_tokens"] / 1_000_000 * price["out"]
    )
    totals["cost_usd"] = round(cost, 6)
    totals["model"] = model_name
    totals["price_per_m"] = price
    return totals


def _build_log_lines(
    final_code: str,
    info_step: dict[str, Any],
    reward: float,
    terminated: bool,
    truncated: bool,
    num_regenerations: int,
    num_finishes: int,
    num_code_blocks: int,
    *,
    prefix: str = "",
    stderr_override: str | None = None,
    token_totals: dict[str, int] | None = None,
) -> list[str]:
    """Build the standard log-line list used for both normal and timeout summaries."""
    stderr = stderr_override if stderr_override is not None else info_step.get("stderr", "")
    lines = ["-" * 100]
    if prefix:
        lines.append(prefix)
    lines.extend([
        "Generated program:",
        final_code if final_code else "(no program available)",
        "\n\nEnvironment response:",
        f"  Sandbox failed: {info_step.get('sandbox_rc', 1)}",
        f"  Stdout: {info_step.get('stdout', '')}",
        f"  Stderr: {stderr}",
        f"  Reward: {reward}",
        f"  Task Completed: {info_step.get('task_completed', False)}",
        f"  Terminated: {terminated}, Truncated: {truncated}",
        f"  Num Regenerations: {num_regenerations}",
        f"  Num Finishes: {num_finishes}",
        f"  Num Code Blocks: {num_code_blocks}",
    ])
    if token_totals:
        lines.extend([
            f"  Tokens (input)        : {token_totals.get('input_tokens', 0)}",
            f"  Tokens (cached input) : {token_totals.get('cached_input_tokens', 0)}",
            f"  Tokens (output)       : {token_totals.get('output_tokens', 0)}",
            f"  Tokens (thinking)     : {token_totals.get('thinking_tokens', 0)}",
            f"  LLM calls             : {token_totals.get('calls', 0)}",
            f"  Cost USD ({token_totals.get('model', '?')}): ${token_totals.get('cost_usd', 0):.4f}",
        ])
    lines.append("-" * 100)
    return lines


# ---------------------------------------------------------------------------
# Trial video directory helper
# ---------------------------------------------------------------------------

def _trial_video_dir(
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
) -> str:
    """Return the trial output directory path used for video saving."""
    # task_completed is None on the gateway client path (success never crosses
    # the boundary, PRELIM_RULEBOOK.md §4.5); record it as 0 in the directory name.
    task_completed = info_step.get("task_completed")
    return os.path.join(
        config["output_dir"],
        f"trial_{trial:02d}_sandboxrc_{info_step['sandbox_rc']}_reward_{reward:.3f}"
        f"_taskcompleted_{0 if task_completed is None else int(task_completed)}",
    )


def _save_trial_video(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    num_code_blocks: int,
    *,
    suffix_extra: str = "",
) -> None:
    """Save recorded video frames from the environment, if available."""
    if not config["record_video"] or not hasattr(env, "get_video_frames"):
        return
    frames = env.get_video_frames(clear=True)
    if not frames or not config["output_dir"]:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)
    suffix = f"{reward:.3f}"
    if suffix_extra:
        suffix += f"_{suffix_extra}"

    if isinstance(frames, list):
        _write_video(frames, base_dir, suffix=suffix)
    elif isinstance(frames, dict):
        for key, frame in frames.items():
            _write_video(frame, base_dir, suffix=f"{suffix}_{key}")


def _save_turn_and_combined_videos(
    env: CodeExecutionEnvBase,
    config: dict[str, Any],
    trial: int,
    info_step: dict[str, Any],
    reward: float,
    turn_frame_ranges: list[tuple[int, int]],
) -> None:
    """Save per-turn videos and a combined video of all turns.

    Gets all frames from the environment (clearing the buffer), then writes:
      - ``video_turn_00.mp4``, ``video_turn_01.mp4``, ... for each turn
      - ``video_combined.mp4`` for the full trial
      - If wrist camera is enabled: ``video_turn_00_wrist.mp4``, etc.
    """
    if not config["record_video"] or not config["output_dir"]:
        return
    if not hasattr(env, "get_video_frames"):
        return

    all_frames = env.get_video_frames(clear=True)
    if not all_frames:
        return

    base_dir = _trial_video_dir(config, trial, info_step, reward)

    # all_frames may be a list (Robosuite) or a dict of lists (R1Pro multi-camera).
    # Normalise to a list for slicing; dict case is handled by _write_multi_video.
    if isinstance(all_frames, dict):
        # Multi-camera: write each camera stream as a combined video
        for key, frames in all_frames.items():
            if frames:
                _write_video(frames, base_dir, suffix=f"combined_{key}")
        return

    # Per-turn videos
    for i, (start, end) in enumerate(turn_frame_ranges):
        turn_frames = all_frames[start:end]
        if turn_frames:
            _write_video(turn_frames, base_dir, suffix=f"turn_{i:02d}")

    # Combined video
    _write_video(all_frames, base_dir, suffix="combined")

    # Wrist camera videos
    if config.get("use_wrist_camera") and hasattr(env, "get_wrist_video_frames"):
        wrist_frames = env.get_wrist_video_frames(clear=True)
        if wrist_frames:
            for i, (start, end) in enumerate(turn_frame_ranges):
                wrist_turn = wrist_frames[start:end]
                if wrist_turn:
                    _write_video(wrist_turn, base_dir, suffix=f"turn_{i:02d}_wrist")
            _write_video(wrist_frames, base_dir, suffix="combined_wrist")


# ---------------------------------------------------------------------------
# Visual feedback and image differencing
# ---------------------------------------------------------------------------

def _capture_initial_visual_feedback(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    config: dict[str, Any],
    args: LaunchArgs,
    visual_differencing_args: ModelQueryArgs,
) -> tuple[list, list[str], str]:
    """Capture the initial environment image and optionally describe it.

    Returns:
        (visual_feedback_imgs, visual_feedback_base64_history, task_description)
    """
    visual_feedback_imgs: list = []
    visual_feedback_base64_history: list[str] = []
    task_description = ""

    use_wrist = config.get("use_wrist_camera", False)

    needs_visual = (
        config["use_visual_feedback"]
        or config["use_img_differencing"]
        or config.get("use_video_differencing", False)
    )
    if not (needs_visual and hasattr(env, "render")):
        return visual_feedback_imgs, visual_feedback_base64_history, task_description

    with turn_timing.phase("visual"):
        initial_base64, initial_img = _get_visual_feedback(env)
        visual_feedback_imgs.append(initial_img)
        visual_feedback_base64_history.append(initial_base64)
        task_description = copy.deepcopy(obs["full_prompt"][-1]["content"][0]["text"])

        # Also capture wrist camera image for multiview initial description
        initial_wrist_base64 = None
        if use_wrist and hasattr(env, "render_wrist"):
            wrist_img = env.render_wrist()
            if wrist_img is not None:
                pil_wrist = Image.fromarray(wrist_img)
                buf = io.BytesIO()
                pil_wrist.save(buf, format="png")
                initial_wrist_base64 = (
                    f"data:image/png;base64,"
                    f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
                )
                visual_feedback_imgs.append(pil_wrist)

    # Append image to the prompt for VLM visual feedback
    if config["use_visual_feedback"]:
        obs["full_prompt"][-1]["content"][0]["text"] += (
            "\n\nIncluded below is an image of the initial state of the environment."
        )
        obs["full_prompt"][-1]["content"].append(
            {"type": "image_url", "image_url": {"url": initial_base64}}
        )
        if initial_wrist_base64 is not None:
            obs["full_prompt"][-1]["content"].append(
                {
                    "type": "text",
                    "text": "Included below is an image from the robot's wrist camera.",
                }
            )
            obs["full_prompt"][-1]["content"].append(
                {"type": "image_url", "image_url": {"url": initial_wrist_base64}}
            )

    # Image differencing: ask a VLM to describe the initial scene
    if config["use_img_differencing"] or config.get("use_video_differencing", False):
        with turn_timing.phase("differencing"):
            description = _describe_initial_scene(
                visual_differencing_args, task_description, initial_base64,
                wrist_image_base64=initial_wrist_base64,
            )
        feedback = f"The initial state of the environment is described as follows:\n{description}"
        obs["full_prompt"][-1]["content"][0]["text"] += f"\n\n{feedback}"
        if args.debug:
            print(description)

    return visual_feedback_imgs, visual_feedback_base64_history, task_description


def _describe_initial_scene(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    image_base64: str,
    wrist_image_base64: str | None = None,
) -> str:
    """Query a VLM to describe the initial environment state."""
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the initial state of the environment with the goal of the "
                "task in mind. You should try to provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera view:"},
        {"type": "image_url", "image_url": {"url": image_base64}},
    ]
    if wrist_image_base64 is not None:
        user_content.extend([
            {"type": "text", "text": "Wrist camera view:"},
            {"type": "image_url", "image_url": {"url": wrist_image_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the initial state of the "
                "environment with the goal of the task in mind. You should try to provide "
                "objective information and no assumptions. Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return query_model(visual_differencing_args, prompt)["content"]


def _get_visual_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    visual_feedback_base64_history: list[str],
    wrist_base64_history: list[str] | None = None,
) -> str | None:
    """Query a VLM to describe what changed between the two most recent frames.

    Args:
        wrist_base64_history: Optional history of wrist camera images.  When provided
            and has >=2 entries, the before/after wrist images are included in the prompt.
    """
    if len(visual_feedback_base64_history) < 2:
        return None

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "Describe the difference between the current state of the "
                "environment and the previous state of the environment with the "
                "goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code.."
            ),
        },
        {"type": "text", "text": "Previous state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-2]}},
        {"type": "text", "text": "Current state (main camera):"},
        {"type": "image_url", "image_url": {"url": visual_feedback_base64_history[-1]}},
    ]

    if wrist_base64_history and len(wrist_base64_history) >= 2:
        user_content.extend([
            {"type": "text", "text": "Previous state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-2]}},
            {"type": "text", "text": "Current state (wrist camera):"},
            {"type": "image_url", "image_url": {"url": wrist_base64_history[-1]}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that describes the difference between the "
                "current state of the environment and the previous state of the environment "
                "with the goal of the task in mind and whether the task has been completed. "
                "You should try to provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Video differencing
# ---------------------------------------------------------------------------

def _get_video_differencing_feedback(
    visual_differencing_args: ModelQueryArgs,
    task_description: str,
    turn_frames: list[np.ndarray],
    wrist_turn_frames: list[np.ndarray] | None = None,
) -> str | None:
    """Query a VLM with a video of the turn execution to describe what happened.

    Args:
        visual_differencing_args: Model query args for the VDM model.
        task_description: The task goal.
        turn_frames: RGB frames from the main camera for this turn.
        wrist_turn_frames: RGB frames from the wrist camera for this turn (optional).

    Returns:
        Text description of the execution, or None if no frames.
    """
    if not turn_frames:
        return None

    video_base64 = _encode_video_base64(turn_frames)

    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": task_description},
        {
            "type": "text",
            "text": (
                "The following video shows the robot executing code in the "
                "environment from the main camera view. Describe what happened "
                "during execution, including what actions the robot took, how "
                "the objects in the scene changed, and whether the task appears "
                "to have been completed. Provide objective information and no "
                "assumptions. Do *NOT* write any code."
            ),
        },
        {"type": "text", "text": "Main camera video:"},
        {"type": "image_url", "image_url": {"url": video_base64}},
    ]

    if wrist_turn_frames:
        wrist_video_base64 = _encode_video_base64(wrist_turn_frames)
        user_content.extend([
            {
                "type": "text",
                "text": (
                    "The following video shows the same execution from the "
                    "robot's wrist-mounted camera (eye-in-hand view), providing "
                    "a close-up perspective of the gripper and objects being "
                    "manipulated."
                ),
            },
            {"type": "text", "text": "Wrist camera video:"},
            {"type": "image_url", "image_url": {"url": wrist_video_base64}},
        ])

    prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes robot execution "
                "videos. You describe what happened during the robot's code "
                "execution, what actions were taken, how the environment "
                "changed, and whether the task appears to have been completed. "
                "Provide objective information and no assumptions. "
                "Do *NOT* write any code."
            ),
        },
        {"role": "user", "content": user_content},
    ]
    return query_model(visual_differencing_args, prompt)["content"]


# ---------------------------------------------------------------------------
# Initial code generation
# ---------------------------------------------------------------------------

def _query_initial_code(
    args: LaunchArgs,
    config: dict[str, Any],
    obs: dict[str, Any],
) -> tuple[str, str | None, dict | None, dict | None]:
    """Query the model for the initial code generation.

    Returns:
        (raw_code, reasoning, ensemble_data, usage)
        ``usage`` is a dict with input/output/thinking token counts, or None
        when ensemble paths obscure single-call accounting.
    """
    # Save the initial prompt
    with open(os.path.join(config["output_dir"], "initial_prompt.txt"), "w") as f:
        f.write(str(obs["full_prompt"]))

    ensemble_data = None
    if config["use_parallel_ensemble"]:
        if config.get("use_multimodel", False):
            print("RUNNING MULTIMODEL ENSEMBLE QUERY")
            out = query_model_ensemble(args, obs["full_prompt"], is_multiturn=False)
        else:
            print("RUNNING SINGLE MODEL ENSEMBLE QUERY")
            out = query_single_model_ensemble(args, obs["full_prompt"], args.model, is_multiturn=False)
        ensemble_data = {
            "ensemble_candidates_txt": out["ensemble_candidates_txt"],
            "ensemble_synthesis_txt": out["ensemble_synthesis_txt"],
        }
    else:
        out = query_model(args, obs["full_prompt"])

    return out["content"], out["reasoning"], ensemble_data, out.get("usage")


# ---------------------------------------------------------------------------
# Multi-turn decision handling
# ---------------------------------------------------------------------------

def _handle_multi_turn_step(
    env: CodeExecutionEnvBase,
    obs: dict[str, Any],
    args: LaunchArgs,
    config: dict[str, Any],
    visual_differencing_args: ModelQueryArgs,
    multi_turn_prompt: str,
    code_blocks: list[str],
    code_block_idx: int,
    info_step: dict[str, Any],
    task_description: str,
    visual_feedback_imgs: list,
    visual_feedback_base64_history: list[str],
    turn_frames: list[np.ndarray] | None = None,
    wrist_turn_frames: list[np.ndarray] | None = None,
    wrist_base64_history: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict | None, list[dict], dict | None]:
    """Execute one multi-turn decision step.

    Captures visual feedback, builds the decision prompt, queries the model,
    and returns the parsed decision.

    Args:
        turn_frames: Frames from the main camera for this turn (for video differencing).
        wrist_turn_frames: Frames from the wrist camera for this turn (for video differencing).
        wrist_base64_history: History of wrist camera base64 images for image-based
            differencing with multiview.

    Returns:
        (decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt, usage)
        where decision is "regenerate" or "finish".
    """
    use_wrist = config.get("use_wrist_camera", False)

    executed_code = "\n".join(code_blocks[:code_block_idx])
    complete_multi_turn_prompt = multi_turn_prompt.format(
        executed_code=executed_code,
        console_stdout=info_step["stdout"],
        console_stderr=info_step["stderr"],
    )

    # Capture visual feedback if applicable
    visual_feedback_base64 = None
    needs_visual = config["use_visual_feedback"] or config["use_img_differencing"]
    if needs_visual and hasattr(env, "render"):
        with turn_timing.phase("visual"):
            vf_base64, vf_img = _get_visual_feedback(env)
            visual_feedback_imgs.append(vf_img)
            visual_feedback_base64_history.append(vf_base64)

            # Also capture wrist camera snapshot for image-based multiview
            if use_wrist and hasattr(env, "render_wrist") and wrist_base64_history is not None:
                wrist_result = _get_visual_feedback(env, use_wrist_camera=True)
                if wrist_result[0] is not None and isinstance(wrist_result[0], list) and len(wrist_result[0]) > 1:
                    wrist_base64_history.append(wrist_result[0][1])  # index 1 = wrist image

    # Determine differencing feedback
    differencing_feedback = None
    is_video_feedback = False

    with turn_timing.phase("differencing"):
        if config.get("use_video_differencing") and turn_frames:
            # Video-based differencing: pass video of this turn to VDM
            differencing_feedback = _get_video_differencing_feedback(
                visual_differencing_args, task_description, turn_frames, wrist_turn_frames,
            )
            is_video_feedback = True
        elif config["use_img_differencing"] and len(visual_feedback_base64_history) >= 2:
            # Image-based differencing: pass before/after images to VDM
            differencing_feedback = _get_visual_differencing_feedback(
                visual_differencing_args, task_description, visual_feedback_base64_history,
                wrist_base64_history=wrist_base64_history,
            )

    # Only pass visual feedback to prompt if visual_feedback is enabled
    if not config["use_visual_feedback"]:
        visual_feedback_base64 = None
    elif needs_visual and hasattr(env, "render"):
        visual_feedback_base64 = visual_feedback_base64_history[-1] if visual_feedback_base64_history else None

    decision_prompt = _build_multi_turn_decision_prompt(
        obs, complete_multi_turn_prompt, visual_feedback_base64, differencing_feedback,
        is_video_feedback=is_video_feedback,
    )

    # Query model
    multiturn_ensemble_entry = None
    with turn_timing.phase("generate"):
        if config["use_parallel_ensemble"]:
            if config.get("use_multimodel", False):
                print("RUNNING MULTITURN MULTIMODEL ENSEMBLE QUERY")
                content = query_model_ensemble(args, decision_prompt, is_multiturn=True)
            else:
                print("RUNNING MULTITURN SINGLE MODEL ENSEMBLE QUERY")
                content = query_single_model_ensemble(args, decision_prompt, args.model, is_multiturn=True)
            multiturn_ensemble_entry = {
                "ensemble_candidates_txt": content.get("ensemble_candidates_txt", ""),
                "ensemble_synthesis_txt": content.get("ensemble_synthesis_txt", ""),
            }
        else:
            content = query_model(args, decision_prompt)

    reasoning = content["reasoning"]
    usage = content.get("usage") if isinstance(content, dict) else None
    decision, new_code = _parse_multi_turn_decision(content["content"])

    return decision, new_code, reasoning, multiturn_ensemble_entry, decision_prompt, usage


# ---------------------------------------------------------------------------
# Core single-trial execution
# ---------------------------------------------------------------------------

def _run_single_trial(
    env: CodeExecutionEnvBase,
    trial: int,
    args: LaunchArgs,
    config: dict[str, Any],
    multi_turn_prompt: str | None,
    partial_artifacts: dict[str, Any] | None = None,
) -> TrialSummary:
    """Execute a single trial end-to-end.

    Steps:
        1. Reset the environment.
        2. Capture initial visual feedback (if configured).
        3. Query the model for initial code generation.
        4. Execute code blocks one-by-one, with optional multi-turn regeneration.
        5. Save artifacts (code, logs, per-turn videos, combined video) and return a TrialSummary.
    """
    trial_start_time = time.time()

    use_video_diff = config.get("use_video_differencing", False)
    use_wrist = config.get("use_wrist_camera", False)

    turn_timing.start_trial(
        config["output_dir"],
        trial,
        config.get("task_name"),
        {
            "model": getattr(args, "model", None),
            "use_parallel_ensemble": config["use_parallel_ensemble"],
            "use_visual_feedback": config["use_visual_feedback"],
            "use_img_differencing": config["use_img_differencing"],
            "use_video_differencing": use_video_diff,
        },
    )

    # --- 1. Reset environment ---
    with turn_timing.phase("reset"):
        obs, _ = env.reset(options={"trial": trial}, seed=trial)
    # Restart the runner's SIGALRM timer AFTER env.reset() so the timeout only covers
    # actual task execution, not scene loading / cuRobo JIT compilation. Imported
    # here because capx.envs.runner imports this module.
    from capx.envs.runner import TRIAL_TIMEOUT_SECONDS

    if signal.alarm(0) > 0:
        signal.alarm(TRIAL_TIMEOUT_SECONDS)
    obs["full_prompt"] = copy.deepcopy(obs["full_prompt"])
    _patch_libero_goal(env, obs)

    if config["record_video"] and hasattr(env, "enable_video_capture"):
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)
    elif use_video_diff and hasattr(env, "enable_video_capture"):
        # Video differencing needs frame recording even without record_video
        env.enable_video_capture(True, clear=True, wrist_camera=use_wrist)

    # --- Shared trial state ---
    code_blocks: list[str] = []
    code_block_metadata: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    stderr_history: list[str] = []
    num_regenerations = 0
    num_finishes = 0
    info_step: dict[str, Any] = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    ensemble_data = None
    multiturn_ensemble_data: list[dict[str, Any]] = []

    # Per-turn frame tracking (for video differencing and per-turn video saving)
    turn_frame_ranges: list[tuple[int, int]] = []

    # Wrist camera base64 history for image-based multiview differencing
    wrist_base64_history: list[str] | None = [] if use_wrist else None

    visual_differencing_args = ModelQueryArgs(
        model=args.visual_differencing_model,
        server_url=args.visual_differencing_model_server_url,
        api_key=args.visual_differencing_model_api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        debug=args.debug,
    )

    # --- 2. Capture initial visual feedback ---
    visual_feedback_imgs, visual_feedback_base64_history, task_description = (
        _capture_initial_visual_feedback(env, obs, config, args, visual_differencing_args)
    )

    # Seed wrist base64 history with initial wrist image
    if use_wrist and wrist_base64_history is not None and hasattr(env, "render_wrist"):
        wrist_img = env.render_wrist()
        if wrist_img is not None:
            pil_wrist = Image.fromarray(wrist_img)
            buf = io.BytesIO()
            pil_wrist.save(buf, format="png")
            wrist_base64_history.append(
                f"data:image/png;base64,"
                f"{base64.b64encode(buf.getvalue()).decode('utf-8')}"
            )

    # --- 3. Initial code generation ---
    if config["use_oracle_code"]:
        raw_code = env.oracle_code
        with open(os.path.join(config["output_dir"], "oracle_code.py"), "w") as f:
            f.write(raw_code)
        reasoning = None
        ensemble_data = None
        initial_usage = None
    else:
        with turn_timing.phase("generate"):
            raw_code, reasoning, ensemble_data, initial_usage = _query_initial_code(args, config, obs)

    # Initialize partial artifacts for timeout recovery
    if partial_artifacts is not None:
        partial_artifacts.update({
            "raw_code": raw_code,
            "code_blocks": code_blocks,
            "code_block_metadata": code_block_metadata,
            "all_responses": all_responses,
            "visual_feedback_imgs": visual_feedback_imgs,
            "info_step": info_step,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "num_regenerations": num_regenerations,
            "num_finishes": num_finishes,
            "num_code_blocks": 0,
            "ensemble_data": ensemble_data,
            "multiturn_ensemble_data": multiturn_ensemble_data,
        })

    # Parse initial code into blocks
    initial_blocks = _extract_code(raw_code)
    code_blocks.extend(initial_blocks)
    code_block_metadata.extend([{"generation": 0, "regenerated": False}] * len(initial_blocks))
    all_responses.append({
        "block_idx": [0],
        "code_blocks": initial_blocks,
        "decision": "initial",
        "initial_prompt": copy.deepcopy(obs["full_prompt"]),
        "reasoning": reasoning if reasoning is not None else "",
        "usage": initial_usage,
    })

    with open(os.path.join(config["output_dir"], "all_responses.json"), "w") as f:
        json.dump(all_responses, f)

    if args.debug:
        with open(os.path.join(config["output_dir"], "code_init.txt"), "w") as f:
            f.write("\n".join(initial_blocks))

    turn_timing.end_turn(decision="initial", blocks=len(initial_blocks))

    # --- 4. Execute code blocks (with optional multi-turn) ---
    info_step = {"sandbox_rc": -1, "stdout": "", "stderr": ""}
    reward = 0.0
    terminated = truncated = False
    code_block_idx = 0
    frame_start = frame_end = 0
    decision: str | None = None

    def end_turn(outcome: str | None) -> None:
        turn_timing.end_turn(
            decision=outcome,
            sandbox_rc=info_step["sandbox_rc"],
            frames=frame_end - frame_start,
        )

    # Track whether we're recording frames (for video diff or record_video)
    recording_frames = (
        (config["record_video"] or use_video_diff)
        and hasattr(env, "get_video_frame_count")
    )

    while code_block_idx < len(code_blocks) and code_block_idx <= MULTITURN_LIMIT:
        code = code_blocks[code_block_idx]
        code_block_idx += 1

        # Record frame index before step
        frame_start = env.get_video_frame_count() if recording_frames else 0

        with turn_timing.phase("execute"):
            obs_next, reward, terminated, truncated, info_step = env.step(code)

        # Record frame index after step
        frame_end = env.get_video_frame_count() if recording_frames else 0
        turn_frame_ranges.append((frame_start, frame_end))

        if partial_artifacts is not None:
            partial_artifacts.update({
                "info_step": info_step,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            })

        obs = obs_next
        if info_step["stderr"]:
            stderr_history.append(info_step["stderr"])

        # The agent's code may catch the error a motion raises on an ended episode, so
        # the flags are what tell; the message covers a local sim whose own horizon
        # comes before max_steps.
        if EPISODE_OVER_MESSAGE in info_step["stderr"]:
            truncated = True
        if terminated or truncated:
            end_turn("terminated")
            break

        # Multi-turn decision. After the last block the loop runs, code regenerated
        # now would never execute, so the (possibly ten-generation) query is skipped.
        if multi_turn_prompt and code_block_idx <= MULTITURN_LIMIT:
            # Get turn frames for video differencing
            turn_frames = None
            wrist_turn_frames = None
            if use_video_diff and recording_frames:
                turn_frames = env.get_video_frames_range(frame_start, frame_end)
                if use_wrist and hasattr(env, "get_wrist_video_frames_range"):
                    wrist_turn_frames = env.get_wrist_video_frames_range(
                        frame_start, frame_end,
                    )

            decision, new_code, mt_reasoning, mt_ensemble, decision_prompt, mt_usage = _handle_multi_turn_step(
                env, obs, args, config, visual_differencing_args,
                multi_turn_prompt, code_blocks, code_block_idx, info_step,
                task_description, visual_feedback_imgs, visual_feedback_base64_history,
                turn_frames=turn_frames,
                wrist_turn_frames=wrist_turn_frames,
                wrist_base64_history=wrist_base64_history,
            )

            if mt_ensemble is not None:
                mt_ensemble["regeneration"] = num_regenerations + 1
                multiturn_ensemble_data.append(mt_ensemble)

            if decision == "regenerate":
                print("Model chose to regenerate code")
                new_blocks = _extract_code(new_code)
                all_responses.append({
                    "multi_turn_prompt": decision_prompt if config.get("save_multiturn_prompts", False) else None,
                    "block_idx": [code_block_idx],
                    "code_blocks": new_blocks,
                    "decision": "regenerate",
                    "reasoning": mt_reasoning if mt_reasoning is not None else "",
                    "usage": mt_usage,
                })
                del code_blocks[code_block_idx:]
                del code_block_metadata[code_block_idx:]
                code_blocks.extend(new_blocks)
                code_block_metadata.extend(
                    [{"generation": num_regenerations + 1, "regenerated": True,
                      "regenerated_at_idx": code_block_idx}]
                    * len(new_blocks)
                )
                num_regenerations += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_regenerations"] = num_regenerations

            elif decision == "finish":
                all_responses.append({
                    "decision": "finish",
                    "reasoning": mt_reasoning if mt_reasoning is not None else (new_code or ""),
                    "usage": mt_usage,
                })
                print("Model chose to finish")
                num_finishes += 1
                if partial_artifacts is not None:
                    partial_artifacts["num_finishes"] = num_finishes
                end_turn("finish")
                break

        print(f"Code block {code_block_idx} done")
        print(f"Number of code blocks: {len(code_blocks)}")

        # Save intermediate artifacts (code, logs) per code block
        with turn_timing.phase("artifacts"):
            final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
            _save_trial_artifacts(
                config, trial, info_step["sandbox_rc"], reward,
                info_step.get("task_completed", False), final_code, raw_code,
                all_responses, ["-" * 100, "Generated program:", final_code],
                visual_feedback_imgs,
            )

            # Only save intermediate video if NOT doing per-turn saving
            # (per-turn saving is deferred to after the loop to avoid clearing the buffer)
            if not recording_frames:
                _save_trial_video(
                    env, config, trial, info_step, reward, len(code_blocks),
                    suffix_extra=str(len(code_blocks)),
                )

        end_turn(decision)

    print("Code blocks done")

    # --- 5. Build final summary ---
    final_code = _annotate_code_blocks(code_blocks, code_block_metadata)
    num_code_blocks = len(code_blocks)

    if partial_artifacts is not None:
        partial_artifacts["final_code"] = final_code
        partial_artifacts["num_code_blocks"] = num_code_blocks

    # Running into the end of the episode is not a failure of the code.
    if EPISODE_OVER_MESSAGE in info_step["stderr"]:
        info_step["sandbox_rc"] = 0

    stderr = "\n\n".join(stderr_history) if stderr_history else info_step["stderr"]
    token_totals = _aggregate_usage(all_responses, getattr(args, "model", None))
    print(
        "[trial] token totals: "
        f"input={token_totals['input_tokens']} "
        f"(cached={token_totals['cached_input_tokens']}) "
        f"output={token_totals['output_tokens']} "
        f"thinking={token_totals['thinking_tokens']} "
        f"calls={token_totals['calls']} "
        f"cost_usd=${token_totals['cost_usd']:.4f} "
        f"model={token_totals['model']}"
    )
    log_lines = _build_log_lines(
        final_code, info_step, reward, terminated, truncated,
        num_regenerations, num_finishes, num_code_blocks,
        stderr_override=stderr,
        token_totals=token_totals,
    )

    code_path = _save_trial_artifacts(
        config, trial, info_step["sandbox_rc"], reward,
        info_step.get("task_completed", False), final_code, raw_code,
        all_responses, log_lines, visual_feedback_imgs,
        ensemble_data=ensemble_data,
        multiturn_ensemble_data=multiturn_ensemble_data,
    )

    # Save per-turn and combined videos
    if recording_frames and turn_frame_ranges:
        _save_turn_and_combined_videos(
            env, config, trial, info_step, reward, turn_frame_ranges,
        )
    else:
        _save_trial_video(env, config, trial, info_step, reward, num_code_blocks)

    success = info_step["sandbox_rc"] == 0

    print(f"Trial {trial} took {time.time() - trial_start_time:.2f} seconds")

    turn_timing.finish_trial(
        trial_s=round(time.time() - trial_start_time, 3),
        reward=reward,
        success=success,
        truncated=truncated,
        num_regenerations=num_regenerations,
        num_code_blocks=num_code_blocks,
    )

    gc.collect()

    return TrialSummary(
        trial=trial,
        success=success,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        sandbox_rc=info_step["sandbox_rc"],
        log="\n".join(log_lines),
        task_completed=info_step.get("task_completed", None),
        code_path=code_path,
        num_regenerations=num_regenerations,
        num_finishes=num_finishes,
        num_code_blocks=num_code_blocks,
    )


def _patch_libero_goal(env: CodeExecutionEnvBase, obs: dict[str, Any]) -> None:
    """Inject the LIBERO task language into the prompt template if applicable.

    ``str.replace``, not ``str.format``: the API docs in the prompt contain braces.
    """
    handle = getattr(env.low_level_env, "handle", None)
    if handle is None or not hasattr(handle, "task_language"):
        return
    content = obs["full_prompt"][-1]["content"][0]
    content["text"] = content["text"].replace("{libero_environment_goal}", handle.task_language)
