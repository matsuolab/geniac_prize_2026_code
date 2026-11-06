"""LLM client module for querying language models.

This module provides utilities for querying various LLM providers
(OpenAI, Claude, open-source models behind an OpenAI-compatible server) with
support for streaming and ensemble queries.
"""

from capx.llm.client import (
    CLAUDE_MODELS,
    ENSEMBLE_CONFIGS,
    GPT_MODELS,
    VLM_MODELS,
    ModelQueryArgs,
    collapse_text_image_inputs,
    parse_multi_turn_decision,
    query_model,
    query_model_ensemble,
    query_model_streaming,
    query_single_model_ensemble,
)

__all__ = [
    "CLAUDE_MODELS",
    "ENSEMBLE_CONFIGS",
    "GPT_MODELS",
    "VLM_MODELS",
    "ModelQueryArgs",
    "collapse_text_image_inputs",
    "parse_multi_turn_decision",
    "query_model",
    "query_model_ensemble",
    "query_model_streaming",
    "query_single_model_ensemble",
]
