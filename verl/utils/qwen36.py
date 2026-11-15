# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Helpers for Qwen3.5/3.6 MoE (model_type=qwen3_5_moe) checkpoints used as text LMs.
from __future__ import annotations

from typing import Any, Optional


def is_qwen3_5_family(config) -> bool:
    model_type = getattr(config, "model_type", None)
    return model_type in ("qwen3_5_moe", "qwen3_5", "qwen3_5_moe_text", "qwen3_5_text")


def get_text_config(config):
    """Return the text tower config when present, else the config itself."""
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def get_max_position_embeddings(config) -> int:
    for candidate in (config, get_text_config(config)):
        value = getattr(candidate, "max_position_embeddings", None)
        if value is not None:
            return int(value)
    raise ValueError("max_position_embeddings not found on model config or text_config")


def get_tie_word_embeddings(config) -> bool:
    for candidate in (config, get_text_config(config)):
        if hasattr(candidate, "tie_word_embeddings"):
            return bool(candidate.tie_word_embeddings)
    return False


def apply_token_id_overrides(config, override_config_kwargs: dict) -> None:
    """Set bos/eos/pad on both top-level and nested text_config when present."""
    for key, val in override_config_kwargs.items():
        setattr(config, key, val)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, key):
            setattr(text_config, key, val)


def resolve_attn_implementation(config, requested: Optional[str] = "flash_attention_2") -> Optional[str]:
    """Qwen3.5/3.6 hybrid GDN layers are happier with sdpa than forced FA2."""
    if is_qwen3_5_family(config):
        return "sdpa"
    return requested
