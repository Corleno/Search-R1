"""Resolve SDPO self-distillation reprompt templates from config or files."""

from __future__ import annotations

from typing import Mapping

_TEMPLATE_KEYS = (
    "reprompt_template",
    "solution_template",
    "feedback_template",
)


def _read_template_file(path: str, template_key: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"SDPO {template_key}_file not found: {path}"
        ) from exc


def resolve_self_distillation_templates(cfg: Mapping) -> dict[str, str]:
    """Return reprompt/solution/feedback templates, loading from file when configured."""
    resolved: dict[str, str] = {}
    for template_key in _TEMPLATE_KEYS:
        file_key = f"{template_key}_file"
        template_file = cfg.get(file_key)
        if template_file:
            resolved[template_key] = _read_template_file(str(template_file), template_key)
        else:
            resolved[template_key] = cfg[template_key]
    return resolved
