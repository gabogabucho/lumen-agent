"""Canonical configured-model to capability-tier resolver."""

from __future__ import annotations

MODEL_TIER_UNKNOWN = "unknown"

_TIER_RANKS = {
    "tier-1": 1,
    "tier-2": 2,
    "tier-3": 3,
}

_EXACT_MODEL_TIERS = {
    "deepseek/deepseek-chat": "tier-1",
    "deepseek-chat": "tier-1",
    "gpt-4o-mini": "tier-2",
    "meta-llama/llama-3.3-70b-instruct": "tier-2",
    "ollama/llama3": "tier-1",
    "claude-sonnet-4-20250514": "tier-3",
}

_PREFIX_MODEL_TIERS = (
    ("claude-sonnet-4", "tier-3"),
    ("claude-3-7-sonnet", "tier-3"),
    ("claude-3-5-sonnet", "tier-2"),
    ("gpt-4.1", "tier-3"),
    ("gpt-4o", "tier-3"),
    ("o3", "tier-3"),
    ("o4", "tier-3"),
    ("gemini-2.5-pro", "tier-3"),
    ("gemini-1.5-pro", "tier-2"),
    ("meta-llama/llama-3.3-70b", "tier-2"),
    ("ollama/llama3", "tier-1"),
)


def resolve_configured_model_tier(model: str | None) -> str:
    """Resolve a configured model to a canonical tier or safe unknown."""
    normalized = _normalize_model_name(model)
    if not normalized:
        return MODEL_TIER_UNKNOWN

    exact = _EXACT_MODEL_TIERS.get(normalized)
    if exact:
        return exact

    for prefix, tier in _PREFIX_MODEL_TIERS:
        if normalized.startswith(prefix):
            return tier

    return MODEL_TIER_UNKNOWN


def normalize_capability_tier(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _TIER_RANKS else MODEL_TIER_UNKNOWN


def is_model_tier_below_minimum(
    model_tier: str | None, minimum_tier: str | None
) -> bool:
    resolved_model_tier = normalize_capability_tier(model_tier)
    resolved_minimum_tier = normalize_capability_tier(minimum_tier)
    if (
        resolved_model_tier == MODEL_TIER_UNKNOWN
        or resolved_minimum_tier == MODEL_TIER_UNKNOWN
    ):
        return False
    return _TIER_RANKS[resolved_model_tier] < _TIER_RANKS[resolved_minimum_tier]


def _normalize_model_name(model: str | None) -> str:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return ""
    if ":" in normalized:
        normalized = normalized.split(":", 1)[0]
    return normalized
