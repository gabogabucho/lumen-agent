"""Tests for `build_llm_client()` — the factory + auto-detect routing.

Design ref: sdd/perfil-core/design -> "Client selection" row.
Spec ref: sdd/perfil-core/spec -> Domain: llm-client, Domain: lazy-import.

Phase 4 scope (highest-risk area per apply-progress #2106): this file is
written FIRST (RED), before `build_llm_client()` exists in
`lumen/core/llm_client.py`, per Strict TDD Mode.

Two axes are covered:
1. Explicit selection via `config["llm"]["client"]` always wins over
   auto-detect, regardless of the model string.
2. Auto-detect by model prefix, table-driven over every prefix
   `lumen/core/model_router.py` can emit plus the prefixes documented in the
   design's OpenAI-compat table (`_OPENAI_COMPAT_PREFIXES` in llm_client.py:
   "openai/", "deepseek/", "together_ai/") and known litellm-only providers
   ("anthropic/", "vertex_ai/", "gemini/", "bedrock/", "cohere/", "groq/").

Decision (see mem_save at apply time): `groq/` is NOT in
`_OPENAI_COMPAT_PREFIXES` (OpenAICompatClient's own known-provider table), so
per the design's auto-detect rule ("no `/` prefix matching known
OpenAI-compatible providers, or explicit provider prefixes litellm-only ->
fallback to litellm") it routes to LiteLLMClient, not OpenAICompatClient,
until `_OPENAI_COMPAT_PREFIXES` is extended. This test file asserts the
CURRENT behavior of the table, not an aspirational one.

Unknown/unrecognized prefixes (no "/" at all, e.g. bare "gpt-4o", AND no
explicit `base_url` override) fall back to LiteLLMClient — the safe
multi-provider default per design (litellm already ships as a base
dependency; OpenAICompatClient without an explicit base_url has no
endpoint to call). This is the DOCUMENTED default decision for Phase 4
(design was silent on the no-base_url bare-model-name case).
"""

from __future__ import annotations

import sys

import pytest

from lumen.core.llm_client import (
    LLMClient,
    OpenAICompatClient,
    build_llm_client,
)


@pytest.fixture(autouse=True)
def _clean_litellm_module():
    sys.modules.pop("litellm", None)
    yield
    sys.modules.pop("litellm", None)


# ---------------------------------------------------------------------------
# Explicit selection always wins over auto-detect
# ---------------------------------------------------------------------------


def test_explicit_openai_compat_selection_wins_over_model_prefix():
    """Even an anthropic/-prefixed model must yield OpenAICompatClient when
    explicitly configured, per design: "Explicit selection ... wins"."""
    config = {
        "llm": {"client": "openai_compat", "base_url": "http://localhost:1234"},
        "models": {"default": "anthropic/claude-3-opus"},
    }
    client = build_llm_client(config)
    assert isinstance(client, OpenAICompatClient)
    assert "litellm" not in sys.modules


def test_explicit_litellm_selection_wins_over_model_prefix():
    """Even an openai/-prefixed model must yield LiteLLMClient when
    explicitly configured."""
    config = {
        "llm": {"client": "litellm"},
        "models": {"default": "openai/gpt-4o"},
    }
    client = build_llm_client(config)
    from lumen.core.llm_client_litellm import LiteLLMClient

    assert isinstance(client, LiteLLMClient)


def test_explicit_selection_is_case_and_whitespace_tolerant():
    config = {
        "llm": {"client": " OpenAI_Compat "},
        "models": {"default": "deepseek/deepseek-chat"},
    }
    client = build_llm_client(config)
    assert isinstance(client, OpenAICompatClient)


def test_unknown_explicit_client_value_raises():
    config = {
        "llm": {"client": "not-a-real-client"},
        "models": {"default": "deepseek/deepseek-chat"},
    }
    with pytest.raises(ValueError, match="not-a-real-client"):
        build_llm_client(config)


# ---------------------------------------------------------------------------
# Auto-detect by model prefix (table-driven) — no explicit config["llm"]["client"]
# ---------------------------------------------------------------------------

# (model_string, expected_client_kind) — expected_client_kind is "openai_compat"
# or "litellm". Exhaustive over prefixes model_router.py can emit
# (VALID_ROLES-agnostic; the router only ever returns a model *string*, format
# unchanged) plus the design's documented OpenAI-compat/litellm-only split.
AUTO_DETECT_TABLE = [
    # -- OpenAI-compatible prefixes (per _OPENAI_COMPAT_PREFIXES in llm_client.py)
    pytest.param("openai/gpt-4o", "openai_compat", id="openai-prefix"),
    pytest.param("deepseek/deepseek-chat", "openai_compat", id="deepseek-prefix"),
    pytest.param("together_ai/some-model", "openai_compat", id="together_ai-prefix"),
    # -- litellm-only prefixes (non-OpenAI-compatible providers)
    pytest.param("anthropic/claude-3-opus", "litellm", id="anthropic-prefix"),
    pytest.param("vertex_ai/gemini-1.5-pro", "litellm", id="vertex_ai-prefix"),
    pytest.param("gemini/gemini-1.5-flash", "litellm", id="gemini-prefix"),
    pytest.param("bedrock/anthropic.claude-v2", "litellm", id="bedrock-prefix"),
    pytest.param("cohere/command-r", "litellm", id="cohere-prefix"),
    pytest.param("groq/llama3-70b", "litellm", id="groq-prefix-not-in-compat-table"),
    # -- unknown / no-prefix default (documented decision: safe litellm fallback)
    pytest.param("gpt-4o", "litellm", id="bare-model-name-no-base-url"),
    pytest.param("some-custom-model", "litellm", id="unrecognized-bare-name"),
]


@pytest.mark.parametrize("model, expected_kind", AUTO_DETECT_TABLE)
def test_auto_detect_by_model_prefix(model, expected_kind):
    config = {"models": {"default": model}}
    client = build_llm_client(config)
    if expected_kind == "openai_compat":
        assert isinstance(client, OpenAICompatClient)
        assert "litellm" not in sys.modules
    else:
        from lumen.core.llm_client_litellm import LiteLLMClient

        assert isinstance(client, LiteLLMClient)


def test_auto_detect_bare_model_name_with_explicit_base_url_uses_openai_compat():
    """design: 'bare model names with explicit base_url' -> OpenAICompatClient."""
    config = {
        "llm": {"base_url": "http://localhost:8000/v1"},
        "models": {"default": "my-local-model"},
    }
    client = build_llm_client(config)
    assert isinstance(client, OpenAICompatClient)
    assert "litellm" not in sys.modules


def test_default_config_never_imports_litellm():
    """spec: 'Default config never touches litellm' — building the default
    (openai_compat-eligible) client must not trigger the litellm import."""
    config = {"models": {"default": "deepseek/deepseek-chat"}}
    build_llm_client(config)
    assert "litellm" not in sys.modules


def test_litellm_only_imported_when_litellm_client_selected():
    """The factory itself imports LiteLLMClient's module lazily (inside the
    branch), but LiteLLMClient's own lazy-import guarantee (tested in
    test_litellm_client_lazy_import.py) means `litellm` the package still
    isn't imported merely by building the client — only by calling it."""
    config = {"models": {"default": "anthropic/claude-3-opus"}}
    build_llm_client(config)
    assert "litellm" not in sys.modules


def test_returned_client_satisfies_llm_client_protocol():
    config = {"models": {"default": "deepseek/deepseek-chat"}}
    client = build_llm_client(config)
    assert isinstance(client, LLMClient)


def test_factory_passes_through_base_url_and_api_key_for_openai_compat():
    config = {
        "llm": {
            "client": "openai_compat",
            "base_url": "http://example.com/v1",
            "api_key": "sk-test",
        },
        "models": {"default": "deepseek/deepseek-chat"},
    }
    client = build_llm_client(config)
    assert isinstance(client, OpenAICompatClient)
    assert client._base_url == "http://example.com/v1"
    assert client._api_key == "sk-test"
