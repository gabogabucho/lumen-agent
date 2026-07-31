"""LLMClient abstraction: normalized request/response contract for LLM backends.

Design ref: sdd/perfil-core/design.

Two concrete implementations are planned:
- `OpenAICompatClient` (Phase 2): httpx-based, zero `litellm` import, default
  for new configs pointing at OpenAI-compatible endpoints.
- `LiteLLMClient` (Phase 3): wraps `litellm.acompletion`, with the `litellm`
  import deferred until first use, preserving full multi-provider support.

Phase 1 scope: only the shared contract lives here — `LLMResponse`,
`LLMChunk`, the `LLMClientError` hierarchy, and the `LLMClient` Protocol.
Concrete clients and the `build_llm_client()` factory land in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    """Normalized non-streaming completion result.

    Decouples callers (brain.py, distiller.py) from any single backend's
    raw response schema (e.g. litellm's `.choices[0].message`).
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    role: str = "assistant"


@dataclass
class LLMChunk:
    """Normalized streaming chunk result.

    The concatenation of `delta_content` across all yielded chunks for a
    given `stream()` call MUST equal the final message content.
    """

    delta_content: str | None = None
    delta_tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    # Reasoning/thinking tokens (DeepSeek R1, Claude extended thinking, ...).
    # Optional-with-default so the Phase 1 contract stays backward compatible.
    delta_reasoning_content: str | None = None


class LLMClientError(Exception):
    """Base exception for all LLMClient implementations.

    Callers (e.g. provider_health.py's record_success/failure) branch on
    subclasses without needing to import any backend-specific exception
    module (litellm's or an HTTP client's).
    """


class LLMConnectionError(LLMClientError):
    """Raised when the transport fails to reach the upstream endpoint
    (network error, DNS failure, connection refused, or timeout)."""


class LLMRateLimitError(LLMClientError):
    """Raised when the upstream endpoint responds with HTTP 429."""


class LLMAuthError(LLMClientError):
    """Raised when the upstream endpoint responds with HTTP 401/403."""


class LLMBadRequestError(LLMClientError):
    """Raised when the upstream endpoint responds with HTTP 400
    (malformed request, invalid model, invalid tool schema, etc.)."""


@runtime_checkable
class LLMClient(Protocol):
    """Common interface both `OpenAICompatClient` and `LiteLLMClient` implement.

    Errors from the underlying transport MUST be mapped to the
    `LLMClientError` hierarchy above regardless of implementation.
    """

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """Run a single non-streaming completion."""
        ...

    def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMChunk]:
        """Run a streaming completion, yielding `LLMChunk` objects."""
        ...


# ---------------------------------------------------------------------------
# Phase 2: OpenAICompatClient
# ---------------------------------------------------------------------------
#
# httpx-based implementation calling `POST {base_url}/chat/completions`
# directly against any OpenAI-compatible endpoint. Zero `litellm` import.
# Design ref: sdd/perfil-core/design -> "Default client: OpenAICompatClient".

import json

import httpx

# Known model prefixes that use the OpenAI chat-completions protocol. The
# mapping supplies a sensible endpoint for the providers offered by Lumen's
# wizard, so the Core profile does not need LiteLLM merely to start them.
_OPENAI_COMPAT_BASE_URLS = {
    "openai/": "https://api.openai.com/v1",
    "deepseek/": "https://api.deepseek.com/v1",
    "ollama/": "http://localhost:11434/v1",
    "openrouter/": "https://openrouter.ai/api/v1",
    "together_ai/": "https://api.together.xyz/v1",
}
_OPENAI_COMPAT_PREFIXES = tuple(_OPENAI_COMPAT_BASE_URLS)


def _strip_openai_compat_prefix(model: str) -> str:
    """Strip a known OpenAI-compatible provider prefix from a model id.

    E.g. "deepseek/deepseek-chat" -> "deepseek-chat". Models without a
    recognized prefix are returned unchanged.
    """
    for prefix in _OPENAI_COMPAT_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _default_openai_compat_base_url(model: str) -> str:
    """Return the provider endpoint implied by a known model prefix."""
    for prefix, base_url in _OPENAI_COMPAT_BASE_URLS.items():
        if model.startswith(prefix):
            return base_url
    return ""


class OpenAICompatClient:
    """LLMClient implementation against any OpenAI-compatible `/chat/completions` endpoint.

    Uses `httpx.AsyncClient` exclusively — never imports `litellm`, satisfying
    the lazy-import guarantee (spec: "Default config never touches litellm").
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: "httpx.AsyncClient | None" = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        # Allow an externally constructed httpx.AsyncClient (e.g. wired with
        # a MockTransport in tests) to be injected directly.
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": _strip_openai_compat_prefix(model),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        return payload

    def _map_error(self, exc: Exception) -> LLMClientError:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            if status in (401, 403):
                return LLMAuthError(str(exc))
            if status == 429:
                return LLMRateLimitError(str(exc))
            if status == 400:
                return LLMBadRequestError(str(exc))
            return LLMClientError(str(exc))
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return LLMConnectionError(str(exc))
        return LLMClientError(str(exc))

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw_calls = message.get("tool_calls")
        if not raw_calls:
            return None
        parsed = []
        for call in raw_calls:
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, TypeError):
                    pass
            entry: dict[str, Any] = {
                "name": function.get("name"),
                "arguments": arguments,
            }
            if call.get("id"):
                entry["id"] = call["id"]
            parsed.append(entry)
        return parsed

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        payload = self._build_payload(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        try:
            response = await self._client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._map_error(exc) from exc
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._map_error(exc) from exc

        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message", {})
        return LLMResponse(
            content=message.get("content") or "",
            tool_calls=self._parse_tool_calls(message),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage", {}),
            role=message.get("role", "assistant"),
        )

    async def stream(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[LLMChunk]:
        payload = self._build_payload(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        try:
            async with self._client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=self._headers(),
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    # Ensure the response body is read before constructing
                    # the error message (httpx requires this for streaming
                    # responses accessed outside the context manager).
                    await response.aread()
                    raise self._map_error(exc) from exc

                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:") :].strip()
                    if data_str == "[DONE]":
                        break
                    chunk_data = json.loads(data_str)
                    choice = chunk_data["choices"][0]
                    delta = choice.get("delta", {})
                    yield LLMChunk(
                        delta_content=delta.get("content"),
                        delta_tool_calls=delta.get("tool_calls"),
                        finish_reason=choice.get("finish_reason"),
                        delta_reasoning_content=delta.get("reasoning_content"),
                    )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._map_error(exc) from exc

    async def aclose(self) -> None:
        """Close the underlying httpx client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Phase 4: build_llm_client() factory + auto-detect routing
# ---------------------------------------------------------------------------
#
# Design ref: sdd/perfil-core/design -> "Client selection" row.
# `LiteLLMClient` lives in a SEPARATE module (`llm_client_litellm.py`, Phase 3
# divergence approved by the orchestrator to allow parallel agents). This is
# the canonical import path other code (brain.py, distiller.py, tests) should
# use: `from lumen.core.llm_client import LiteLLMClient` — but the import
# itself is deferred to inside the branch that needs it, so importing this
# module never imports `litellm` (lazy-import guarantee, verified by
# tests/test_litellm_client_lazy_import.py and
# tests/test_llm_client_factory.py::test_default_config_never_imports_litellm).

# Litellm-only provider prefixes: NOT OpenAI-compatible, always routed to
# LiteLLMClient when auto-detecting. This list documents which prefixes the
# design calls out explicitly ("anthropic/claude-*", "vertex_ai/*") plus the
# other major non-OpenAI-compatible providers litellm supports. Any prefix
# not in `_OPENAI_COMPAT_PREFIXES` (defined above) and not explicitly
# OpenAI-compatible falls through to the same LiteLLMClient default — this
# tuple exists for documentation/readability, not as the sole gate.
_LITELLM_ONLY_PREFIXES = (
    "anthropic/",
    "vertex_ai/",
    "gemini/",
    "bedrock/",
    "cohere/",
    "groq/",
)


def _resolve_model_string(config: dict[str, Any]) -> str:
    """Extract the model string the factory should inspect for auto-detect.

    Mirrors `ModelRouter`'s own resolution (`config["models"]["default"]`,
    falling back to legacy `config["model"]`) without importing
    `ModelRouter` itself, since the factory only needs the *string*, not
    role-based routing.
    """
    models_cfg = config.get("models")
    if isinstance(models_cfg, dict) and isinstance(models_cfg.get("default"), str):
        return models_cfg["default"]
    legacy = config.get("model")
    if isinstance(legacy, str):
        return legacy
    return ""


def _auto_detect_client_kind(config: dict[str, Any]) -> str:
    """Return "openai_compat" or "litellm" based on the model string prefix.

    Decision (documented, per apply-time mem_save): the design is silent on
    bare model names (no "/" prefix) with NO explicit `base_url` configured.
    Default: "litellm" — the safe multi-provider fallback, since litellm is
    already a base dependency and OpenAICompatClient has no endpoint to call
    without an explicit `base_url`. A bare model name WITH an explicit
    `base_url` set is treated as OpenAI-compatible (design: "bare model
    names with explicit base_url").
    """
    model = _resolve_model_string(config)
    llm_cfg = config.get("llm")
    has_base_url = isinstance(llm_cfg, dict) and bool(llm_cfg.get("base_url"))

    if model.startswith(_OPENAI_COMPAT_PREFIXES):
        return "openai_compat"
    if model.startswith(_LITELLM_ONLY_PREFIXES):
        return "litellm"
    if "/" not in model:
        # Bare model name: OpenAI-compatible only if the caller supplied an
        # explicit base_url to actually call; otherwise there's no endpoint
        # to route to and litellm is the safe default.
        return "openai_compat" if has_base_url else "litellm"
    # Any other "/"-prefixed provider not in either known table: litellm is
    # the safe multi-provider fallback (documented default decision).
    return "litellm"


def build_llm_client(config: dict[str, Any] | None, **overrides: Any) -> "LLMClient":
    """Build an `LLMClient` implementation from Lumen's config dict.

    Selection order (design: "Client selection"):
      1. Explicit `config["llm"]["client"]` ("openai_compat" | "litellm") —
         always wins, regardless of the configured model string.
      2. Auto-detect by model prefix (see `_auto_detect_client_kind`).

    Raises:
        ValueError: if `config["llm"]["client"]` is set to an unrecognized
            value.
    """
    config = config or {}
    llm_cfg = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    model = _resolve_model_string(config)

    explicit = llm_cfg.get("client")
    if isinstance(explicit, str) and explicit.strip():
        kind = explicit.strip().lower()
    else:
        kind = _auto_detect_client_kind(config)

    if kind == "openai_compat":
        # Persisted Lumen configuration predates the nested ``llm`` block and
        # stores these values at the root. Keep that public config contract
        # intact while allowing nested values to override it.
        base_url = (
            llm_cfg.get("base_url")
            or config.get("api_base")
            or _default_openai_compat_base_url(model)
        )
        api_key = llm_cfg.get("api_key") or config.get("api_key")
        return OpenAICompatClient(
            base_url=base_url,
            api_key=api_key,
            timeout=llm_cfg.get("timeout", 60.0),
        )
    if kind == "litellm":
        # Lazy import: only reached when the litellm-backed client is
        # actually selected, preserving the lazy-import guarantee at the
        # `llm_client` module boundary (never at module top).
        from lumen.core.llm_client_litellm import LiteLLMClient  # noqa: PLC0415

        return LiteLLMClient()

    raise ValueError(
        f"Unknown config['llm']['client'] value: {explicit!r}. "
        "Expected 'openai_compat' or 'litellm'."
    )


def __getattr__(name: str) -> Any:
    """Module-level lazy attribute access (PEP 562).

    Provides the canonical `from lumen.core.llm_client import LiteLLMClient`
    import path (per apply-progress #2106: "factory must define the
    canonical import path") without eagerly importing
    `lumen.core.llm_client_litellm` — and therefore never `litellm` — at
    THIS module's import time. The submodule import only happens when a
    caller actually accesses `LiteLLMClient` off `lumen.core.llm_client`,
    which itself only imports `litellm` lazily inside its own methods.
    """
    if name == "LiteLLMClient":
        from lumen.core.llm_client_litellm import LiteLLMClient  # noqa: PLC0415

        return LiteLLMClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
