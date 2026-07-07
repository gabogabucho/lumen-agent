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

# Known litellm-style model prefixes that map to OpenAI-compatible providers.
# The prefix is stripped before sending the `model` field upstream, since
# these providers expect their own bare model ids (not the litellm-routing
# prefix). Prefixes not in this table are passed through unchanged (e.g. a
# custom base_url already pointing at a provider that expects the full
# string, or plain "gpt-4o"-style ids with no prefix at all).
_OPENAI_COMPAT_PREFIXES = ("openai/", "deepseek/", "together_ai/")


def _strip_openai_compat_prefix(model: str) -> str:
    """Strip a known OpenAI-compatible provider prefix from a model id.

    E.g. "deepseek/deepseek-chat" -> "deepseek-chat". Models without a
    recognized prefix are returned unchanged.
    """
    for prefix in _OPENAI_COMPAT_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


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
            parsed.append({"name": function.get("name"), "arguments": arguments})
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
                    )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._map_error(exc) from exc

    async def aclose(self) -> None:
        """Close the underlying httpx client if this instance owns it."""
        if self._owns_client:
            await self._client.aclose()
