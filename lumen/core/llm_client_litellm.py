"""LiteLLMClient: `LLMClient` adapter wrapping `litellm.acompletion`.

Design ref: sdd/perfil-core/design, sdd/perfil-core/spec (Domain: lazy-import).

CRITICAL lazy-import guarantee: this module MUST NOT import `litellm` at
module load time, and `LiteLLMClient()` construction MUST NOT import it
either. `litellm` is only imported inside the first `complete()`/`stream()`
call (module-level `_get_litellm()` caches the imported module on the
instance so repeated calls don't re-trigger import machinery).

Kept in a SEPARATE file from `lumen/core/llm_client.py` (Phase 4's factory
will re-export `LiteLLMClient` from there) so Phase 2's `OpenAICompatClient`
work on `llm_client.py` can land in parallel without edit collisions. The
shared contract test suite (`tests/test_llm_client_contract.py`) imports
`LiteLLMClient` from `lumen.core.llm_client` — that re-export is added later
(by whichever phase lands second, or Phase 4's factory), and MUST be a plain
`from lumen.core.llm_client_litellm import LiteLLMClient` line, never an
eager `import litellm`, to preserve the laziness guarantee at the
`llm_client` module boundary too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncIterator

from lumen.core.llm_client import (
    LLMAuthError,
    LLMBadRequestError,
    LLMChunk,
    LLMClientError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponse,
)

if TYPE_CHECKING:
    # Only for type checkers/IDEs — never executed at runtime, so it does
    # not violate the lazy-import guarantee.
    import litellm as litellm_module


class LiteLLMClient:
    """`LLMClient` implementation backed by `litellm.acompletion`.

    The `litellm` package (~170 MiB import weight) is imported lazily: not
    at module import time, not at `__init__` time, only on the first call
    to `complete()` or `stream()`. Subsequent calls reuse the cached
    reference on `self._litellm`.
    """

    def __init__(self) -> None:
        # MUST stay None until the first complete()/stream() call — this
        # is the field the lazy-import guarantee tests assert against
        # indirectly via `sys.modules`.
        self._litellm: Any | None = None

    def _get_litellm(self) -> "litellm_module":
        """Import `litellm` on first use and cache it on the instance.

        Raises a descriptive `LLMClientError` (not a raw `ImportError`) if
        the `litellm` package is not installed, per spec scenario "litellm
        import failure surfaces a clear error only when selected".
        """
        if self._litellm is None:
            try:
                import litellm  # noqa: PLC0415 (intentional lazy import)
            except ImportError as exc:
                raise LLMClientError(
                    "LiteLLMClient requires the 'litellm' package, which is "
                    "not installed. Install it (it is a base dependency of "
                    "this project) or select the 'openai_compat' client "
                    "instead via config['llm']['client']."
                ) from exc
            self._litellm = litellm
        return self._litellm

    async def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        litellm = self._get_litellm()
        try:
            raw = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001 (mapped below)
            raise _map_litellm_error(litellm, exc) from exc

        choice = raw.choices[0]
        message = choice.message
        usage = getattr(raw, "usage", None)
        usage_dict = (
            {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            if usage is not None
            else {}
        )

        tool_calls = _normalize_tool_calls(getattr(message, "tool_calls", None))

        return LLMResponse(
            content=getattr(message, "content", "") or "",
            tool_calls=tool_calls,
            finish_reason=getattr(choice, "finish_reason", None),
            usage=usage_dict,
            role=getattr(message, "role", "assistant") or "assistant",
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
        litellm = self._get_litellm()
        try:
            stream_resp = await litellm.acompletion(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001 (mapped below)
            raise _map_litellm_error(litellm, exc) from exc

        try:
            async for raw_chunk in stream_resp:
                choice = raw_chunk.choices[0]
                delta = choice.delta
                yield LLMChunk(
                    delta_content=getattr(delta, "content", None),
                    delta_tool_calls=_normalize_tool_calls(
                        getattr(delta, "tool_calls", None)
                    ),
                    finish_reason=getattr(choice, "finish_reason", None),
                )
        except Exception as exc:  # noqa: BLE001 (mapped below)
            raise _map_litellm_error(litellm, exc) from exc


def _normalize_tool_calls(
    raw_tool_calls: Any,
) -> list[dict[str, Any]] | None:
    """Normalize litellm's tool_calls shape to the common
    `{"name": ..., "arguments": ...}` contract shape."""
    if not raw_tool_calls:
        return None

    normalized: list[dict[str, Any]] = []
    for call in raw_tool_calls:
        function = getattr(call, "function", None)
        if function is not None:
            name = getattr(function, "name", None)
            arguments = getattr(function, "arguments", None)
        elif isinstance(call, dict):
            function_dict = call.get("function", {})
            name = function_dict.get("name")
            arguments = function_dict.get("arguments")
        else:
            name = None
            arguments = None
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def _map_litellm_error(litellm: Any, exc: Exception) -> LLMClientError:
    """Map a litellm-raised exception to the common LLMClientError taxonomy.

    litellm re-exports OpenAI-style exception classes at the top level
    (`litellm.AuthenticationError`, `litellm.RateLimitError`,
    `litellm.BadRequestError`, `litellm.APIConnectionError`,
    `litellm.Timeout`), so we check against those by attribute lookup
    (defensively, in case a given litellm version doesn't expose one).
    """
    auth_error = getattr(litellm, "AuthenticationError", None)
    rate_limit_error = getattr(litellm, "RateLimitError", None)
    bad_request_error = getattr(litellm, "BadRequestError", None)
    connection_error = getattr(litellm, "APIConnectionError", None)
    timeout_error = getattr(litellm, "Timeout", None)

    if auth_error is not None and isinstance(exc, auth_error):
        return LLMAuthError(str(exc))
    if rate_limit_error is not None and isinstance(exc, rate_limit_error):
        return LLMRateLimitError(str(exc))
    if bad_request_error is not None and isinstance(exc, bad_request_error):
        return LLMBadRequestError(str(exc))
    if timeout_error is not None and isinstance(exc, timeout_error):
        return LLMConnectionError(str(exc))
    if connection_error is not None and isinstance(exc, connection_error):
        return LLMConnectionError(str(exc))

    # Fallback: status-code sniffing for exceptions that don't map cleanly
    # onto litellm's own hierarchy (defensive; keeps mapping honest instead
    # of silently swallowing unknown errors into LLMConnectionError).
    status_code = getattr(exc, "status_code", None)
    if status_code == 401 or status_code == 403:
        return LLMAuthError(str(exc))
    if status_code == 429:
        return LLMRateLimitError(str(exc))
    if status_code == 400:
        return LLMBadRequestError(str(exc))

    return LLMConnectionError(str(exc))
