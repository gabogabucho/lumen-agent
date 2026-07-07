"""Shared contract tests for LLMClient implementations.

Both `OpenAICompatClient` and `LiteLLMClient` (Phase 2/3) MUST pass this
identical parametrized suite with identical assertions and no
implementation-specific branches, per spec `sdd/perfil-core/spec` ->
Domain: llm-client -> Scenario: "Both implementations satisfy the same
contract test suite".

Phase 2 status: `OpenAICompatClient` is implemented (httpx.MockTransport-based
fixtures below) and this suite is GREEN for that implementation.

Phase 3 (LiteLLMClient) is implemented in `lumen/core/llm_client_litellm.py`
(a separate module to avoid edit collisions with Phase 2's work on
`llm_client.py`). Until Phase 4's factory (or a later phase) adds the
`from lumen.core.llm_client_litellm import LiteLLMClient` re-export line to
`llm_client.py`, `LiteLLMClient` is imported defensively here too, falling
back to the dedicated module directly so this suite exercises it even
before that re-export lands.
"""

import json
import sys
import types

import httpx
import pytest

from lumen.core.llm_client import (
    LLMAuthError,
    LLMBadRequestError,
    LLMChunk,
    LLMClientError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponse,
    OpenAICompatClient,
)

try:
    from lumen.core.llm_client import LiteLLMClient
except ImportError:  # Re-export not landed on llm_client.py yet.
    try:
        from lumen.core.llm_client_litellm import LiteLLMClient
    except ImportError:  # Phase 3 module itself not present.
        LiteLLMClient = None

CLIENT_CLASSES = [pytest.param(OpenAICompatClient, id="OpenAICompatClient")]
if LiteLLMClient is not None:
    CLIENT_CLASSES.append(pytest.param(LiteLLMClient, id="LiteLLMClient"))


@pytest.fixture(autouse=True)
def _clean_fake_litellm_module():
    """Ensure the fake `litellm` module injected for LiteLLMClient scenarios
    doesn't leak across tests -- each test starts from a clean baseline,
    same as tests/test_litellm_client_lazy_import.py."""
    sys.modules.pop("litellm", None)
    yield
    sys.modules.pop("litellm", None)

SAMPLE_MESSAGES = [{"role": "user", "content": "Hello, world!"}]

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Fixture payloads (OpenAI chat/completions-shaped) shared across scenarios.
# Keyed by scenario name so both implementations receive identical mocked
# responses -- only the transport-wiring differs per client class.
# ---------------------------------------------------------------------------

_NON_STREAMING_RESPONSE = {
    "id": "chatcmpl-test",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello back!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}

_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-test-tool",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"location": "Paris"}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_STREAM_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {"content": " back!"}, "finish_reason": None}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
]


def _sse_body(chunks: list[dict]) -> bytes:
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks]
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _openai_compat_transport(scenario: str) -> httpx.MockTransport:
    """Build an httpx.MockTransport for OpenAICompatClient per scenario."""

    def handler(request: httpx.Request) -> httpx.Response:
        if scenario == "success":
            return httpx.Response(200, json=_NON_STREAMING_RESPONSE)
        if scenario == "tool_call":
            return httpx.Response(200, json=_TOOL_CALL_RESPONSE)
        if scenario == "stream":
            return httpx.Response(
                200,
                content=_sse_body(_STREAM_CHUNKS),
                headers={"content-type": "text/event-stream"},
            )
        if scenario == "http_401":
            return httpx.Response(401, json={"error": "unauthorized"})
        if scenario == "http_429":
            return httpx.Response(429, json={"error": "rate limited"})
        if scenario == "http_400":
            return httpx.Response(400, json={"error": "bad request"})
        if scenario == "connection_error":
            raise httpx.ConnectError("mock connection failure", request=request)
        raise ValueError(f"Unknown scenario: {scenario}")

    return httpx.MockTransport(handler)


def _litellm_response_from(scenario: str):
    """Build a fake litellm.acompletion() non-streaming return value
    mirroring the same fixture payloads used for OpenAICompatClient, keyed
    by the same `scenario` names, so both branches exercise equivalent
    upstream behavior."""
    import litellm  # the fake module injected into sys.modules by the caller

    if scenario == "success":
        msg = _NON_STREAMING_RESPONSE["choices"][0]["message"]
        message = types.SimpleNamespace(
            content=msg["content"], role=msg["role"], tool_calls=None
        )
        choice = types.SimpleNamespace(
            message=message,
            finish_reason=_NON_STREAMING_RESPONSE["choices"][0]["finish_reason"],
        )
        usage_dict = _NON_STREAMING_RESPONSE["usage"]
        usage = types.SimpleNamespace(**usage_dict)
        return types.SimpleNamespace(choices=[choice], usage=usage)
    if scenario == "tool_call":
        raw_msg = _TOOL_CALL_RESPONSE["choices"][0]["message"]
        tool_calls = [
            types.SimpleNamespace(
                function=types.SimpleNamespace(
                    name=tc["function"]["name"],
                    arguments=tc["function"]["arguments"],
                )
            )
            for tc in raw_msg["tool_calls"]
        ]
        message = types.SimpleNamespace(
            content=raw_msg["content"], role=raw_msg["role"], tool_calls=tool_calls
        )
        choice = types.SimpleNamespace(
            message=message,
            finish_reason=_TOOL_CALL_RESPONSE["choices"][0]["finish_reason"],
        )
        usage = types.SimpleNamespace(**_TOOL_CALL_RESPONSE["usage"])
        return types.SimpleNamespace(choices=[choice], usage=usage)
    if scenario == "http_401":
        raise litellm.AuthenticationError("unauthorized")
    if scenario == "http_429":
        raise litellm.RateLimitError("rate limited")
    if scenario == "http_400":
        raise litellm.BadRequestError("bad request")
    if scenario == "connection_error":
        raise litellm.APIConnectionError("mock connection failure")
    raise ValueError(f"Unknown scenario: {scenario}")


def _install_fake_litellm_for_contract(scenario: str):
    """Inject a fake `litellm` module into sys.modules wired to `scenario`,
    covering both non-streaming (acompletion) and streaming (async
    generator) call shapes."""

    class _FakeAuthenticationError(Exception):
        pass

    class _FakeRateLimitError(Exception):
        pass

    class _FakeBadRequestError(Exception):
        pass

    class _FakeAPIConnectionError(Exception):
        pass

    async def _fake_stream():
        for chunk_data in _STREAM_CHUNKS:
            delta_data = chunk_data["choices"][0]["delta"]
            delta = types.SimpleNamespace(
                content=delta_data.get("content"),
                tool_calls=delta_data.get("tool_calls"),
            )
            choice = types.SimpleNamespace(
                delta=delta, finish_reason=chunk_data["choices"][0]["finish_reason"]
            )
            yield types.SimpleNamespace(choices=[choice])

    async def fake_acompletion(**kwargs):
        if kwargs.get("stream"):
            if scenario in ("http_401", "http_429", "http_400", "connection_error"):
                _litellm_response_from(scenario)  # raises
            return _fake_stream()
        return _litellm_response_from(scenario)

    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = fake_acompletion
    fake_module.AuthenticationError = _FakeAuthenticationError
    fake_module.RateLimitError = _FakeRateLimitError
    fake_module.BadRequestError = _FakeBadRequestError
    fake_module.APIConnectionError = _FakeAPIConnectionError
    fake_module.Timeout = _FakeAPIConnectionError
    sys.modules["litellm"] = fake_module
    return fake_module


def _make_client(client_cls, scenario: str = "success"):
    """Construct a client instance wired to a mocked transport for `scenario`.

    Symmetric per-implementation construction:
    - `OpenAICompatClient`: httpx.AsyncClient with a MockTransport tuned to
      `scenario` is injected via the `client=` constructor kwarg.
    - `LiteLLMClient`: a fake `litellm` module is injected into
      `sys.modules` (never the real package), scenario-keyed to the same
      fixture data used for `OpenAICompatClient`, exercised on first
      `.complete()`/`.stream()` call per the lazy-import contract.
    """
    if client_cls is OpenAICompatClient:
        transport = _openai_compat_transport(scenario)
        http_client = httpx.AsyncClient(
            transport=transport, base_url="https://example.invalid/v1"
        )
        return OpenAICompatClient(
            base_url="https://example.invalid/v1",
            api_key="test-key",
            client=http_client,
        )
    if LiteLLMClient is not None and client_cls is LiteLLMClient:
        _install_fake_litellm_for_contract(scenario)
        return LiteLLMClient()
    raise ValueError(f"Unknown client class: {client_cls}")


@pytest.mark.parametrize("client_cls", CLIENT_CLASSES)
class TestLLMClientContract:
    """Identical assertions run against every LLMClient implementation."""

    async def test_non_streaming_completion_succeeds(self, client_cls):
        """spec: Non-streaming completion succeeds."""
        client = _make_client(client_cls, scenario="success")
        response = await client.complete(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tools=None,
            temperature=0.7,
            max_tokens=256,
        )
        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, str)
        assert response.usage is not None
        assert "prompt_tokens" in response.usage
        assert "completion_tokens" in response.usage
        assert "total_tokens" in response.usage

    async def test_streaming_completion_yields_chunks(self, client_cls):
        """spec: Streaming completion yields chunks; concatenation equals final content."""
        client = _make_client(client_cls, scenario="stream")
        chunks = []
        async for chunk in client.stream(
            model="test-model",
            messages=SAMPLE_MESSAGES,
            tools=None,
            temperature=0.7,
            max_tokens=256,
        ):
            assert isinstance(chunk, LLMChunk)
            chunks.append(chunk)

        assert len(chunks) > 0
        concatenated = "".join(c.delta_content or "" for c in chunks)
        assert isinstance(concatenated, str)
        assert concatenated != ""

    async def test_tool_call_request_passed_through(self, client_cls):
        """spec: Tool call request is passed through and parsed identically."""
        client = _make_client(client_cls, scenario="tool_call")
        response = await client.complete(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=SAMPLE_TOOLS,
            temperature=0.7,
            max_tokens=256,
        )
        assert isinstance(response, LLMResponse)
        if response.tool_calls:
            for call in response.tool_calls:
                assert "name" in call
                assert "arguments" in call

    @pytest.mark.parametrize(
        "http_status,expected_error",
        [
            (401, LLMAuthError),
            (429, LLMRateLimitError),
            (400, LLMBadRequestError),
        ],
    )
    async def test_upstream_error_mapped_to_common_exception(
        self, client_cls, http_status, expected_error
    ):
        """spec: Upstream error is mapped to common exception (never a raw provider exception)."""
        client = _make_client(client_cls, scenario=f"http_{http_status}")
        with pytest.raises(expected_error):
            await client.complete(
                model="test-model",
                messages=SAMPLE_MESSAGES,
                tools=None,
                temperature=0.7,
                max_tokens=256,
            )

    async def test_connection_error_mapped(self, client_cls):
        """spec: transport-level connection failure maps to LLMConnectionError."""
        client = _make_client(client_cls, scenario="connection_error")
        with pytest.raises(LLMConnectionError):
            await client.complete(
                model="test-model",
                messages=SAMPLE_MESSAGES,
                tools=None,
                temperature=0.7,
                max_tokens=256,
            )

    async def test_errors_are_llm_client_error_subclasses(self, client_cls):
        """All mapped exceptions must subclass LLMClientError for uniform except-handling."""
        assert issubclass(LLMAuthError, LLMClientError)
        assert issubclass(LLMRateLimitError, LLMClientError)
        assert issubclass(LLMBadRequestError, LLMClientError)
        assert issubclass(LLMConnectionError, LLMClientError)
