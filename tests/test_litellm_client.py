"""Behavioral tests for LiteLLMClient: response/chunk normalization and
error mapping, using a fake `litellm` module injected into `sys.modules`.

These are litellm-specific tests (not the shared contract suite in
tests/test_llm_client_contract.py) because Phase 2's OpenAICompatClient
has not landed in lumen/core/llm_client.py yet at the time this file was
written — wiring LiteLLMClient into the shared parametrized contract suite
is deferred to whichever phase lands second (per the coordination note in
lumen/core/llm_client_litellm.py), or to Phase 4's factory work.
"""

from __future__ import annotations

import sys
import types

import pytest

from lumen.core.llm_client import (
    LLMAuthError,
    LLMBadRequestError,
    LLMChunk,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponse,
)
from lumen.core.llm_client_litellm import LiteLLMClient


class _FakeAuthenticationError(Exception):
    pass


class _FakeRateLimitError(Exception):
    pass


class _FakeBadRequestError(Exception):
    pass


class _FakeAPIConnectionError(Exception):
    pass


class _FakeTimeout(Exception):
    pass


def _install_fake_litellm(monkeypatch, *, acompletion):
    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = acompletion
    fake_module.AuthenticationError = _FakeAuthenticationError
    fake_module.RateLimitError = _FakeRateLimitError
    fake_module.BadRequestError = _FakeBadRequestError
    fake_module.APIConnectionError = _FakeAPIConnectionError
    fake_module.Timeout = _FakeTimeout
    monkeypatch.setitem(sys.modules, "litellm", fake_module)
    return fake_module


SAMPLE_MESSAGES = [{"role": "user", "content": "Hello, world!"}]


async def test_complete_normalizes_response(monkeypatch):
    async def fake_acompletion(**kwargs):
        message = types.SimpleNamespace(
            content="Hi there!", role="assistant", tool_calls=None
        )
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    client = LiteLLMClient()
    response = await client.complete(
        model="test-model", messages=SAMPLE_MESSAGES, max_tokens=256
    )

    assert isinstance(response, LLMResponse)
    assert response.content == "Hi there!"
    assert response.finish_reason == "stop"
    assert response.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


async def test_complete_normalizes_tool_calls(monkeypatch):
    async def fake_acompletion(**kwargs):
        function = types.SimpleNamespace(
            name="get_weather", arguments='{"location": "Paris"}'
        )
        tool_call = types.SimpleNamespace(function=function)
        message = types.SimpleNamespace(
            content="", role="assistant", tool_calls=[tool_call]
        )
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    client = LiteLLMClient()
    response = await client.complete(
        model="test-model",
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        max_tokens=256,
    )

    assert response.tool_calls == [
        {"name": "get_weather", "arguments": '{"location": "Paris"}'}
    ]


async def test_stream_yields_chunks_concatenating_to_full_content(monkeypatch):
    async def fake_stream():
        for text, finish in [("Hel", None), ("lo!", "stop")]:
            delta = types.SimpleNamespace(content=text, tool_calls=None)
            choice = types.SimpleNamespace(delta=delta, finish_reason=finish)
            yield types.SimpleNamespace(choices=[choice])

    async def fake_acompletion(**kwargs):
        assert kwargs["stream"] is True
        return fake_stream()

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    client = LiteLLMClient()
    chunks = []
    async for chunk in client.stream(
        model="test-model", messages=SAMPLE_MESSAGES, max_tokens=256
    ):
        assert isinstance(chunk, LLMChunk)
        chunks.append(chunk)

    assert len(chunks) == 2
    concatenated = "".join(c.delta_content or "" for c in chunks)
    assert concatenated == "Hello!"
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.parametrize(
    "fake_exc_cls,expected_error",
    [
        (_FakeAuthenticationError, LLMAuthError),
        (_FakeRateLimitError, LLMRateLimitError),
        (_FakeBadRequestError, LLMBadRequestError),
        (_FakeAPIConnectionError, LLMConnectionError),
        (_FakeTimeout, LLMConnectionError),
    ],
)
async def test_complete_maps_litellm_errors(
    monkeypatch, fake_exc_cls, expected_error
):
    async def fake_acompletion(**kwargs):
        raise fake_exc_cls("boom")

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    client = LiteLLMClient()
    with pytest.raises(expected_error):
        await client.complete(
            model="test-model", messages=SAMPLE_MESSAGES, max_tokens=256
        )


async def test_stream_maps_litellm_errors(monkeypatch):
    async def fake_acompletion(**kwargs):
        raise _FakeRateLimitError("rate limited")

    _install_fake_litellm(monkeypatch, acompletion=fake_acompletion)

    client = LiteLLMClient()
    with pytest.raises(LLMRateLimitError):
        async for _ in client.stream(
            model="test-model", messages=SAMPLE_MESSAGES, max_tokens=256
        ):
            pass
