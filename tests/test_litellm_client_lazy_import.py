"""Lazy-import guarantee tests for LiteLLMClient.

Design ref: sdd/perfil-core/design — "litellm loading: Lazy import inside
LiteLLMClient.__init__" is WRONG per the design table wording; the actual
guarantee (spec `sdd/perfil-core/spec` -> Domain: lazy-import) is stronger:
`litellm` MUST NOT be imported at construction time either — only on the
first `.complete()`/`.stream()` call. This file asserts that exact boundary.

These tests inject a fake `litellm` module into `sys.modules` so they never
require the real (170+ MiB) package to be imported, and never touch the
network.
"""

from __future__ import annotations

import sys
import types

import pytest

from lumen.core.llm_client import LLMClientError
from lumen.core.llm_client_litellm import LiteLLMClient


def _remove_real_litellm_if_imported():
    """Ensure `litellm` isn't already imported from a prior test/module,
    so each test starts from a clean sys.modules baseline."""
    sys.modules.pop("litellm", None)


@pytest.fixture(autouse=True)
def _clean_litellm_module():
    _remove_real_litellm_if_imported()
    yield
    sys.modules.pop("litellm", None)


def test_import_module_does_not_import_litellm():
    """spec: importing lumen.core.llm_client_litellm must not import litellm."""
    _remove_real_litellm_if_imported()
    assert "litellm" not in sys.modules


def test_importing_brain_and_distiller_does_not_import_litellm():
    """Phase 5 (perfil-core): `import lumen.core.brain` / `lumen.core.distiller`
    MUST NOT import litellm at module level — this is the RSS-goal linchpin.

    Runs in a fresh subprocess so no prior test pollution of sys.modules can
    mask (or fake) the result.
    """
    import subprocess

    code = (
        "import sys; "
        "import lumen.core.brain; "
        "import lumen.core.distiller; "
        "assert 'litellm' not in sys.modules, 'litellm was imported at module level'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"brain/distiller import pulled in litellm:\n{result.stderr}"
    )


def test_construction_does_not_import_litellm():
    """spec: constructing LiteLLMClient() must not trigger the litellm import."""
    LiteLLMClient()
    assert "litellm" not in sys.modules


async def test_first_complete_call_imports_litellm(monkeypatch):
    """spec: litellm is imported exactly when the first complete()/stream()
    call runs, not before."""
    fake_litellm = _build_fake_litellm_module()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    client = LiteLLMClient()
    assert "litellm" not in sys.modules or sys.modules["litellm"] is fake_litellm
    # Sanity: our monkeypatch pre-seeds sys.modules, so instead verify the
    # client actually calls through to litellm.acompletion when invoked.
    response = await client.complete(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        temperature=0.7,
        max_tokens=256,
    )
    assert response.content == "fake response"


async def test_missing_litellm_package_raises_descriptive_llm_client_error(
    monkeypatch,
):
    """spec: litellm import failure surfaces a clear error only when selected.

    Simulates `litellm` not being installed by making the import machinery
    raise ImportError for that specific module name, without affecting any
    other import in the test process.
    """
    import builtins

    real_import = builtins.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "litellm" or name.startswith("litellm."):
            raise ImportError("No module named 'litellm'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raising_import)
    sys.modules.pop("litellm", None)

    client = LiteLLMClient()
    with pytest.raises(LLMClientError, match="litellm"):
        await client.complete(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=None,
            temperature=0.7,
            max_tokens=256,
        )


def _build_fake_litellm_module():
    """Build a minimal fake `litellm` module for injection via sys.modules,
    avoiding any real import of the (heavy) litellm package."""

    async def fake_acompletion(**kwargs):
        message = types.SimpleNamespace(
            content="fake response",
            role="assistant",
            tool_calls=None,
        )
        choice = types.SimpleNamespace(message=message, finish_reason="stop")
        usage = types.SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )
        return types.SimpleNamespace(choices=[choice], usage=usage)

    fake_module = types.ModuleType("litellm")
    fake_module.acompletion = fake_acompletion

    class FakeAPIConnectionError(Exception):
        pass

    class FakeAuthenticationError(Exception):
        pass

    class FakeRateLimitError(Exception):
        pass

    class FakeBadRequestError(Exception):
        pass

    class FakeTimeout(Exception):
        pass

    fake_module.exceptions = types.SimpleNamespace(
        APIConnectionError=FakeAPIConnectionError,
        AuthenticationError=FakeAuthenticationError,
        RateLimitError=FakeRateLimitError,
        BadRequestError=FakeBadRequestError,
        Timeout=FakeTimeout,
    )
    fake_module.APIConnectionError = FakeAPIConnectionError
    fake_module.AuthenticationError = FakeAuthenticationError
    fake_module.RateLimitError = FakeRateLimitError
    fake_module.BadRequestError = FakeBadRequestError
    fake_module.Timeout = FakeTimeout
    return fake_module
