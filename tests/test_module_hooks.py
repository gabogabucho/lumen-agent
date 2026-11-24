"""Public module hooks: proactive messages and durable system events.

These tests exercise the surface a kit outside of a chat turn uses:
``ModuleRuntimeContext.notify(...)`` and
``ModuleRuntimeContext.record_system_event(...)``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lumen.core.awareness import CapabilityAwareness
from lumen.core.brain import Brain
from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.consciousness import Consciousness
from lumen.core.llm_client import LLMChunk, LLMResponse
from lumen.core.memory import Memory
from lumen.core.module_runtime import ModuleRuntimeContext
from lumen.core.personality import Personality
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry
from lumen.core.session import Session


class RecordingLLM:
    """Minimal LLMClient that records every completion request."""

    def __init__(self, content: str = "ok"):
        self.content = content
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(content=self.content)

    async def stream(self, **kwargs):
        response = await self.complete(**kwargs)
        yield LLMChunk(delta_content=response.content, finish_reason="stop")


async def _make_brain(tmp_path: Path, *, with_memory: bool = True):
    personality_path = tmp_path / "personality.yaml"
    personality_path.write_text(
        yaml.safe_dump(
            {
                "identity": {"name": "TestLumen", "role": "Test Assistant"},
                "tone": {"style": "direct"},
            }
        ),
        encoding="utf-8",
    )
    memory = None
    if with_memory:
        memory = Memory(tmp_path / "memory.db")
        await memory.init()
    llm = RecordingLLM()
    brain = Brain(
        consciousness=Consciousness(),
        personality=Personality(personality_path),
        memory=memory,
        connectors=ConnectorRegistry(),
        registry=Registry(),
        catalog=Catalog(),
        model="test-model",
        llm_client=llm,
    )
    return brain, memory, llm


def _make_context(brain: Brain | None, memory: Memory | None, tmp_path: Path) -> ModuleRuntimeContext:
    return ModuleRuntimeContext(
        name="test-kit",
        module_dir=tmp_path / "module",
        runtime_dir=tmp_path / "runtime",
        manifest={},
        config={},
        brain=brain,
        memory=memory,
    )


def _captured_system_prompt(llm: RecordingLLM) -> str:
    assert llm.calls, "the LLM was never called"
    return llm.calls[-1]["messages"][0]["content"]


async def test_notify_reaches_registered_sink_and_returns_recipient_count(tmp_path):
    brain, memory, _ = await _make_brain(tmp_path)
    received: list[dict] = []

    async def web_sink(message):
        received.append({"text": message.text, "kind": message.kind})
        return 2  # two live sockets

    def telegram_sink(message):
        received.append({"text": message.text, "kind": message.kind, "channel": "telegram"})
        return 1

    brain.register_message_sink(web_sink, channel="web", name="web-test")
    brain.register_message_sink(telegram_sink, channel="telegram", name="telegram-test")
    ctx = _make_context(brain, memory, tmp_path)

    assert await ctx.notify("Listo, instalé el pack", persist=False, channel="web") == 2
    assert received[-1] == {"text": "Listo, instalé el pack", "kind": "assistant"}

    assert await ctx.notify("Aviso para todos", persist=False) == 3
    assert len(received) == 3

    await memory.close()


async def test_notify_without_sinks_or_memory_does_not_break(tmp_path):
    brain, memory, _ = await _make_brain(tmp_path)
    ctx = _make_context(brain, memory, tmp_path)

    assert await ctx.notify("no hay nadie escuchando") == 0
    brain.memory = None
    assert await ctx.notify("tampoco hay memoria") == 0

    await memory.close()


async def test_notify_persists_with_session_as_assistant_not_user(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.notify("Instalé Navegador y SEO.", persist=True, session=session)

    assert session.history[-1] == {
        "role": "assistant",
        "content": "Instalé Navegador y SEO.",
    }
    assert not any(
        msg["role"] == "user" and "Instalé Navegador y SEO." in msg["content"]
        for msg in session.history
    )

    await brain.think("¿ya está?", session)

    messages = llm.calls[-1]["messages"]
    assert any(
        msg["role"] == "assistant" and "Instalé Navegador y SEO." in msg["content"]
        for msg in messages
    )
    assert not any(
        msg["role"] == "user" and "Instalé Navegador y SEO." in msg["content"]
        for msg in messages
    )

    await memory.close()


async def test_notify_system_kind_is_persisted_without_becoming_user(tmp_path):
    brain, memory, _ = await _make_brain(tmp_path)
    session = Session()
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.notify("Evento interno del sistema.", kind="system", session=session)

    assert session.history[-1] == {
        "role": "system",
        "content": "Evento interno del sistema.",
    }
    stored = await memory.load_conversation(brain._scoped_session_id(session))
    assert stored[-1] == {
        "role": "system",
        "content": "Evento interno del sistema.",
    }

    await memory.close()


async def test_notify_without_session_is_injected_into_next_think_once(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.notify("El pack quedó aprobado.", persist=True)

    await brain.think("hola", session)
    messages = llm.calls[-1]["messages"]
    assert any(
        msg["role"] == "assistant" and "El pack quedó aprobado." in msg["content"]
        for msg in messages
    )

    # Delivered messages live in history once; they are not duplicated on
    # every later turn.
    await brain.think("otra pregunta", session)
    messages = llm.calls[-1]["messages"]
    assert sum(
        msg["content"].count("El pack quedó aprobado.") for msg in messages
    ) == 1

    await memory.close()


async def test_system_event_reaches_next_prompt_without_recall_or_user_history(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.record_system_event(
        "El cliente aprobó el pack pack-123. Se instalaron Navegador y SEO.",
        scope="session",
        category="sistema:instalacion",
        importance=0.9,
        key="pack-123",
    )

    await brain.think("¿cómo va todo?", session)

    prompt = _captured_system_prompt(llm)
    assert "El cliente aprobó el pack pack-123" in prompt
    assert "sistema:instalacion" in prompt
    assert not any(
        "pack-123" in msg["content"] for msg in session.history
    )

    await memory.close()


async def test_system_event_can_be_updated_and_deleted_by_key(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()
    ctx = _make_context(brain, memory, tmp_path)
    key = "pack-123"

    await ctx.record_system_event(
        "Pack pack-123 pendiente de configurar.",
        scope="session",
        category="sistema:instalacion",
        importance=0.9,
        key=key,
    )
    await brain.think("estado", session)
    assert "pendiente de configurar" in _captured_system_prompt(llm)

    await ctx.record_system_event(
        "Pack pack-123 desinstalado.",
        scope="session",
        category="sistema:instalacion",
        importance=0.9,
        key=key,
    )
    await brain.think("estado", session)
    prompt = _captured_system_prompt(llm)
    assert "Pack pack-123 desinstalado." in prompt
    assert "pendiente de configurar" not in prompt

    assert await ctx.delete_system_event(key, scope="session") is True
    await brain.think("estado", session)
    assert "pack-123" not in _captured_system_prompt(llm)

    await memory.close()


async def test_system_event_global_scope_is_visible_to_other_sessions(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    admin = Session(workspace="acme", team="ops", user_email="admin@acme.com", role="admin")
    other = Session(workspace="acme", team="sales", user_email="other@acme.com", role="member")
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.record_system_event(
        "Mantenimiento programado el viernes.",
        scope="global",
        category="sistema:operaciones",
        importance=0.7,
        session=admin,
    )

    await brain.think("hola", other)
    assert "Mantenimiento programado el viernes." in _captured_system_prompt(llm)

    await memory.close()


async def test_user_scope_is_not_visible_to_a_different_user(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    alice = Session(workspace="acme", team="sales", user_email="alice@acme.com", role="member")
    bob = Session(workspace="acme", team="sales", user_email="bob@acme.com", role="member")
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.record_system_event(
        "Alice tiene un pack propio.",
        scope="user",
        category="sistema:cliente",
        importance=0.8,
        session=alice,
    )

    await brain.think("hola", bob)
    assert "Alice tiene un pack propio." not in _captured_system_prompt(llm)

    await brain.think("hola", alice)
    assert "Alice tiene un pack propio." in _captured_system_prompt(llm)

    await memory.close()


async def test_prompt_context_provider_is_a_public_extension_point(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()

    def provider(current_session):
        assert current_session is session
        return ["Línea inyectada por un provider externo."]

    unregister = brain.register_prompt_context_provider(provider, name="test-provider")

    await brain.think("hola", session)
    assert "Línea inyectada por un provider externo." in _captured_system_prompt(llm)

    unregister()
    await brain.think("hola", session)
    assert "Línea inyectada por un provider externo." not in _captured_system_prompt(llm)

    await memory.close()


async def test_behavior_is_unchanged_when_hooks_are_not_used(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    session = Session()

    await brain.think("hola", session)
    prompt = _captured_system_prompt(llm)
    assert "## System events (durable, current state)" not in prompt
    assert "## Context from extensions" not in prompt

    await memory.close()


async def test_notify_does_not_replace_capability_awareness(tmp_path):
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    brain, memory, llm = await _make_brain(tmp_path)
    brain.registry = registry
    brain.capability_awareness = awareness
    ctx = _make_context(brain, memory, tmp_path)

    await ctx.notify("Aviso proactivo", persist=False)
    registry.register(
        Capability(
            kind=CapabilityKind.SKILL,
            name="nueva-skill",
            description="Skill nueva",
            status=CapabilityStatus.READY,
        )
    )

    assert awareness.has_pending()
    await brain.think("hola", Session())
    assert "nueva-skill" in _captured_system_prompt(llm)

    await memory.close()


async def test_think_proactive_accepts_authoritative_text_and_delivers(tmp_path):
    brain, memory, llm = await _make_brain(tmp_path)
    received: list[str] = []
    brain.register_message_sink(
        lambda message: received.append(message.text) or 1,
        channel="web",
    )

    text = await brain.think_proactive(
        announce_text="Listo, el pack pack-123 quedó instalado.",
        deliver=True,
        channel="web",
    )

    assert text == "Listo, el pack pack-123 quedó instalado."
    assert received == [text]
    assert llm.calls == []  # no need to ask the model for known facts

    await memory.close()


async def test_notify_and_system_event_survive_without_brain(tmp_path):
    ctx = ModuleRuntimeContext(
        name="test-kit",
        module_dir=tmp_path / "module",
        runtime_dir=tmp_path / "runtime",
        manifest={},
        config={},
        brain=None,
        memory=None,
    )

    assert await ctx.notify("sin cerebro", persist=True) == 0
    assert await ctx.record_system_event("sin memoria", scope="session") is None
    assert await ctx.delete_system_event("nada") is False
