"""Tests for the generic `continue` hook on runtime flows.

A flow may declare `continue: true` (plus an optional `continue_message`
template) so that, once it completes, the SAME turn goes on to run the normal
Brain (`think`/`think_stream`) with the collected slots as context instead of
ending on the flow's `first_message`/`on_complete` message.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from lumen.core.brain import Brain
from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.consciousness import Consciousness
from lumen.core.llm_client import LLMChunk, LLMResponse
from lumen.core.memory import Memory
from lumen.core.personality import Personality
from lumen.core.registry import Registry
from lumen.core.session import Session


def _make_consciousness():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(
        {
            "identity": {"name": "TestLumen", "type": "test agent"},
            "nature": ["I am modular"],
        },
        tmp,
    )
    tmp.flush()
    return Consciousness(config_path=Path(tmp.name))


def _make_personality():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(
        {
            "identity": {"name": "TestLumen", "role": "Test Assistant"},
            "tone": {"style": "direct"},
            "rules": ["Be helpful"],
        },
        tmp,
    )
    tmp.flush()
    return Personality(path=Path(tmp.name))


def _make_catalog():
    with tempfile.TemporaryDirectory() as tmp:
        catalog_path = Path(tmp) / "index.yaml"
        catalog_path.write_text(yaml.dump({"modules": []}), encoding="utf-8")
        return Catalog(catalog_path)


class FakeLLM:
    def __init__(self, content="Te propongo el pack mínimo."):
        self.content = content
        self.calls = []

    @property
    def call_count(self):
        return len(self.calls)

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        return LLMResponse(content=self.content)

    async def stream(self, **kwargs):
        self.calls.append(kwargs)
        yield LLMChunk(delta_content=self.content, finish_reason="stop")


def _make_brain(*, flows, handler=None, llm_content="Te propongo el pack mínimo."):
    brain = Brain(
        consciousness=_make_consciousness(),
        personality=_make_personality(),
        memory=MagicMock(spec=Memory),
        connectors=ConnectorRegistry(),
        registry=Registry(),
        catalog=_make_catalog(),
        model="test-model",
        flows=flows,
        flow_action_handler=handler,
    )
    brain.memory.recall = AsyncMock(return_value=[])
    brain.memory.list_session_summaries = AsyncMock(return_value=[])
    brain.memory.list_session_facts = AsyncMock(return_value=[])
    brain.memory.save_conversation_turn = AsyncMock()
    fake = FakeLLM(llm_content)
    brain._llm_client = fake
    brain._distiller._llm_client = fake
    return brain, fake


def _flow(**overrides):
    flow = {
        "intent": "module-setup-demo",
        "triggers": ["setup:demo"],
        "slots": {
            "que_hace": {"required": True, "ask": "¿Qué hace?"},
            "para_que": {"required": True, "ask": "¿Para qué?"},
        },
        "on_complete": "save_module_env:demo",
        "first_message": "Necesito unos datos.",
    }
    flow.update(overrides)
    return flow


async def _reach_completion(brain, session, *, second="hago pan", third="para vender"):
    """Walk a two-slot flow up to the turn that completes it."""
    await brain.think("setup:demo", session)
    await brain.think(second, session)
    return await brain.think(third, session)


def _handler(message="Listo, demo configurado."):
    return AsyncMock(return_value={"status": "ok", "message": message})


def _last_prompt(fake) -> str:
    messages = fake.calls[-1]["messages"]
    return "\n".join(str(m.get("content", "")) for m in messages)


@pytest.mark.asyncio
async def test_flow_continue_corre_el_seguimiento_en_el_mismo_turno():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    third = await _reach_completion(brain, session)

    assert third["message"] == "Listo, demo configurado.\n\nTe propongo el pack mínimo."
    assert third["continued"] is True
    assert fake.call_count == 1
    # The visible turn is the real user message followed by ONE assistant
    # message that closes the flow and delivers the continuation.
    assert session.history[-2:] == [
        {"role": "user", "content": "para vender"},
        {
            "role": "assistant",
            "content": "Listo, demo configurado.\n\nTe propongo el pack mínimo.",
        },
    ]
    # Three turns, two messages each, no synthetic user turn.
    assert len(session.history) == 6
    # The internal continuation instruction never becomes conversation.
    assert all(
        "Seguí con hago pan" not in msg["content"] for msg in session.history
    )


@pytest.mark.asyncio
async def test_flow_continue_usa_los_slots_en_el_prompt():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    await _reach_completion(brain, session)

    # The LLM receives the internal directive as prompt context...
    assert "Seguí con hago pan" in _last_prompt(fake)
    # ...but never as a user turn.
    user_contents = [
        m["content"]
        for m in fake.calls[-1]["messages"]
        if m["role"] == "user"
    ]
    assert "Seguí con hago pan" not in user_contents


@pytest.mark.asyncio
async def test_flow_continue_sin_template_usa_los_slots():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True})],
        handler=_handler(),
    )
    session = Session()

    await _reach_completion(brain, session)

    prompt = _last_prompt(fake)
    assert '"que_hace": "hago pan"' in prompt
    assert '"para_que": "para vender"' in prompt
    # Slots survive `complete_flow()` long enough to reach the follow-up.
    assert session.slots == {}


@pytest.mark.asyncio
async def test_flow_continue_stream_emite_un_solo_final():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    await brain.think("setup:demo", session)
    await brain.think("hago pan", session)

    events = [event async for event in brain.think_stream("para vender", session)]

    finals = [event for event in events if event["type"] == "final"]
    deltas = "".join(
        event["content"] for event in events if event["type"] == "delta"
    )
    assert len(finals) == 1
    assert finals[0]["content"] == (
        "Listo, demo configurado.\n\nTe propongo el pack mínimo."
    )
    # The flow text and the follow-up are each present exactly once.
    assert deltas == "Listo, demo configurado.Te propongo el pack mínimo."
    assert fake.call_count == 1
    # The internal continuation instruction is neither streamed nor stored.
    streamed = "".join(
        str(event.get("content", "")) for event in events
    )
    assert "Seguí con hago pan" not in streamed
    assert session.history[-2:] == [
        {"role": "user", "content": "para vender"},
        {
            "role": "assistant",
            "content": "Listo, demo configurado.\n\nTe propongo el pack mínimo.",
        },
    ]
    assert all(
        "Seguí con hago pan" not in msg["content"] for msg in session.history
    )
    assert "Seguí con hago pan" in _last_prompt(fake)


@pytest.mark.asyncio
async def test_flow_sin_continue_no_cambia():
    brain, fake = _make_brain(
        flows=[_flow()],
        handler=_handler(),
    )
    session = Session()

    third = await _reach_completion(brain, session)

    assert third["message"] == "Listo, demo configurado."
    assert "continued" not in third
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_flow_continue_false_se_ignora():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": False, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    third = await _reach_completion(brain, session)

    assert third["message"] == "Listo, demo configurado."
    assert "continued" not in third
    assert fake.call_count == 0


@pytest.mark.asyncio
async def test_flow_continue_finaliza_cada_parte_una_vez():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    await _reach_completion(brain, session)

    saved = [call.args for call in brain.memory.save_conversation_turn.await_args_list]
    combined = "Listo, demo configurado.\n\nTe propongo el pack mínimo."
    assert saved.count((session.session_id, "user", "para vender")) == 1
    assert saved.count((session.session_id, "assistant", combined)) == 1
    # The continuation instruction is internal: never persisted as a turn.
    assert all("Seguí con hago pan" not in args[2] for args in saved)
    # Three turns, two messages each.
    assert len(saved) == 6


@pytest.mark.asyncio
async def test_continue_message_no_ensucia_memoria_ni_historial():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    await _reach_completion(brain, session)

    instruction = "Seguí con hago pan"
    assert instruction not in [m["content"] for m in session.history]
    saved = [call.args[2] for call in brain.memory.save_conversation_turn.await_args_list]
    assert instruction not in saved
    # Yet the model did receive it, as internal prompt context.
    assert instruction in _last_prompt(fake)


@pytest.mark.asyncio
async def test_continue_message_no_emite_eventos_en_stream():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=_handler(),
    )
    session = Session()

    await brain.think("setup:demo", session)
    await brain.think("hago pan", session)
    events = [event async for event in brain.think_stream("para vender", session)]

    instruction = "Seguí con hago pan"
    assert all(
        instruction not in str(event.get("content", "")) for event in events
    )


@pytest.mark.asyncio
async def test_flow_continue_sin_handler():
    brain, fake = _make_brain(
        flows=[_flow(**{"continue": True, "continue_message": "Seguí con $que_hace"})],
        handler=None,
    )
    session = Session()

    third = await _reach_completion(brain, session)

    assert third["message"] == "Listo.\n\nTe propongo el pack mínimo."
    assert third["continued"] is True
    assert fake.call_count == 1
    assert "Seguí con hago pan" in _last_prompt(fake)
