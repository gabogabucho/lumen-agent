"""Tests for the post-execution tool audit hook (upstream plug E3).

Covers the scenarios of spec ``enchufes-upstream`` requirement E3: every
registered auditor sees success and failure, internal shortcuts included, and
a broken hook does not interrupt the turn. No LLM or external API is called.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import yaml

from lumen.core.brain import Brain
from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.memory import Memory
from lumen.core.registry import Registry
from lumen.core.session import Session


def _make_brain(tmp_path, *, with_tools: bool = True) -> Brain:
    consciousness = tmp_path / "consciousness.yaml"
    consciousness.write_text(
        yaml.safe_dump({"identity": {"name": "TestLumen", "type": "test"}}),
        encoding="utf-8",
    )
    personality = tmp_path / "personality.yaml"
    personality.write_text(
        yaml.safe_dump({"identity": {"name": "TestLumen", "role": "Test"}}),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "index.yaml"
    catalog_path.write_text(yaml.safe_dump({"modules": []}), encoding="utf-8")

    from lumen.core.consciousness import Consciousness
    from lumen.core.personality import Personality

    connectors = ConnectorRegistry()
    if with_tools:
        async def ok_tool(value: str):
            return {"ok": value}

        async def boom_tool():
            raise RuntimeError("boom")

        connectors.register_tool("audit__ok", "Ok", {}, ok_tool)
        connectors.register_tool("audit__boom", "Boom", {}, boom_tool)

    return Brain(
        consciousness=Consciousness(config_path=consciousness),
        personality=Personality(path=personality),
        memory=MagicMock(spec=Memory),
        connectors=connectors,
        registry=Registry(),
        catalog=Catalog(catalog_path),
        model="test-model",
    )


def _tool_call(name: str, arguments: str = "{}"):
    call = MagicMock()
    call.id = f"call-{name}"
    call.function.name = name
    call.function.arguments = arguments
    return call


def _response(tool_calls=None, content: str = "done"):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump.return_value = {"role": "assistant", "content": content}
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _collector(events: list) -> object:
    async def collect(event):
        events.append(event)

    return collect


def _prepare(brain: Brain):
    brain._acomplete = AsyncMock(return_value=_response())


class TestToolAuditHooks:
    async def test_two_hooks_see_success_and_error(self, tmp_path):
        """Spec scenario: each hook gets one event per execution."""
        brain = _make_brain(tmp_path)
        _prepare(brain)
        first: list = []
        second: list = []
        brain.register_tool_audit_hook(_collector(first), name="first")
        brain.register_tool_audit_hook(_collector(second), name="second")
        session = Session()

        result = await brain._tool_use_loop(
            _response([
                _tool_call("audit__ok", '{"value": "hola"}'),
                _tool_call("audit__boom"),
            ]),
            [],
            None,
            session=session,
            max_iterations=1,
        )

        assert result["message"] == "done"
        for events in (first, second):
            assert len(events) == 2
            success, failure = events
            assert success.tool_name == "audit__ok"
            assert success.action == "ok"
            assert success.params == {"value": "hola"}
            assert success.result == {"ok": "hola"}
            assert success.error == ""
            assert success.duration_ms > 0
            assert success.session_id == session.session_id
            assert failure.tool_name == "audit__boom"
            assert failure.error == "boom"
            assert failure.duration_ms > 0

    async def test_internal_shortcuts_are_audited(self, tmp_path):
        brain = _make_brain(tmp_path, with_tools=False)
        _prepare(brain)
        events: list = []
        brain.register_tool_audit_hook(_collector(events), name="auditor")

        await brain._tool_use_loop(
            _response([
                _tool_call("neo__read_skill"),
                _tool_call("render_surface", '{"surface": "missing"}'),
            ]),
            [],
            None,
            session=Session(),
            max_iterations=1,
        )

        assert [event.tool_name for event in events] == [
            "neo__read_skill",
            "render_surface",
        ]
        assert all(event.error for event in events)

    async def test_broken_hook_does_not_stop_turn(self, tmp_path, caplog):
        """Spec scenario: other hooks run and the answer still goes out."""
        brain = _make_brain(tmp_path)
        _prepare(brain)

        async def broken(event):
            raise RuntimeError("hook broken")

        events: list = []
        brain.register_tool_audit_hook(broken, name="broken")
        brain.register_tool_audit_hook(_collector(events), name="good")

        with caplog.at_level(logging.ERROR, logger="lumen.core.brain"):
            result = await brain._tool_use_loop(
                _response([_tool_call("audit__ok", '{"value": "hola"}')]),
                [],
                None,
                session=Session(),
                max_iterations=1,
            )

        assert result["message"] == "done"
        assert [event.tool_name for event in events] == ["audit__ok"]
        assert any("broken" in record.message for record in caplog.records)

    async def test_streaming_loop_audits_executions(self, tmp_path):
        brain = _make_brain(tmp_path)
        _prepare(brain)
        events: list = []
        brain.register_tool_audit_hook(_collector(events), name="auditor")

        chunks = [
            chunk
            async for chunk in brain._tool_use_loop_streaming(
                _response([_tool_call("audit__ok", '{"value": "hola"}')]),
                [],
                None,
                session=Session(session_id="stream-session"),
                max_iterations=1,
            )
        ]

        assert chunks[-1]["_result"]["message"] == "done"
        assert len(events) == 1
        assert events[0].tool_name == "audit__ok"
        assert events[0].session_id == "stream-session"
