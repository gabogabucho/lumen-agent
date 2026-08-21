"""Tool handlers that accept `session` receive it; others stay unchanged."""

import asyncio
import unittest

from lumen.core.connectors import ConnectorRegistry
from lumen.core.session import Session


class ToolSessionPassthroughTests(unittest.TestCase):
    def test_handler_with_session_param_gets_the_session(self):
        seen = {}

        async def handler(value: str = "", session=None, **_):
            seen["value"] = value
            seen["ticket"] = (session.metadata or {}).get("ticket") if session else None
            return {"ok": True}

        registry = ConnectorRegistry()
        registry.register_tool(
            "host__act",
            "test",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            handler,
        )
        session = Session(session_id="t1")
        session.metadata = {"ticket": "secreto"}
        result = asyncio.run(registry.execute_tool(
            "host__act", {"value": "x"}, session=session
        ))
        assert result == {"ok": True}
        assert seen["value"] == "x"
        assert seen["ticket"] == "secreto"

    def test_handler_without_session_param_does_not_break(self):
        async def handler(value: str = "", **_):
            return value

        registry = ConnectorRegistry()
        registry.register_tool(
            "host__plain",
            "test",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            handler,
        )
        session = Session(session_id="t1")
        result = asyncio.run(registry.execute_tool(
            "host__plain", {"value": "y"}, session=session
        ))
        assert result == "y"
