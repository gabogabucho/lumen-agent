"""Tests for Inbox with channel status tracking."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from lumen.core.agent_status import ChannelStatus
from lumen.core.inbox import Inbox, IncomingMessage


class TestInboxBasic(unittest.TestCase):
    def test_push_and_stop(self):
        inbox = Inbox()

        async def run():
            await inbox.push(IncomingMessage("test", "user1", "hello"))
            inbox.stop()

        asyncio.run(run())
        assert inbox.get_registered_adapters() == []

    def test_register_adapter(self):
        inbox = Inbox()
        adapter = AsyncMock()
        inbox.register_adapter("test", adapter)
        assert "test" in inbox.get_registered_adapters()
        assert "test" in inbox._channel_status

    def test_unregister_adapter(self):
        inbox = Inbox()
        adapter = AsyncMock()
        inbox.register_adapter("test", adapter)
        inbox.unregister_adapter("test")
        assert "test" not in inbox.get_registered_adapters()
        assert inbox._channel_status["test"].status == "disconnected"

    def test_unregister_nonexistent(self):
        inbox = Inbox()
        inbox.unregister_adapter("nonexistent")  # Should not raise


class TestChannelStatus(unittest.TestCase):
    def test_default_status(self):
        inbox = Inbox()
        inbox.register_adapter("telegram", AsyncMock())
        status = inbox.get_channel_status()
        assert len(status) == 1
        assert status[0]["name"] == "telegram"
        assert status[0]["status"] == "disconnected"

    def test_set_channel_status(self):
        inbox = Inbox()
        inbox.register_adapter("telegram", AsyncMock())
        inbox.set_channel_status("telegram", "connected")
        status = inbox.get_channel_status()
        assert status[0]["status"] == "connected"
        assert status[0]["last_activity"] is not None

    def test_set_channel_status_with_error(self):
        inbox = Inbox()
        inbox.register_adapter("telegram", AsyncMock())
        inbox.set_channel_status("telegram", "error", error="Connection refused")
        status = inbox.get_channel_status()
        assert status[0]["status"] == "error"
        assert status[0]["error"] == "Connection refused"

    def test_mark_internal(self):
        inbox = Inbox()
        inbox.register_adapter("web", AsyncMock())
        inbox.mark_internal_channel("web")
        status = inbox.get_channel_status()
        assert status[0]["is_internal"] is True
        assert status[0]["status"] == "connected"

    def test_mark_internal_nonexistent(self):
        inbox = Inbox()
        inbox.mark_internal_channel("nonexistent")
        status = inbox.get_channel_status()
        assert status[0]["is_internal"] is True


class TestProcessMessage(unittest.TestCase):
    def test_process_with_no_adapter(self):
        inbox = Inbox()
        brain = AsyncMock()
        brain.think = AsyncMock(return_value={"message": "hi"})
        session_mgr = MagicMock()
        session_mgr.get_or_create = MagicMock(return_value=MagicMock(history=[]))

        async def run():
            await inbox._process(brain, session_mgr, IncomingMessage("test", "u1", "hello"))

        asyncio.run(run())
        brain.think.assert_called_once_with("hello", session_mgr.get_or_create.return_value)

    def test_process_tracks_message_count(self):
        inbox = Inbox()
        adapter = AsyncMock()
        inbox.register_adapter("telegram", adapter)

        brain = AsyncMock()
        brain.think = AsyncMock(return_value={"message": "response"})

        session_mgr = MagicMock()
        session_mgr.get_or_create = MagicMock(return_value=MagicMock(history=[]))

        async def run():
            await inbox._process(brain, session_mgr, IncomingMessage("telegram", "u1", "hi"))

        asyncio.run(run())
        status = inbox.get_channel_status()
        assert status[0]["message_count"] == 1


class TestChannelStatusDataclass(unittest.TestCase):
    def test_defaults(self):
        cs = ChannelStatus(name="test")
        assert cs.name == "test"
        assert cs.status == "disconnected"
        assert cs.message_count == 0
        assert cs.last_activity is None
        assert cs.error is None
        assert cs.is_internal is False


if __name__ == "__main__":
    unittest.main()
