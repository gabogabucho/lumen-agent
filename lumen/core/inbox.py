"""Unified inbox — one queue, one brain, any channel.

All external channels (Telegram, WhatsApp, email, etc.) push incoming
messages here. A single consumer routes them through the same
``brain.think()`` call, maintaining one Lumen identity across channels.

The web channel bypasses the inbox (it calls ``brain.think()`` directly
via WebSocket) but registers itself as an adapter so cross-channel
responses can reach the web UI too.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Protocol, runtime_checkable

from lumen.core.agent_status import ChannelStatus

logger = logging.getLogger(__name__)


@dataclass
class IncomingMessage:
    """A message arriving from an external channel."""

    channel: str  # "telegram", "whatsapp", "email", ...
    sender_id: str  # chat_id, phone, email — identifies the user
    text: str


@runtime_checkable
class ChannelAdapter(Protocol):
    """Protocol for channel adapters that can send messages."""

    async def send(self, recipient_id: str, message: str) -> None: ...


# Type alias for the send callback
SendCallback = Callable[[str, str], Coroutine[Any, Any, None]]


class Inbox:
    """Unified message queue that feeds all channels through one brain.

    Usage::

        inbox = Inbox()
        inbox.register_adapter("telegram", telegram_adapter)
        inbox.register_adapter("whatsapp", whatsapp_adapter)

        # In web channel startup:
        brain.inbox = inbox
        asyncio.create_task(inbox.start_consumer(brain, session_mgr))

        # In telegram module:
        brain.inbox.push(IncomingMessage("telegram", str(chat_id), text))
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[IncomingMessage | None] = asyncio.Queue()
        self._adapters: dict[str, ChannelAdapter | SendCallback] = {}
        self._channel_status: dict[str, ChannelStatus] = {}
        self._default_session_id: str = "inbox-unified"

    def register_adapter(
        self, channel_id: str, adapter: ChannelAdapter | SendCallback
    ) -> None:
        """Register a channel adapter for response routing."""
        self._adapters[channel_id] = adapter
        # Initialize status if not present
        if channel_id not in self._channel_status:
            self._channel_status[channel_id] = ChannelStatus(name=channel_id)

    def unregister_adapter(self, channel_id: str) -> None:
        """Remove a channel adapter."""
        self._adapters.pop(channel_id, None)
        if channel_id in self._channel_status:
            self._channel_status[channel_id].status = "disconnected"

    async def push(self, message: IncomingMessage) -> None:
        """Push an incoming message to the queue."""
        await self._queue.put(message)

    @property
    def default_session_id(self) -> str:
        return self._default_session_id

    @default_session_id.setter
    def default_session_id(self, value: str) -> None:
        self._default_session_id = value

    async def start_consumer(
        self,
        brain: Any,
        session_manager: Any,
    ) -> None:
        """Background consumer: dequeue messages → brain.think() → respond.

        Runs forever until ``stop()`` is called.
        """
        while True:
            item = await self._queue.get()
            if item is None:
                break

            try:
                await self._process(brain, session_manager, item)
            except Exception:
                logger.exception(
                    "inbox: error processing message from %s/%s",
                    item.channel,
                    item.sender_id,
                )

    def stop(self) -> None:
        """Signal the consumer to stop after processing the current message."""
        asyncio.get_event_loop().call_soon_threadsafe(
            self._queue.put_nowait, None
        )

    def set_channel_status(
        self, channel_id: str, status: str, error: str | None = None
    ) -> None:
        """Update the status of a registered channel."""
        if channel_id in self._channel_status:
            self._channel_status[channel_id].status = status
            self._channel_status[channel_id].error = error
            if status == "connected":
                self._channel_status[channel_id].last_activity = (
                    datetime.now(timezone.utc).isoformat()
                )

    def mark_internal_channel(self, channel_id: str) -> None:
        """Mark a channel as internal (e.g., web — doesn't use inbox queue)."""
        if channel_id in self._channel_status:
            self._channel_status[channel_id].is_internal = True
            self._channel_status[channel_id].status = "connected"
        else:
            self._channel_status[channel_id] = ChannelStatus(
                name=channel_id, status="connected", is_internal=True
            )

    def get_channel_status(self) -> list[dict[str, Any]]:
        """Return status of all registered channels."""
        result = []
        for name, status in self._channel_status.items():
            result.append({
                "name": name,
                "status": status.status,
                "message_count": status.message_count,
                "last_activity": status.last_activity,
                "error": status.error,
                "is_internal": status.is_internal,
            })
        return result

    def get_registered_adapters(self) -> list[str]:
        """Return list of registered channel adapter names."""
        return list(self._adapters.keys())

    async def _process(
        self,
        brain: Any,
        session_manager: Any,
        msg: IncomingMessage,
    ) -> None:
        session = session_manager.get_or_create(self._default_session_id)

        result = await brain.think(msg.text, session)
        response_text = str(result.get("message") or "").strip()
        if not response_text:
            return

        adapter = self._adapters.get(msg.channel)
        if adapter is None:
            logger.warning("inbox: no adapter registered for channel %s", msg.channel)
            return

        try:
            if callable(adapter):
                await adapter(msg.sender_id, response_text)
            else:
                await adapter.send(msg.sender_id, response_text)
        except Exception:
            logger.exception(
                "inbox: error sending response to %s/%s",
                msg.channel,
                msg.sender_id,
            )
            return

        # Track message count and last activity
        if msg.channel in self._channel_status:
            self._channel_status[msg.channel].message_count += 1
            self._channel_status[msg.channel].last_activity = (
                datetime.now(timezone.utc).isoformat()
            )
