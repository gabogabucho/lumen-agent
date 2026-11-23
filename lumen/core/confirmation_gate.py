"""Confirmation gate — pause tool execution pending user approval.

When a tool requires confirmation (per ToolPolicy), the brain yields a
`tool_confirm_request` event and blocks until the user approves or rejects,
or the timeout expires.

The gate is channel-agnostic: each channel (web, cli) registers a callback
that handles the actual UI interaction.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class ConfirmDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ConfirmRequest:
    """A pending confirmation request."""
    call_id: str
    tool_name: str
    action: str
    risk: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "action": self.action,
            "risk": self.risk,
            "description": self.description,
            "params": self.params,
            "created_at": self.created_at,
        }


@dataclass
class ConfirmResponse:
    """User's decision on a confirmation request."""
    call_id: str
    decision: ConfirmDecision
    message: str = ""


# Type alias for the async callback
ConfirmHandler = Callable[[ConfirmRequest], Coroutine[Any, Any, ConfirmResponse]]


class ConfirmationGate:
    """Manages tool confirmation flow.

    Usage:
        gate = ConfirmationGate(timeout=60)
        gate.set_handler(my_async_handler)

        # In the tool execution loop:
        if gate.requires_confirmation(tool_name, action, risk):
            response = await gate.ask(tool_name, action, risk, params)
            if response.decision != ConfirmDecision.APPROVED:
                # Skip execution
                ...

    The handler is called with the request and must return a ConfirmResponse.
    If no handler is set, the gate auto-approves (backward compat).
    """

    # Name reserved for the channel handler installed via ``set_handler``.
    # ``register_handler`` refuses to reuse it so the legacy slot stays unique.
    _CHANNEL_HANDLER_NAME = "__channel__"

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._handlers: list[dict[str, Any]] = []
        self._handler_seq = 0
        self._pending: dict[str, asyncio.Future] = {}
        self._history: list[dict[str, Any]] = []

    def set_handler(self, handler: ConfirmHandler) -> None:
        """Register the channel-specific confirmation handler.

        Kept as-is for backward compatibility (``web.py`` uses it): it replaces
        the channel slot of the ordered chain. Handlers added with
        ``register_handler`` stay in place, so a module can defer and let the
        channel decide. With no registered handlers this is the only handler,
        exactly like before.
        """
        self._handlers = [
            item
            for item in self._handlers
            if item["name"] != self._CHANNEL_HANDLER_NAME
        ]
        self._add_handler(handler, self._CHANNEL_HANDLER_NAME, 100)

    def register_handler(
        self,
        handler: ConfirmHandler,
        *,
        name: str,
        order: int = 100,
    ) -> Callable[[], None]:
        """Add a handler to the ordered chain. Returns an unregister token.

        Handlers run by ascending ``order`` (ties keep registration order). The
        first response that is not a deferral ends the chain; a ``REJECTED``
        whose message starts with ``deferred:`` passes to the next handler.
        """
        if name == self._CHANNEL_HANDLER_NAME:
            raise ValueError(f"handler name {name!r} is reserved")
        item = self._add_handler(handler, name, order)

        def _unregister() -> None:
            if item in self._handlers:
                self._handlers.remove(item)

        return _unregister

    def _add_handler(
        self, handler: ConfirmHandler, name: str, order: int
    ) -> dict[str, Any]:
        self._handler_seq += 1
        item = {
            "handler": handler,
            "name": name,
            "order": order,
            "seq": self._handler_seq,
        }
        self._handlers.append(item)
        return item

    def clear_handler(self) -> None:
        """Remove the channel handler (e.g. on channel disconnect).

        Module handlers registered with ``register_handler`` are left intact.
        """
        self._handlers = [
            item
            for item in self._handlers
            if item["name"] != self._CHANNEL_HANDLER_NAME
        ]

    @property
    def has_handler(self) -> bool:
        return bool(self._handlers)

    def _ordered_handlers(self) -> list[dict[str, Any]]:
        return sorted(self._handlers, key=lambda item: (item["order"], item["seq"]))

    async def ask(
        self,
        tool_name: str,
        action: str,
        risk: str,
        params: dict[str, Any] | None = None,
        description: str = "",
    ) -> ConfirmResponse:
        """Ask for confirmation. Blocks until response or timeout.

        Returns APPROVED if handler approves, REJECTED if handler rejects,
        TIMEOUT if no response within timeout, or AUTO_APPROVED if no handler
        is registered (backward compatibility).
        """
        call_id = uuid.uuid4().hex[:12]
        request = ConfirmRequest(
            call_id=call_id,
            tool_name=tool_name,
            action=action,
            risk=risk,
            description=description or f"{tool_name}__{action} ({risk})",
            params=params or {},
        )

        # No handler — auto-approve (backward compat for CLI/non-interactive)
        handlers = self._ordered_handlers()
        if not handlers:
            logger.debug("No confirmation handler — auto-approving %s", call_id)
            self._record(request, ConfirmDecision.AUTO_APPROVED)
            return ConfirmResponse(
                call_id=call_id,
                decision=ConfirmDecision.AUTO_APPROVED,
                message="Auto-approved (no handler)",
            )

        # Create a future that a handler or resolve() can set
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ConfirmResponse] = loop.create_future()
        self._pending[call_id] = future

        last_rejected: ConfirmResponse | None = None
        try:
            for item in handlers:
                try:
                    result = await self._invoke_handler(item["handler"], request, future)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Confirmation timeout for %s (%ds)", call_id, self.timeout
                    )
                    self._record(request, ConfirmDecision.TIMEOUT)
                    return ConfirmResponse(
                        call_id=call_id,
                        decision=ConfirmDecision.TIMEOUT,
                        message=f"Timed out after {self.timeout}s",
                    )
                except Exception:
                    # Pattern of invoke_inbound_hooks: failures pass through.
                    logger.exception(
                        "confirmation handler %r failed; passing through", item["name"]
                    )
                    continue

                if result is None:
                    continue

                if (
                    result.decision == ConfirmDecision.REJECTED
                    and str(result.message or "").startswith("deferred:")
                ):
                    last_rejected = result
                    continue

                self._record(request, result.decision)
                return result

            if last_rejected is not None:
                self._record(request, last_rejected.decision)
                return last_rejected

            # Handlers existed but none answered: never auto-approve silently.
            self._record(request, ConfirmDecision.TIMEOUT)
            return ConfirmResponse(
                call_id=call_id,
                decision=ConfirmDecision.TIMEOUT,
                message="No confirmation handler responded",
            )
        finally:
            self._pending.pop(call_id, None)

    async def _invoke_handler(
        self,
        handler: ConfirmHandler,
        request: ConfirmRequest,
        future: asyncio.Future,
    ) -> ConfirmResponse | None:
        """Run one handler, racing its return value with ``resolve()``."""
        task = asyncio.create_task(handler(request))
        try:
            done, _ = await asyncio.wait(
                {task, future},
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if task in done:
                result = task.result()
                if result is not None:
                    return result
            if future in done:
                return future.result()
            if task in done:
                return None
            raise asyncio.TimeoutError
        finally:
            if not task.done():
                task.cancel()

    def resolve(self, call_id: str, decision: ConfirmDecision, message: str = "") -> bool:
        """Resolve a pending confirmation (called by the channel when user responds).

        Returns True if the call_id was found and resolved, False otherwise.
        This is a sync method — it just sets the future result.
        """
        future = self._pending.get(call_id)
        if not future or future.done():
            return False

        request_info = {"call_id": call_id, "decision": decision.value}
        future.set_result(ConfirmResponse(call_id=call_id, decision=decision, message=message))
        logger.info("Confirmation resolved: %s", request_info)
        return True

    def get_pending_count(self) -> int:
        """Number of currently pending confirmations."""
        return len(self._pending)

    def get_pending(self) -> list[dict[str, Any]]:
        """List of pending call IDs."""
        return list(self._pending.keys())

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Recent confirmation decisions."""
        return list(reversed(self._history[-limit:]))

    def _record(self, request: ConfirmRequest, decision: ConfirmDecision) -> None:
        """Record a confirmation decision for audit."""
        self._history.append({
            "call_id": request.call_id,
            "tool_name": request.tool_name,
            "action": request.action,
            "risk": request.risk,
            "decision": decision.value,
            "timestamp": time.time(),
        })
        # Keep history bounded
        if len(self._history) > 200:
            self._history = self._history[-200:]
