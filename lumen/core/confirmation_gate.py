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

    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._handler: ConfirmHandler | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._history: list[dict[str, Any]] = []

    def set_handler(self, handler: ConfirmHandler) -> None:
        """Register the channel-specific confirmation handler."""
        self._handler = handler

    def clear_handler(self) -> None:
        """Remove the handler (e.g. on channel disconnect)."""
        self._handler = None

    @property
    def has_handler(self) -> bool:
        return self._handler is not None

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
        if not self._handler:
            logger.debug("No confirmation handler — auto-approving %s", call_id)
            self._record(request, ConfirmDecision.AUTO_APPROVED)
            return ConfirmResponse(
                call_id=call_id,
                decision=ConfirmDecision.AUTO_APPROVED,
                message="Auto-approved (no handler)",
            )

        # Create a future that the handler or resolve() will set
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ConfirmResponse] = loop.create_future()
        self._pending[call_id] = future

        try:
            # Call the handler (non-blocking — it should yield the request to the client)
            handler_task = asyncio.create_task(self._handler(request))

            # Wait for response with timeout
            try:
                result = await asyncio.wait_for(future, timeout=self.timeout)
                return result
            except asyncio.TimeoutError:
                logger.warning("Confirmation timeout for %s (%ds)", call_id, self.timeout)
                self._record(request, ConfirmDecision.TIMEOUT)
                # Cancel the handler task if still running
                handler_task.cancel()
                return ConfirmResponse(
                    call_id=call_id,
                    decision=ConfirmDecision.TIMEOUT,
                    message=f"Timed out after {self.timeout}s",
                )
        finally:
            self._pending.pop(call_id, None)

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
