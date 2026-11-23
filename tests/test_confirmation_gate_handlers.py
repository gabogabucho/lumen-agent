"""Tests for the ordered confirmation-handler chain (upstream plug E1).

Covers every scenario of the spec ``enchufes-upstream`` (requirement E1):

- two handlers coexist: the channel via ``set_handler`` and a module via
  ``register_handler(order=...)``;
- a handler that raises is logged and the chain passes to the next one;
- a ``REJECTED`` with ``message`` starting with ``deferred:`` hands off;
- if every handler defers, the last ``REJECTED`` is returned;
- ``set_handler`` keeps its current behaviour (a single channel handler) and
  is not stacked by repeated calls.
"""

import logging

import pytest

from lumen.core.confirmation_gate import (
    ConfirmationGate,
    ConfirmDecision,
    ConfirmResponse,
)


def _approved(message=""):
    async def handler(request):
        return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED, message)

    return handler


def _rejected(message=""):
    async def handler(request):
        return ConfirmResponse(request.call_id, ConfirmDecision.REJECTED, message)

    return handler


class TestHandlerChain:
    async def test_two_handlers_coexist(self):
        """Spec scenario: the channel and a module handler live together."""
        gate = ConfirmationGate(timeout=5)
        seen = []

        async def module_handler(request):
            seen.append("module")
            if request.tool_name == "defer_me":
                return ConfirmResponse(
                    request.call_id,
                    ConfirmDecision.REJECTED,
                    "deferred:apr_123",
                )
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        async def channel_handler(request):
            seen.append("channel")
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        # The module registers first; the channel then attaches with set_handler.
        gate.register_handler(module_handler, name="module", order=50)
        gate.set_handler(channel_handler)

        deferred = await gate.ask("defer_me", "run", "mutating")
        assert deferred.decision == ConfirmDecision.APPROVED
        assert seen == ["module", "channel"]

        seen.clear()
        approved = await gate.ask("module_approves", "run", "mutating")
        assert approved.decision == ConfirmDecision.APPROVED
        assert seen == ["module"], "the channel must not be consulted"

    async def test_failing_handler_passes_to_next(self, caplog):
        """Spec scenario: failures pass through to the next handler."""
        gate = ConfirmationGate(timeout=5)

        async def broken(request):
            raise RuntimeError("boom")

        async def good(request):
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        gate.register_handler(broken, name="broken", order=10)
        gate.register_handler(good, name="good", order=20)

        with caplog.at_level(logging.ERROR, logger="lumen.core.confirmation_gate"):
            response = await gate.ask("tool", "run", "mutating")

        assert response.decision == ConfirmDecision.APPROVED
        assert any("broken" in record.message for record in caplog.records)

    async def test_handlers_run_in_order(self):
        gate = ConfirmationGate(timeout=5)
        seen = []

        async def late(request):
            seen.append("late")
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        async def early(request):
            seen.append("early")
            return ConfirmResponse(
                request.call_id, ConfirmDecision.REJECTED, "deferred:x"
            )

        gate.register_handler(late, name="late", order=200)
        gate.register_handler(early, name="early", order=10)

        await gate.ask("tool", "run", "mutating")
        assert seen == ["early", "late"]

    async def test_all_deferred_returns_last_rejected(self):
        gate = ConfirmationGate(timeout=5)

        gate.register_handler(_rejected("deferred:first"), name="a", order=10)
        gate.register_handler(_rejected("deferred:second"), name="b", order=20)

        response = await gate.ask("tool", "run", "mutating")
        assert response.decision == ConfirmDecision.REJECTED
        assert response.message == "deferred:second"

    async def test_register_handler_returns_unregister(self):
        gate = ConfirmationGate(timeout=5)
        called = []

        async def handler(request):
            called.append(True)
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        unregister = gate.register_handler(handler, name="temp")
        unregister()
        unregister()  # idempotent

        response = await gate.ask("tool", "run", "mutating")
        # With no handlers left the gate falls back to auto-approval.
        assert response.decision == ConfirmDecision.AUTO_APPROVED
        assert called == []


class TestSetHandlerBackwardCompat:
    async def test_set_handler_alone_decides(self):
        gate = ConfirmationGate(timeout=5)

        async def channel(request):
            return ConfirmResponse(
                request.call_id, ConfirmDecision.REJECTED, "nope"
            )

        gate.set_handler(channel)
        response = await gate.ask("tool", "run", "mutating")
        assert response.decision == ConfirmDecision.REJECTED
        assert response.message == "nope"

    async def test_set_handler_is_not_stacked(self):
        gate = ConfirmationGate(timeout=5)
        seen = []

        async def first(request):
            seen.append("first")
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        async def second(request):
            seen.append("second")
            return ConfirmResponse(request.call_id, ConfirmDecision.REJECTED)

        gate.set_handler(first)
        gate.set_handler(second)

        response = await gate.ask("tool", "run", "mutating")
        assert response.decision == ConfirmDecision.REJECTED
        assert seen == ["second"]

    async def test_no_handler_auto_approves(self):
        gate = ConfirmationGate(timeout=5)
        response = await gate.ask("tool", "run", "mutating")
        assert response.decision == ConfirmDecision.AUTO_APPROVED

    async def test_clear_handler_removes_channel_handler(self):
        gate = ConfirmationGate(timeout=5)
        gate.set_handler(_approved())
        assert gate.has_handler is True
        gate.clear_handler()
        assert gate.has_handler is False
        response = await gate.ask("tool", "run", "mutating")
        assert response.decision == ConfirmDecision.AUTO_APPROVED
