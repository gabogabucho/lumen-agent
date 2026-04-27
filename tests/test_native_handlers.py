"""Tests for native module event broadcasting — REQ-N1 through REQ-N4."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from lumen.core.module_runtime import (
    ModuleRuntimeContext,
    ModuleRuntimeManager,
    _build_context,
)


class BroadcastEventCallbackTests(unittest.TestCase):
    """REQ-N1: ModuleRuntimeContext.broadcast_event delegates to callback."""

    def test_broadcast_delegates_to_callback(self):
        callback = AsyncMock(return_value=3)
        ctx = ModuleRuntimeContext(
            name="test-mod",
            module_dir=Path("/fake/mod"),
            runtime_dir=Path("/fake/runtime"),
            manifest={},
            config={},
            _broadcast_callback=callback,
        )

        result = asyncio.run(ctx.broadcast_event("status", {"ok": True}))

        assert result == 3
        callback.assert_awaited_once_with("status", {"ok": True})

    def test_broadcast_returns_zero_when_no_callback(self):
        ctx = ModuleRuntimeContext(
            name="test-mod",
            module_dir=Path("/fake/mod"),
            runtime_dir=Path("/fake/runtime"),
            manifest={},
            config={},
            _broadcast_callback=None,
        )

        result = asyncio.run(ctx.broadcast_event("alert", {"msg": "hello"}))

        assert result == 0

    def test_multiple_modules_broadcast_independently(self):
        callback_a = AsyncMock(return_value=1)
        callback_b = AsyncMock(return_value=2)

        ctx_a = ModuleRuntimeContext(
            name="mod-a",
            module_dir=Path("/fake/a"),
            runtime_dir=Path("/fake/runtime/a"),
            manifest={},
            config={},
            _broadcast_callback=callback_a,
        )
        ctx_b = ModuleRuntimeContext(
            name="mod-b",
            module_dir=Path("/fake/b"),
            runtime_dir=Path("/fake/runtime/b"),
            manifest={},
            config={},
            _broadcast_callback=callback_b,
        )

        result_a = asyncio.run(ctx_a.broadcast_event("a_event", {"x": 1}))
        result_b = asyncio.run(ctx_b.broadcast_event("b_event", {"y": 2}))

        assert result_a == 1
        assert result_b == 2
        callback_a.assert_awaited_once_with("a_event", {"x": 1})
        callback_b.assert_awaited_once_with("b_event", {"y": 2})


class ModuleRuntimeManagerBroadcastTests(unittest.TestCase):
    """REQ-N2: ModuleRuntimeManager passes callback through to context."""

    def test_manager_passes_callback_to_context(self):
        callback = AsyncMock(return_value=5)
        manager = ModuleRuntimeManager(
            pkg_dir=Path("/fake/pkg"),
            lumen_dir=Path("/fake/lumen"),
            config={},
            connectors=MagicMock(),
            memory=MagicMock(),
            broadcast_callback=callback,
        )

        assert manager._broadcast_callback is callback

    def test_set_broadcast_callback_updates_manager(self):
        manager = ModuleRuntimeManager(
            pkg_dir=Path("/fake/pkg"),
            lumen_dir=Path("/fake/lumen"),
            config={},
            connectors=MagicMock(),
            memory=MagicMock(),
        )

        new_callback = AsyncMock()
        manager.set_broadcast_callback(new_callback)

        assert manager._broadcast_callback is new_callback


class WebBroadcastEventTests(unittest.TestCase):
    """REQ-N3: broadcast_event delivers to WebSockets and handles stale ones."""

    def setUp(self):
        from lumen.channels import web

        self.web = web
        self._original_sockets = set(web._active_websockets)
        web._active_websockets.clear()

    def tearDown(self):
        self.web._active_websockets.clear()
        self.web._active_websockets.update(self._original_sockets)

    def test_broadcast_event_sends_json_to_all_clients(self):
        ws1 = MagicMock()
        ws1.send_text = AsyncMock()
        ws2 = MagicMock()
        ws2.send_text = AsyncMock()

        self.web._active_websockets.add(ws1)
        self.web._active_websockets.add(ws2)

        result = asyncio.run(self.web.broadcast_event("status", {"ok": True}))

        assert result == 2
        expected = json.dumps({"type": "status", "payload": {"ok": True}})
        ws1.send_text.assert_awaited_once_with(expected)
        ws2.send_text.assert_awaited_once_with(expected)

    def test_broadcast_event_removes_stale_websockets(self):
        ws_good = MagicMock()
        ws_good.send_text = AsyncMock()
        ws_stale = MagicMock()
        ws_stale.send_text = AsyncMock(side_effect=RuntimeError("Connection lost"))

        self.web._active_websockets.add(ws_good)
        self.web._active_websockets.add(ws_stale)

        result = asyncio.run(self.web.broadcast_event("alert", {"msg": "hello"}))

        assert result == 1
        assert ws_stale not in self.web._active_websockets
        assert ws_good in self.web._active_websockets

    def test_broadcast_event_returns_zero_with_no_clients(self):
        result = asyncio.run(self.web.broadcast_event("ping", {}))
        assert result == 0


class BuildContextBroadcastTests(unittest.TestCase):
    """REQ-N4: _build_context wires broadcast_callback correctly."""

    def test_build_context_passes_broadcast_callback(self):
        callback = AsyncMock(return_value=7)
        ctx = _build_context(
            name="test-mod",
            module_dir=Path("/fake/mod"),
            runtime_root=Path("/fake/runtime"),
            config={},
            broadcast_callback=callback,
        )

        assert ctx._broadcast_callback is callback
        result = asyncio.run(ctx.broadcast_event("test", {"v": 1}))
        assert result == 7


class BuildTerminalEnvNamespacedTests(unittest.TestCase):
    """Tests for _build_terminal_env namespacing and cross-section lookup."""

    def test_build_terminal_env_no_overwrite(self):
        """Module-prefixed keys prevent silent overwrites between modules."""
        from lumen.core.handlers import _build_terminal_env

        config = {
            "secrets": {
                "module-a": {"public": {"SCRIPTS_DIR": "/a/scripts", "X": "1"}},
                "module-b": {"public": {"SCRIPTS_DIR": "/b/scripts", "Y": "2"}},
            },
            "terminal": {"env": {
                "public": ["SCRIPTS_DIR", "X", "Y"],
                "secret": [],
                "modules": ["module-a", "module-b"],
            }},
        }
        env = _build_terminal_env(config)
        # Module-prefixed: no overwrite
        assert env["MODULE_A_SCRIPTS_DIR"] == "/a/scripts"
        assert env["MODULE_B_SCRIPTS_DIR"] == "/b/scripts"
        # Cross-section: X from module-a, Y from module-b
        assert env["MODULE_A_X"] == "1"
        assert env["MODULE_B_Y"] == "2"

    def test_build_terminal_env_cross_section(self):
        """Keys in terminal.env.public can live in module's secret section."""
        from lumen.core.handlers import _build_terminal_env

        config = {
            "secrets": {
                "tn": {"secret": {"API_TOKEN": "tok123"}},
            },
            "terminal": {"env": {
                "public": ["API_TOKEN"],  # listed as public...
                "secret": [],
                "modules": ["tn"],
            }},
        }
        env = _build_terminal_env(config)
        assert env["API_TOKEN"] == "tok123"  # ...but found in secret section


if __name__ == "__main__":
    unittest.main()
