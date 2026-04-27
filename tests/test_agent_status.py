"""Tests for Agent Status module — REQ-S1 through REQ-S20."""

import time
import unittest

from lumen.core.agent_status import (
    AgentStatusCollector,
    AgentStatusSnapshot,
    ChannelStatus,
    MemoryStats,
    ModuleStatus,
)


# ─── REQ-S1: Empty snapshot returns defaults ────────────────────────────────


class TestEmptySnapshot(unittest.TestCase):
    """REQ-S1: No callbacks registered → returns safe defaults."""

    def test_default_values(self):
        """All fields have safe defaults when no callbacks are registered."""
        collector = AgentStatusCollector(version="1.0.0")
        snap = collector.snapshot()

        self.assertEqual(snap.version, "1.0.0")
        self.assertEqual(snap.model, "")
        self.assertEqual(snap.provider, "")
        self.assertEqual(snap.provider_status, "unknown")
        self.assertFalse(snap.degraded_mode)
        self.assertEqual(snap.channels, [])
        self.assertEqual(snap.modules, [])
        self.assertEqual(snap.tools_count, 0)
        self.assertEqual(snap.active_tools, [])
        self.assertIsInstance(snap.memory_stats, MemoryStats)
        self.assertEqual(snap.memory_stats.total_memories, 0)
        self.assertEqual(snap.sessions_count, 0)
        self.assertEqual(snap.warnings, [])

    def test_uptime_non_negative(self):
        """Uptime is zero or slightly positive right after creation."""
        collector = AgentStatusCollector()
        snap = collector.snapshot()
        self.assertGreaterEqual(snap.uptime_seconds, 0)


# ─── REQ-S2: Model callback ────────────────────────────────────────────────


class TestModelCallback(unittest.TestCase):
    """REQ-S2: Model callback returns correct model string."""

    def test_model_callback_value(self):
        """Snapshot model matches what the callback returns."""
        collector = AgentStatusCollector()
        collector.register_model_callback(lambda: "deepseek-chat")
        snap = collector.snapshot()
        self.assertEqual(snap.model, "deepseek-chat")

    def test_model_callback_can_change(self):
        """Model updates when callback returns different value."""
        model_holder = {"value": "model-a"}
        collector = AgentStatusCollector()
        collector.register_model_callback(lambda: model_holder["value"])

        snap1 = collector.snapshot()
        self.assertEqual(snap1.model, "model-a")

        model_holder["value"] = "model-b"
        snap2 = collector.snapshot()
        self.assertEqual(snap2.model, "model-b")


# ─── REQ-S3: Provider callback ─────────────────────────────────────────────


class TestProviderCallback(unittest.TestCase):
    """REQ-S3: Provider callback returns correct provider name."""

    def test_provider_callback_value(self):
        """Snapshot provider matches what the callback returns."""
        collector = AgentStatusCollector()
        collector.register_provider_callback(lambda: "openrouter")
        snap = collector.snapshot()
        self.assertEqual(snap.provider, "openrouter")


# ─── REQ-S4: Provider status callback ──────────────────────────────────────


class TestProviderStatusCallback(unittest.TestCase):
    """REQ-S4: Provider status callback returns healthy/degraded/down."""

    def test_healthy(self):
        collector = AgentStatusCollector()
        collector.register_provider_status_callback(lambda: "healthy")
        snap = collector.snapshot()
        self.assertEqual(snap.provider_status, "healthy")

    def test_degraded(self):
        collector = AgentStatusCollector()
        collector.register_provider_status_callback(lambda: "degraded")
        snap = collector.snapshot()
        self.assertEqual(snap.provider_status, "degraded")

    def test_down(self):
        collector = AgentStatusCollector()
        collector.register_provider_status_callback(lambda: "down")
        snap = collector.snapshot()
        self.assertEqual(snap.provider_status, "down")


# ─── REQ-S5: Degraded mode callback ────────────────────────────────────────


class TestDegradedModeCallback(unittest.TestCase):
    """REQ-S5: Degraded mode callback returns bool."""

    def test_degraded_true(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: True)
        snap = collector.snapshot()
        self.assertTrue(snap.degraded_mode)

    def test_degraded_false(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: False)
        snap = collector.snapshot()
        self.assertFalse(snap.degraded_mode)


# ─── REQ-S6: Auto-warning for degraded mode ────────────────────────────────


class TestAutoWarningDegradedMode(unittest.TestCase):
    """REQ-S6: degraded_mode=True auto-generates a warning."""

    def test_degraded_mode_adds_warning(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: True)
        snap = collector.snapshot()
        self.assertTrue(
            any("degraded mode" in w.lower() for w in snap.warnings),
            f"Expected degraded mode warning, got: {snap.warnings}",
        )

    def test_no_degraded_no_auto_warning(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: False)
        snap = collector.snapshot()
        self.assertFalse(
            any("degraded mode" in w.lower() for w in snap.warnings),
        )


# ─── REQ-S7: Auto-warning for provider down ────────────────────────────────


class TestAutoWarningProviderDown(unittest.TestCase):
    """REQ-S7: provider_status='down' auto-generates warning."""

    def test_provider_down_adds_warning(self):
        collector = AgentStatusCollector()
        collector.register_provider_callback(lambda: "deepseek")
        collector.register_provider_status_callback(lambda: "down")
        snap = collector.snapshot()
        self.assertTrue(
            any("deepseek" in w and "down" in w for w in snap.warnings),
            f"Expected provider down warning, got: {snap.warnings}",
        )


# ─── REQ-S8: Auto-warning for provider degraded ────────────────────────────


class TestAutoWarningProviderDegraded(unittest.TestCase):
    """REQ-S8: provider_status='degraded' auto-generates warning."""

    def test_provider_degraded_adds_warning(self):
        collector = AgentStatusCollector()
        collector.register_provider_callback(lambda: "openrouter")
        collector.register_provider_status_callback(lambda: "degraded")
        snap = collector.snapshot()
        self.assertTrue(
            any("openrouter" in w and "degraded" in w for w in snap.warnings),
            f"Expected provider degraded warning, got: {snap.warnings}",
        )


# ─── REQ-S9: Channels callback ─────────────────────────────────────────────


class TestChannelsCallback(unittest.TestCase):
    """REQ-S9: Channels callback returns list of ChannelStatus."""

    def test_channels_list(self):
        channels = [
            ChannelStatus(name="telegram", status="connected", message_count=42),
            ChannelStatus(name="web", status="disconnected"),
        ]
        collector = AgentStatusCollector()
        collector.register_channels_callback(lambda: channels)
        snap = collector.snapshot()
        self.assertEqual(len(snap.channels), 2)
        self.assertEqual(snap.channels[0].name, "telegram")
        self.assertEqual(snap.channels[0].message_count, 42)
        self.assertEqual(snap.channels[1].name, "web")


# ─── REQ-S10: Modules callback ─────────────────────────────────────────────


class TestModulesCallback(unittest.TestCase):
    """REQ-S10: Modules callback returns list of ModuleStatus."""

    def test_modules_list(self):
        modules = [
            ModuleStatus(
                name="memory",
                status="active",
                version="1.2.0",
                capabilities=["recall", "store"],
            ),
            ModuleStatus(name="calculator", status="inactive", error="not loaded"),
        ]
        collector = AgentStatusCollector()
        collector.register_modules_callback(lambda: modules)
        snap = collector.snapshot()
        self.assertEqual(len(snap.modules), 2)
        self.assertEqual(snap.modules[0].name, "memory")
        self.assertEqual(snap.modules[0].capabilities, ["recall", "store"])
        self.assertEqual(snap.modules[1].error, "not loaded")


# ─── REQ-S11: Tools callback ───────────────────────────────────────────────


class TestToolsCallback(unittest.TestCase):
    """REQ-S11: Tools callback returns tool names, tools_count matches."""

    def test_tools_count_and_names(self):
        tools = ["web_search", "calculator", "code_exec"]
        collector = AgentStatusCollector()
        collector.register_tools_callback(lambda: tools)
        snap = collector.snapshot()
        self.assertEqual(snap.tools_count, 3)
        self.assertEqual(snap.active_tools, ["web_search", "calculator", "code_exec"])

    def test_empty_tools(self):
        collector = AgentStatusCollector()
        collector.register_tools_callback(lambda: [])
        snap = collector.snapshot()
        self.assertEqual(snap.tools_count, 0)
        self.assertEqual(snap.active_tools, [])


# ─── REQ-S12: Memory callback ──────────────────────────────────────────────


class TestMemoryCallback(unittest.TestCase):
    """REQ-S12: Memory callback returns MemoryStats."""

    def test_memory_stats(self):
        stats = MemoryStats(
            total_memories=150,
            total_sessions=8,
            categories={"work": 80, "personal": 70},
            database_size_kb=2048.5,
        )
        collector = AgentStatusCollector()
        collector.register_memory_callback(lambda: stats)
        snap = collector.snapshot()
        self.assertEqual(snap.memory_stats.total_memories, 150)
        self.assertEqual(snap.memory_stats.total_sessions, 8)
        self.assertEqual(snap.memory_stats.categories, {"work": 80, "personal": 70})
        self.assertAlmostEqual(snap.memory_stats.database_size_kb, 2048.5)


# ─── REQ-S13: Sessions callback ────────────────────────────────────────────


class TestSessionsCallback(unittest.TestCase):
    """REQ-S13: Sessions callback returns session count."""

    def test_sessions_count(self):
        collector = AgentStatusCollector()
        collector.register_sessions_callback(lambda: 12)
        snap = collector.snapshot()
        self.assertEqual(snap.sessions_count, 12)


# ─── REQ-S14: Warnings callback merged with auto-detected ──────────────────


class TestWarningsCallback(unittest.TestCase):
    """REQ-S14: Custom warnings merged with auto-detected ones."""

    def test_custom_warnings_merged(self):
        collector = AgentStatusCollector()
        collector.register_warnings_callback(lambda: ["Disk space low"])
        snap = collector.snapshot()
        self.assertIn("Disk space low", snap.warnings)

    def test_custom_and_auto_warnings_coexist(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: True)
        collector.register_warnings_callback(lambda: ["Disk space low"])
        snap = collector.snapshot()
        self.assertIn("Disk space low", snap.warnings)
        self.assertTrue(
            any("degraded mode" in w.lower() for w in snap.warnings),
        )


# ─── REQ-S15: Warning deduplication ────────────────────────────────────────


class TestWarningDeduplication(unittest.TestCase):
    """REQ-S15: Same warning not repeated."""

    def test_duplicate_auto_warning(self):
        """If callback returns same text as auto-detected, no duplicate."""
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: True)
        # Callback returns exact same warning text as auto-detect
        collector.register_warnings_callback(
            lambda: ["All providers are down — running in degraded mode"]
        )
        snap = collector.snapshot()
        degraded_warnings = [w for w in snap.warnings if "degraded mode" in w.lower()]
        self.assertEqual(len(degraded_warnings), 1)

    def test_multiple_custom_duplicates(self):
        collector = AgentStatusCollector()
        collector.register_warnings_callback(
            lambda: ["Warn A", "Warn A", "Warn B", "Warn B"]
        )
        snap = collector.snapshot()
        self.assertEqual(snap.warnings, ["Warn A", "Warn B"])


# ─── REQ-S16: Callback error handling ──────────────────────────────────────


class TestCallbackErrorHandling(unittest.TestCase):
    """REQ-S16: Throwing callback adds warning, default is used."""

    def test_model_callback_error(self):
        """Model callback raises → warning added, model defaults to ''."""
        collector = AgentStatusCollector()
        collector.register_model_callback(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        snap = collector.snapshot()
        self.assertEqual(snap.model, "")
        self.assertTrue(any("Status callback error" in w for w in snap.warnings))

    def test_provider_callback_error(self):
        """Provider callback raises → provider defaults to ''."""
        collector = AgentStatusCollector()
        collector.register_provider_callback(lambda: 1 / 0)
        snap = collector.snapshot()
        self.assertEqual(snap.provider, "")

    def test_tools_callback_error(self):
        """Tools callback raises → tools defaults to []."""
        collector = AgentStatusCollector()
        collector.register_tools_callback(lambda: (_ for _ in ()).throw(ValueError("nope")))
        snap = collector.snapshot()
        self.assertEqual(snap.tools_count, 0)
        self.assertEqual(snap.active_tools, [])

    def test_memory_callback_error(self):
        """Memory callback raises → defaults to empty MemoryStats."""
        collector = AgentStatusCollector()
        collector.register_memory_callback(lambda: (_ for _ in ()).throw(Exception("mem err")))
        snap = collector.snapshot()
        self.assertIsInstance(snap.memory_stats, MemoryStats)
        self.assertEqual(snap.memory_stats.total_memories, 0)

    def test_other_callbacks_still_work_when_one_fails(self):
        """One failing callback does not prevent others from being collected."""
        collector = AgentStatusCollector()
        collector.register_model_callback(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
        collector.register_provider_callback(lambda: "openrouter")
        snap = collector.snapshot()
        self.assertEqual(snap.model, "")
        self.assertEqual(snap.provider, "openrouter")


# ─── REQ-S17: Health check ─────────────────────────────────────────────────


class TestHealthCheck(unittest.TestCase):
    """REQ-S17: health_check() returns lightweight dict."""

    def test_health_ok_when_not_degraded(self):
        collector = AgentStatusCollector(version="2.0.0")
        collector.register_model_callback(lambda: "gpt-4")
        collector.register_provider_callback(lambda: "openai")
        result = collector.health_check()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["version"], "2.0.0")
        self.assertEqual(result["model"], "gpt-4")
        self.assertEqual(result["provider"], "openai")
        self.assertIn("uptime_seconds", result)
        self.assertIn("provider_status", result)

    def test_health_degraded_when_degraded_mode(self):
        collector = AgentStatusCollector()
        collector.register_degraded_mode_callback(lambda: True)
        result = collector.health_check()
        self.assertEqual(result["status"], "degraded")

    def test_health_uptime_is_number(self):
        collector = AgentStatusCollector()
        result = collector.health_check()
        self.assertIsInstance(result["uptime_seconds"], float)
        self.assertGreaterEqual(result["uptime_seconds"], 0)


# ─── REQ-S18: Uptime increases ─────────────────────────────────────────────


class TestUptime(unittest.TestCase):
    """REQ-S18: uptime_seconds increases over time."""

    def test_uptime_increases(self):
        collector = AgentStatusCollector()
        snap1 = collector.snapshot()
        time.sleep(0.05)
        snap2 = collector.snapshot()
        self.assertGreater(snap2.uptime_seconds, snap1.uptime_seconds)

    def test_uptime_reasonable(self):
        collector = AgentStatusCollector()
        time.sleep(0.1)
        snap = collector.snapshot()
        # Should be at least 0.1s, less than 5s
        self.assertGreater(snap.uptime_seconds, 0.05)
        self.assertLess(snap.uptime_seconds, 5.0)


# ─── REQ-S19: to_dict serialization ────────────────────────────────────────


class TestToDictSerialization(unittest.TestCase):
    """REQ-S19: snapshot.to_dict() serializes completely."""

    def test_full_serialization(self):
        collector = AgentStatusCollector(version="3.0.0")
        collector.register_model_callback(lambda: "claude-3")
        collector.register_provider_callback(lambda: "anthropic")
        collector.register_provider_status_callback(lambda: "healthy")
        collector.register_channels_callback(
            lambda: [ChannelStatus(name="tg", status="connected", message_count=5)]
        )
        collector.register_modules_callback(
            lambda: [ModuleStatus(name="mem", status="active", version="1.0")]
        )
        collector.register_tools_callback(lambda: ["search"])
        collector.register_memory_callback(
            lambda: MemoryStats(total_memories=10, total_sessions=2, database_size_kb=100.123)
        )
        collector.register_sessions_callback(lambda: 7)

        snap = collector.snapshot()
        d = snap.to_dict()

        # Check all top-level keys exist
        expected_keys = [
            "version", "uptime_seconds", "model", "provider", "provider_status",
            "degraded_mode", "channels", "modules", "tools_count", "active_tools",
            "memory_stats", "sessions_count", "warnings",
        ]
        for key in expected_keys:
            self.assertIn(key, d, f"Missing key: {key}")

        # Check nested structures
        self.assertEqual(d["version"], "3.0.0")
        self.assertEqual(d["model"], "claude-3")
        self.assertEqual(d["provider"], "anthropic")
        self.assertEqual(d["provider_status"], "healthy")
        self.assertFalse(d["degraded_mode"])
        self.assertIsInstance(d["channels"], list)
        self.assertEqual(len(d["channels"]), 1)
        self.assertEqual(d["channels"][0]["name"], "tg")
        self.assertEqual(d["channels"][0]["message_count"], 5)
        self.assertIsInstance(d["modules"], list)
        self.assertEqual(d["modules"][0]["name"], "mem")
        self.assertEqual(d["modules"][0]["version"], "1.0")
        self.assertEqual(d["tools_count"], 1)
        self.assertEqual(d["active_tools"], ["search"])
        self.assertEqual(d["memory_stats"]["total_memories"], 10)
        self.assertEqual(d["memory_stats"]["total_sessions"], 2)
        self.assertEqual(d["memory_stats"]["database_size_kb"], 100.1)
        self.assertEqual(d["sessions_count"], 7)

    def test_active_tools_limited_to_20(self):
        """to_dict truncates active_tools to 20 items."""
        snap = AgentStatusSnapshot(
            active_tools=[f"tool_{i}" for i in range(25)],
        )
        d = snap.to_dict()
        self.assertEqual(len(d["active_tools"]), 20)

    def test_uptime_rounded(self):
        """uptime_seconds is rounded to 1 decimal in dict."""
        snap = AgentStatusSnapshot(uptime_seconds=123.456)
        d = snap.to_dict()
        self.assertEqual(d["uptime_seconds"], 123.5)


# ─── REQ-S20: None callback handling ───────────────────────────────────────


class TestNoneCallbackHandling(unittest.TestCase):
    """REQ-S20: None callbacks treated as not registered."""

    def test_register_none_model(self):
        collector = AgentStatusCollector()
        collector.register_model_callback(None)
        snap = collector.snapshot()
        self.assertEqual(snap.model, "")

    def test_register_none_tools(self):
        collector = AgentStatusCollector()
        collector.register_tools_callback(None)
        snap = collector.snapshot()
        self.assertEqual(snap.tools_count, 0)

    def test_register_none_channels(self):
        collector = AgentStatusCollector()
        collector.register_channels_callback(None)
        snap = collector.snapshot()
        self.assertEqual(snap.channels, [])

    def test_register_none_memory(self):
        collector = AgentStatusCollector()
        collector.register_memory_callback(None)
        snap = collector.snapshot()
        self.assertIsInstance(snap.memory_stats, MemoryStats)
        self.assertEqual(snap.memory_stats.total_memories, 0)

    def test_register_none_warnings(self):
        collector = AgentStatusCollector()
        collector.register_warnings_callback(None)
        snap = collector.snapshot()
        self.assertIsInstance(snap.warnings, list)


if __name__ == "__main__":
    unittest.main()
