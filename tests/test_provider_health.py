"""Tests for provider health tracking — EWMA scoring, backoff, auto-recovery, fallback."""

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from lumen.core.provider_health import (
    DegradationConfig,
    ProviderEntry,
    ProviderHealthTracker,
    ProviderStatus,
)


class TestRegisterProviders(unittest.TestCase):
    """REQ-1: Register providers and verify storage."""

    def test_register_three_providers(self):
        """Register 3 providers, all stored and retrievable."""
        tracker = ProviderHealthTracker()
        tracker.register("deepseek", model="deepseek/deepseek-chat", priority=1)
        tracker.register("gemini", model="google/gemini-2.0-flash", priority=2)
        tracker.register("openai", model="openai/gpt-4o-mini", priority=3)

        self.assertIsNotNone(tracker.get_provider("deepseek"))
        self.assertIsNotNone(tracker.get_provider("gemini"))
        self.assertIsNotNone(tracker.get_provider("openai"))
        self.assertIsNone(tracker.get_provider("nonexistent"))

    def test_register_returns_entry(self):
        """register() returns the ProviderEntry."""
        tracker = ProviderHealthTracker()
        entry = tracker.register("test", model="test/model")
        self.assertIsInstance(entry, ProviderEntry)
        self.assertEqual(entry.name, "test")
        self.assertEqual(entry.model, "test/model")

    def test_register_preserves_fields(self):
        """All registration fields are stored correctly."""
        tracker = ProviderHealthTracker()
        entry = tracker.register(
            name="p1",
            model="m1/model",
            api_base="https://api.example.com",
            api_key_env="MY_KEY",
            priority=10,
        )
        self.assertEqual(entry.api_base, "https://api.example.com")
        self.assertEqual(entry.api_key_env, "MY_KEY")
        self.assertEqual(entry.priority, 10)
        self.assertEqual(entry.status, ProviderStatus.HEALTHY)


class TestRecordSuccess(unittest.TestCase):
    """REQ-2: Record success updates EWMA and resets failures."""

    def setUp(self):
        self.tracker = ProviderHealthTracker()
        self.tracker.register("p1", model="test/model")

    def test_success_updates_counts(self):
        """success_count and total_requests increment."""
        self.tracker.record_success("p1", latency=0.5)
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.success_count, 1)
        self.assertEqual(entry.total_requests, 1)

    def test_success_resets_consecutive_failures(self):
        """consecutive_failures resets to 0 after success."""
        self.tracker.record_failure("p1", error="fail")
        self.tracker.record_failure("p1", error="fail")
        self.tracker.record_success("p1", latency=0.5)
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.consecutive_failures, 0)

    def test_success_sets_status_healthy(self):
        """After success, status is HEALTHY."""
        self.tracker.record_failure("p1", error="fail")
        self.tracker.record_success("p1", latency=0.5)
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.status, ProviderStatus.HEALTHY)

    def test_success_clears_last_error(self):
        """last_error is cleared on success."""
        self.tracker.record_failure("p1", error="boom")
        self.tracker.record_success("p1", latency=0.5)
        entry = self.tracker.get_provider("p1")
        self.assertIsNone(entry.last_error)

    def test_success_clears_backoff(self):
        """backoff_until is cleared on success."""
        self.tracker.record_failure("p1", error="fail")
        self.tracker.record_failure("p1", error="fail")
        self.tracker.record_failure("p1", error="fail")
        entry = self.tracker.get_provider("p1")
        self.assertIsNotNone(entry.backoff_until)
        self.tracker.record_success("p1", latency=0.5)
        self.assertIsNone(entry.backoff_until)

    def test_success_ignores_unknown_provider(self):
        """record_success on unknown provider does not raise."""
        self.tracker.record_success("nonexistent", latency=0.5)  # no error


class TestEWMACalculation(unittest.TestCase):
    """REQ-3: EWMA calculation — first value exact, subsequent smoothed."""

    def setUp(self):
        self.tracker = ProviderHealthTracker()
        self.tracker.register("p1", model="test/model")

    def test_first_latency_sets_ewma_exactly(self):
        """First recorded latency becomes the EWMA value."""
        self.tracker.record_success("p1", latency=1.0)
        entry = self.tracker.get_provider("p1")
        self.assertAlmostEqual(entry.ewma_latency, 1.0)

    def test_ewma_is_smoothed(self):
        """Second value blends with first using alpha=0.3."""
        self.tracker.record_success("p1", latency=1.0)
        self.tracker.record_success("p1", latency=2.0)
        entry = self.tracker.get_provider("p1")
        # EWMA = 0.3 * 2.0 + 0.7 * 1.0 = 0.6 + 0.7 = 1.3
        self.assertAlmostEqual(entry.ewma_latency, 1.3)

    def test_ewma_custom_alpha(self):
        """EWMA uses the alpha from DegradationConfig."""
        config = DegradationConfig(ewma_alpha=0.5)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="test/model")
        tracker.record_success("p1", latency=1.0)
        tracker.record_success("p1", latency=3.0)
        entry = tracker.get_provider("p1")
        # EWMA = 0.5 * 3.0 + 0.5 * 1.0 = 2.0
        self.assertAlmostEqual(entry.ewma_latency, 2.0)

    def test_ewma_responds_to_recent_values(self):
        """After many high latencies, EWMA shifts toward them."""
        for _ in range(20):
            self.tracker.record_success("p1", latency=10.0)
        entry = self.tracker.get_provider("p1")
        self.assertGreater(entry.ewma_latency, 8.0)


class TestRecordFailure(unittest.TestCase):
    """REQ-4: Record failure increments error and consecutive failure counts."""

    def setUp(self):
        self.tracker = ProviderHealthTracker()
        self.tracker.register("p1", model="test/model")

    def test_failure_increments_error_count(self):
        """error_count increments on each failure."""
        self.tracker.record_failure("p1", error="err1")
        self.tracker.record_failure("p1", error="err2")
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.error_count, 2)

    def test_failure_increments_consecutive_failures(self):
        """consecutive_failures increments on each failure."""
        self.tracker.record_failure("p1", error="err")
        self.tracker.record_failure("p1", error="err")
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.consecutive_failures, 2)

    def test_failure_sets_last_error(self):
        """last_error is updated on failure."""
        self.tracker.record_failure("p1", error="Rate limit")
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.last_error, "Rate limit")

    def test_failure_increments_total_requests(self):
        """total_requests increments on failure too."""
        self.tracker.record_failure("p1", error="err")
        entry = self.tracker.get_provider("p1")
        self.assertEqual(entry.total_requests, 1)

    def test_failure_ignores_unknown_provider(self):
        """record_failure on unknown provider does not raise."""
        self.tracker.record_failure("nonexistent", error="err")  # no error


class TestBackoffTriggered(unittest.TestCase):
    """REQ-5: Backoff is triggered after max_consecutive_failures."""

    def test_backoff_set_after_threshold(self):
        """backoff_until is set once consecutive failures >= threshold."""
        config = DegradationConfig(max_consecutive_failures=3)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="test/model")

        tracker.record_failure("p1", error="err1")
        tracker.record_failure("p1", error="err2")
        entry = tracker.get_provider("p1")
        self.assertIsNone(entry.backoff_until)  # not yet

        tracker.record_failure("p1", error="err3")
        entry = tracker.get_provider("p1")
        self.assertIsNotNone(entry.backoff_until)

    def test_backoff_status_is_down(self):
        """Provider status becomes DOWN when in backoff."""
        config = DegradationConfig(max_consecutive_failures=2)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="test/model")
        tracker.record_failure("p1", error="err1")
        tracker.record_failure("p1", error="err2")
        entry = tracker.get_provider("p1")
        self.assertEqual(entry.status, ProviderStatus.DOWN)


class TestExponentialBackoff(unittest.TestCase):
    """REQ-6: Backoff doubles: 1s, 2s, 4s, 8s... capped at max."""

    def setUp(self):
        self.config = DegradationConfig(
            max_consecutive_failures=3,
            max_backoff_seconds=60.0,
        )
        self.tracker = ProviderHealthTracker(degradation_config=self.config)
        self.tracker.register("p1", model="test/model")

    def test_first_backoff_is_one_second(self):
        """At exactly max_consecutive_failures, backoff = 2^0 = 1s."""
        for i in range(3):
            self.tracker.record_failure("p1", error=f"err{i}")
        entry = self.tracker.get_provider("p1")
        now = datetime.now(timezone.utc)
        diff = (entry.backoff_until - now).total_seconds()
        self.assertAlmostEqual(diff, 1.0, places=0)

    def test_backoff_doubles_each_time(self):
        """Backoff doubles: 1s → 2s → 4s."""
        backoffs = []
        for i in range(5):  # 3 to reach threshold, then 2 more
            self.tracker.record_failure("p1", error=f"err{i}")
            entry = self.tracker.get_provider("p1")
            if entry.backoff_until:
                now = datetime.now(timezone.utc)
                backoffs.append((entry.backoff_until - now).total_seconds())

        # At fail 3: 2^0=1s, fail 4: 2^1=2s, fail 5: 2^2=4s
        self.assertAlmostEqual(backoffs[0], 1.0, places=0)
        self.assertAlmostEqual(backoffs[1], 2.0, places=0)
        self.assertAlmostEqual(backoffs[2], 4.0, places=0)

    def test_backoff_capped_at_max(self):
        """Backoff never exceeds max_backoff_seconds."""
        small_config = DegradationConfig(
            max_consecutive_failures=3,
            max_backoff_seconds=5.0,
        )
        tracker = ProviderHealthTracker(degradation_config=small_config)
        tracker.register("p1", model="test/model")

        # Generate many failures to exceed cap
        for i in range(20):
            tracker.record_failure("p1", error=f"err{i}")
        entry = tracker.get_provider("p1")
        now = datetime.now(timezone.utc)
        diff = (entry.backoff_until - now).total_seconds()
        self.assertLessEqual(diff, 5.0)


class TestGetBestProvider(unittest.TestCase):
    """REQ-7: get_best_provider returns lowest priority healthy provider."""

    def test_returns_lowest_priority(self):
        """Provider with priority=1 wins over priority=2."""
        tracker = ProviderHealthTracker()
        tracker.register("slow", model="slow/model", priority=10)
        tracker.register("fast", model="fast/model", priority=1)
        best = tracker.get_best_provider()
        self.assertEqual(best.name, "fast")

    def test_returns_none_when_empty(self):
        """No providers registered → None."""
        tracker = ProviderHealthTracker()
        self.assertIsNone(tracker.get_best_provider())

    def test_sorts_by_priority_then_latency(self):
        """Same priority: lower EWMA latency wins."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="model1", priority=1)
        tracker.register("p2", model="model2", priority=1)
        tracker.record_success("p1", latency=5.0)
        tracker.record_success("p2", latency=0.5)
        best = tracker.get_best_provider()
        self.assertEqual(best.name, "p2")


class TestGetBestSkipsBackoff(unittest.TestCase):
    """REQ-8: Providers in backoff are skipped by get_best_provider."""

    def test_skips_provider_in_backoff(self):
        """Provider in backoff is not returned."""
        config = DegradationConfig(max_consecutive_failures=2)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("primary", model="primary/model", priority=1)
        tracker.register("backup", model="backup/model", priority=2)

        # Put primary in backoff
        tracker.record_failure("primary", error="err1")
        tracker.record_failure("primary", error="err2")

        best = tracker.get_best_provider()
        self.assertEqual(best.name, "backup")

    def test_returns_none_when_all_in_backoff(self):
        """All providers in backoff → None."""
        config = DegradationConfig(max_consecutive_failures=2)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="m1", priority=1)
        tracker.register("p2", model="m2", priority=2)

        tracker.record_failure("p1", error="err1")
        tracker.record_failure("p1", error="err2")
        tracker.record_failure("p2", error="err1")
        tracker.record_failure("p2", error="err2")

        self.assertIsNone(tracker.get_best_provider())


class TestAutoRecovery(unittest.TestCase):
    """REQ-9: After backoff expires, provider auto-recovers to healthy."""

    def test_auto_recovery_after_backoff_expires(self):
        """Provider recovers when current time >= backoff_until."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")

        # Simulate past backoff
        entry = tracker.get_provider("p1")
        entry.backoff_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        entry.consecutive_failures = 5
        entry.status = ProviderStatus.DOWN

        # get_best_provider triggers _update_status
        best = tracker.get_best_provider()
        self.assertEqual(best.name, "p1")
        self.assertEqual(best.status, ProviderStatus.HEALTHY)
        self.assertEqual(best.consecutive_failures, 0)
        self.assertIsNone(best.backoff_until)

    def test_no_recovery_while_backoff_active(self):
        """Provider stays DOWN while backoff_until is in the future."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        entry = tracker.get_provider("p1")
        entry.backoff_until = datetime.now(timezone.utc) + timedelta(hours=1)
        entry.status = ProviderStatus.DOWN

        best = tracker.get_best_provider()
        self.assertIsNone(best)  # skipped


class TestDegradedMode(unittest.TestCase):
    """REQ-10: is_degraded_mode returns True when all providers are in backoff."""

    def test_degraded_when_all_in_backoff(self):
        """All providers in backoff → degraded mode."""
        config = DegradationConfig(max_consecutive_failures=1)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="m1")
        tracker.register("p2", model="m2")

        tracker.record_failure("p1", error="err")
        tracker.record_failure("p2", error="err")

        self.assertTrue(tracker.is_degraded_mode())

    def test_not_degraded_when_one_available(self):
        """At least one provider available → not degraded."""
        config = DegradationConfig(max_consecutive_failures=1)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="m1")
        tracker.register("p2", model="m2")

        tracker.record_failure("p1", error="err")
        # p2 still healthy

        self.assertFalse(tracker.is_degraded_mode())

    def test_not_degraded_when_empty(self):
        """No providers → not degraded (nothing to be degraded)."""
        tracker = ProviderHealthTracker()
        self.assertFalse(tracker.is_degraded_mode())


class TestRetryProvider(unittest.TestCase):
    """REQ-11: Manual retry clears backoff and resets status."""

    def test_retry_clears_backoff(self):
        """retry_provider clears backoff_until."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        entry = tracker.get_provider("p1")
        entry.backoff_until = datetime.now(timezone.utc) + timedelta(hours=1)
        entry.status = ProviderStatus.DOWN
        entry.consecutive_failures = 5

        result = tracker.retry_provider("p1")
        self.assertTrue(result)
        self.assertIsNone(entry.backoff_until)
        self.assertEqual(entry.consecutive_failures, 0)
        self.assertEqual(entry.status, ProviderStatus.HEALTHY)

    def test_retry_nonexistent_returns_false(self):
        """retry_provider returns False for unknown provider."""
        tracker = ProviderHealthTracker()
        self.assertFalse(tracker.retry_provider("nonexistent"))


class TestFallbackModel(unittest.TestCase):
    """REQ-12: get_summary returns fallback_model when all providers down."""

    def test_summary_uses_fallback_when_all_down(self):
        """When all providers in backoff, current_model = fallback_model."""
        config = DegradationConfig(max_consecutive_failures=1)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="m1")
        tracker.fallback_model = "fallback/fallback-model"

        tracker.record_failure("p1", error="err")

        summary = tracker.get_summary()
        self.assertTrue(summary["degraded_mode"])
        self.assertEqual(summary["current_model"], "fallback/fallback-model")
        self.assertIsNone(summary["current_provider"])

    def test_fallback_property(self):
        """fallback_model getter/setter works."""
        tracker = ProviderHealthTracker()
        self.assertEqual(tracker.fallback_model, "")
        tracker.fallback_model = "openai/gpt-4o-mini"
        self.assertEqual(tracker.fallback_model, "openai/gpt-4o-mini")


class TestStatusTransitions(unittest.TestCase):
    """REQ-13: healthy → degraded → down → healthy."""

    def test_healthy_to_degraded(self):
        """One failure transitions to DEGRADED."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        tracker.record_failure("p1", error="err")
        entry = tracker.get_provider("p1")
        self.assertEqual(entry.status, ProviderStatus.DEGRADED)

    def test_degraded_to_down(self):
        """Reaching max consecutive failures transitions to DOWN."""
        config = DegradationConfig(max_consecutive_failures=3)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="test/model")
        for i in range(3):
            tracker.record_failure("p1", error=f"err{i}")
        entry = tracker.get_provider("p1")
        self.assertEqual(entry.status, ProviderStatus.DOWN)

    def test_down_to_healthy_via_recovery(self):
        """After backoff expires, auto-recovery to HEALTHY."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        entry = tracker.get_provider("p1")
        entry.backoff_until = datetime.now(timezone.utc) - timedelta(seconds=1)
        entry.consecutive_failures = 5
        entry.status = ProviderStatus.DOWN

        tracker.get_best_provider()  # triggers _update_status
        self.assertEqual(entry.status, ProviderStatus.HEALTHY)

    def test_down_to_healthy_via_success(self):
        """Success during backoff clears backoff and sets HEALTHY."""
        config = DegradationConfig(max_consecutive_failures=2)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("p1", model="test/model")
        tracker.record_failure("p1", error="err1")
        tracker.record_failure("p1", error="err2")
        entry = tracker.get_provider("p1")
        self.assertEqual(entry.status, ProviderStatus.DOWN)

        tracker.record_success("p1", latency=0.5)
        self.assertEqual(entry.status, ProviderStatus.HEALTHY)

    def test_degraded_to_healthy_via_success(self):
        """Success after partial failures resets to HEALTHY."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        tracker.record_failure("p1", error="err")
        self.assertEqual(tracker.get_provider("p1").status, ProviderStatus.DEGRADED)

        tracker.record_success("p1", latency=0.5)
        self.assertEqual(tracker.get_provider("p1").status, ProviderStatus.HEALTHY)


class TestGetAllStatus(unittest.TestCase):
    """REQ-14: get_all_status returns complete dicts for all providers."""

    def test_returns_all_providers(self):
        """Returns one dict per registered provider."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="m1", priority=1)
        tracker.register("p2", model="m2", priority=2)

        statuses = tracker.get_all_status()
        self.assertEqual(len(statuses), 2)

    def test_dict_has_all_fields(self):
        """Each status dict contains all expected keys."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        tracker.record_success("p1", latency=0.5)

        statuses = tracker.get_all_status()
        s = statuses[0]
        expected_keys = {
            "name", "model", "api_base", "priority", "status",
            "ewma_latency", "error_count", "success_count",
            "total_requests", "consecutive_failures", "last_error",
            "last_success", "backoff_until", "in_backoff",
        }
        self.assertEqual(set(s.keys()), expected_keys)

    def test_ewma_latency_is_rounded(self):
        """ewma_latency is rounded to 3 decimal places."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="test/model")
        tracker.record_success("p1", latency=0.123456)

        statuses = tracker.get_all_status()
        self.assertEqual(len(str(statuses[0]["ewma_latency"]).split(".")[-1]), 3)

    def test_empty_tracker_returns_empty_list(self):
        """No providers → empty list."""
        tracker = ProviderHealthTracker()
        self.assertEqual(tracker.get_all_status(), [])


class TestGetSummary(unittest.TestCase):
    """REQ-15: get_summary returns aggregate stats."""

    def test_summary_structure(self):
        """Summary dict has all expected top-level keys."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="m1")
        summary = tracker.get_summary()
        expected_keys = {
            "total_providers", "healthy", "degraded", "down",
            "degraded_mode", "current_provider", "current_model",
            "providers",
        }
        self.assertEqual(set(summary.keys()), expected_keys)

    def test_summary_counts(self):
        """Aggregate counts match provider statuses."""
        config = DegradationConfig(max_consecutive_failures=2)
        tracker = ProviderHealthTracker(degradation_config=config)
        tracker.register("healthy_p", model="m1")
        tracker.register("degraded_p", model="m2")
        tracker.register("down_p", model="m3")

        # 1 failure → DEGRADED (below threshold of 2)
        tracker.record_failure("degraded_p", error="err")
        # 2 failures → DOWN (hits threshold, enters backoff)
        tracker.record_failure("down_p", error="err1")
        tracker.record_failure("down_p", error="err2")

        summary = tracker.get_summary()
        self.assertEqual(summary["total_providers"], 3)
        self.assertEqual(summary["healthy"], 1)
        self.assertEqual(summary["degraded"], 1)
        self.assertEqual(summary["down"], 1)

    def test_summary_current_provider(self):
        """current_provider is the best available provider."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="m1", priority=1)
        summary = tracker.get_summary()
        self.assertEqual(summary["current_provider"], "p1")
        self.assertEqual(summary["current_model"], "m1")


class TestFromConfigWithChain(unittest.TestCase):
    """REQ-16: from_config builds tracker from providers.chain."""

    def test_chain_creates_providers(self):
        """Chain entries are registered as providers."""
        config = {
            "providers": {
                "chain": [
                    {"name": "deepseek", "model": "deepseek/deepseek-chat", "priority": 1},
                    {"name": "gemini", "model": "google/gemini-2.0-flash", "priority": 2},
                ],
            },
        }
        tracker = ProviderHealthTracker.from_config(config)
        self.assertIsNotNone(tracker.get_provider("deepseek"))
        self.assertIsNotNone(tracker.get_provider("gemini"))
        self.assertEqual(tracker.get_provider("deepseek").priority, 1)
        self.assertEqual(tracker.get_provider("gemini").priority, 2)

    def test_chain_ignores_invalid_entries(self):
        """Non-dict or nameless entries in chain are skipped."""
        config = {
            "providers": {
                "chain": [
                    {"name": "valid", "model": "m1"},
                    "not a dict",
                    {"model": "no_name"},
                    None,
                ],
            },
        }
        tracker = ProviderHealthTracker.from_config(config)
        self.assertIsNotNone(tracker.get_provider("valid"))
        self.assertEqual(len(tracker.get_all_status()), 1)

    def test_chain_with_fallback(self):
        """models.fallback is set as fallback_model."""
        config = {
            "providers": {
                "chain": [
                    {"name": "p1", "model": "m1"},
                ],
            },
            "models": {"fallback": "openai/gpt-4o-mini"},
        }
        tracker = ProviderHealthTracker.from_config(config)
        self.assertEqual(tracker.fallback_model, "openai/gpt-4o-mini")


class TestFromConfigLegacy(unittest.TestCase):
    """REQ-17: from_config builds tracker from single model config."""

    def test_legacy_model_registration(self):
        """Single model in config root creates a provider."""
        config = {
            "model": "deepseek/deepseek-chat",
            "provider": "deepseek",
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_KEY",
        }
        tracker = ProviderHealthTracker.from_config(config)
        entry = tracker.get_provider("deepseek")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.model, "deepseek/deepseek-chat")
        self.assertEqual(entry.api_base, "https://api.deepseek.com")
        self.assertEqual(entry.api_key_env, "DEEPSEEK_KEY")
        self.assertEqual(entry.priority, 1)

    def test_legacy_defaults_provider_name(self):
        """When no provider name, defaults to 'default'."""
        config = {"model": "some/model"}
        tracker = ProviderHealthTracker.from_config(config)
        self.assertIsNotNone(tracker.get_provider("default"))

    def test_chain_takes_precedence_over_legacy(self):
        """If chain is present, legacy model is ignored."""
        config = {
            "model": "legacy/model",
            "provider": "legacy",
            "providers": {
                "chain": [
                    {"name": "chain_p", "model": "chain/model"},
                ],
            },
        }
        tracker = ProviderHealthTracker.from_config(config)
        self.assertIsNotNone(tracker.get_provider("chain_p"))
        self.assertIsNone(tracker.get_provider("legacy"))


class TestFromConfigEmpty(unittest.TestCase):
    """REQ-18: from_config with None/empty returns empty tracker."""

    def test_none_config(self):
        """None config → empty tracker."""
        tracker = ProviderHealthTracker.from_config(None)
        self.assertEqual(tracker.get_all_status(), [])

    def test_empty_dict_config(self):
        """Empty dict → empty tracker."""
        tracker = ProviderHealthTracker.from_config({})
        self.assertEqual(tracker.get_all_status(), [])

    def test_non_dict_config(self):
        """Non-dict config → empty tracker."""
        tracker = ProviderHealthTracker.from_config("not a dict")
        self.assertEqual(tracker.get_all_status(), [])


class TestUnregister(unittest.TestCase):
    """REQ-19: Unregister removes a provider."""

    def test_unregister_existing(self):
        """Existing provider is removed."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="m1")
        tracker.unregister("p1")
        self.assertIsNone(tracker.get_provider("p1"))
        self.assertEqual(len(tracker.get_all_status()), 0)

    def test_unregister_nonexistent(self):
        """Unregistering unknown provider does not raise."""
        tracker = ProviderHealthTracker()
        tracker.unregister("nonexistent")  # no error

    def test_unregister_from_multiple(self):
        """Unregister one, others remain."""
        tracker = ProviderHealthTracker()
        tracker.register("p1", model="m1")
        tracker.register("p2", model="m2")
        tracker.register("p3", model="m3")
        tracker.unregister("p2")
        self.assertIsNotNone(tracker.get_provider("p1"))
        self.assertIsNone(tracker.get_provider("p2"))
        self.assertIsNotNone(tracker.get_provider("p3"))


class TestDegradationConfigFromConfig(unittest.TestCase):
    """REQ-20: DegradationConfig.from_config parses degradation settings."""

    def test_none_returns_defaults(self):
        """None → default config."""
        cfg = DegradationConfig.from_config(None)
        self.assertEqual(cfg.max_backoff_seconds, 60.0)
        self.assertEqual(cfg.auto_recovery_seconds, 60.0)
        self.assertEqual(cfg.max_consecutive_failures, 3)
        self.assertEqual(cfg.ewma_alpha, 0.3)

    def test_empty_dict_returns_defaults(self):
        """Empty dict → default config."""
        cfg = DegradationConfig.from_config({})
        self.assertEqual(cfg.max_backoff_seconds, 60.0)

    def test_custom_values_parsed(self):
        """Custom degradation values are parsed."""
        config = {
            "degradation": {
                "max_backoff_seconds": 120.0,
                "auto_recovery_seconds": 30.0,
                "max_consecutive_failures": 5,
                "ewma_alpha": 0.5,
            },
        }
        cfg = DegradationConfig.from_config(config)
        self.assertEqual(cfg.max_backoff_seconds, 120.0)
        self.assertEqual(cfg.auto_recovery_seconds, 30.0)
        self.assertEqual(cfg.max_consecutive_failures, 5)
        self.assertEqual(cfg.ewma_alpha, 0.5)

    def test_partial_values_use_defaults(self):
        """Missing degradation keys use defaults."""
        config = {
            "degradation": {
                "max_backoff_seconds": 10.0,
            },
        }
        cfg = DegradationConfig.from_config(config)
        self.assertEqual(cfg.max_backoff_seconds, 10.0)
        self.assertEqual(cfg.auto_recovery_seconds, 60.0)  # default

    def test_non_dict_degradation_returns_defaults(self):
        """degradation value is not a dict → defaults."""
        config = {"degradation": "invalid"}
        cfg = DegradationConfig.from_config(config)
        self.assertEqual(cfg.max_backoff_seconds, 60.0)


if __name__ == "__main__":
    unittest.main()
