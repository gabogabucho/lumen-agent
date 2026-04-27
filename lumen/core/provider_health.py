"""Provider health tracking with EWMA scoring and graceful degradation.

Tracks latency and error rates per provider using Exponentially Weighted
Moving Average (EWMA). Implements exponential backoff on failures and
automatic recovery after cooldown. Supports a provider chain with
priority-based fallback.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass
class ProviderEntry:
    """Health state for a single provider."""
    name: str
    model: str
    api_base: str = ""
    api_key_env: str = ""
    priority: int = 50

    # Health metrics
    ewma_latency: float = 0.0       # EWMA of response latency (seconds)
    error_count: int = 0
    success_count: int = 0
    total_requests: int = 0
    consecutive_failures: int = 0
    last_error: str | None = None
    last_success: datetime | None = None
    last_request: datetime | None = None

    # Backoff state
    backoff_until: datetime | None = None
    status: ProviderStatus = ProviderStatus.HEALTHY

    # EWMA alpha (smoothing factor) — higher = more responsive to recent values
    _ewma_alpha: float = 0.3


@dataclass
class DegradationConfig:
    """Configuration for provider degradation behavior."""
    max_backoff_seconds: float = 60.0
    auto_recovery_seconds: float = 60.0
    max_consecutive_failures: int = 3
    ewma_alpha: float = 0.3

    @classmethod
    def from_config(cls, config: dict | None) -> "DegradationConfig":
        if not config or not isinstance(config, dict):
            return cls()
        deg = config.get("degradation", {})
        if not isinstance(deg, dict):
            return cls()
        return cls(
            max_backoff_seconds=float(deg.get("max_backoff_seconds", cls.max_backoff_seconds)),
            auto_recovery_seconds=float(deg.get("auto_recovery_seconds", cls.auto_recovery_seconds)),
            max_consecutive_failures=int(deg.get("max_consecutive_failures", cls.max_consecutive_failures)),
            ewma_alpha=float(deg.get("ewma_alpha", cls.ewma_alpha)),
        )


class ProviderHealthTracker:
    """Tracks health of multiple providers and manages fallback chain.

    Usage:
        tracker = ProviderHealthTracker()
        tracker.register("deepseek", model="deepseek/deepseek-chat", priority=1)
        tracker.register("gemini", model="google/gemini-2.0-flash", priority=2)

        # After a successful call:
        tracker.record_success("deepseek", latency=0.5)

        # After a failed call:
        tracker.record_failure("deepseek", error="Rate limit exceeded")

        # Get the best available provider:
        entry = tracker.get_best_provider()
        if entry:
            model = entry.model
        else:
            model = tracker.fallback_model  # guaranteed fallback
    """

    def __init__(self, degradation_config: DegradationConfig | None = None):
        self._providers: dict[str, ProviderEntry] = {}
        self._degradation = degradation_config or DegradationConfig()
        self._fallback_model: str = ""

    @property
    def fallback_model(self) -> str:
        """Return the guaranteed fallback model string."""
        return self._fallback_model

    @fallback_model.setter
    def fallback_model(self, value: str) -> None:
        self._fallback_model = value

    def register(
        self,
        name: str,
        model: str,
        api_base: str = "",
        api_key_env: str = "",
        priority: int = 50,
    ) -> ProviderEntry:
        """Register a new provider."""
        entry = ProviderEntry(
            name=name,
            model=model,
            api_base=api_base,
            api_key_env=api_key_env,
            priority=priority,
            _ewma_alpha=self._degradation.ewma_alpha,
        )
        self._providers[name] = entry
        return entry

    def unregister(self, name: str) -> None:
        """Remove a provider."""
        self._providers.pop(name, None)

    def get_provider(self, name: str) -> ProviderEntry | None:
        """Get a provider entry by name."""
        return self._providers.get(name)

    def record_success(self, name: str, latency: float) -> None:
        """Record a successful request. Updates EWMA and resets failure state."""
        entry = self._providers.get(name)
        if not entry:
            return

        now = datetime.now(timezone.utc)
        alpha = entry._ewma_alpha

        # Update EWMA latency
        if entry.ewma_latency == 0.0:
            entry.ewma_latency = latency
        else:
            entry.ewma_latency = alpha * latency + (1 - alpha) * entry.ewma_latency

        entry.success_count += 1
        entry.total_requests += 1
        entry.consecutive_failures = 0
        entry.last_success = now
        entry.last_request = now
        entry.last_error = None

        # Clear backoff
        entry.backoff_until = None

        # Update status
        self._update_status(entry)

    def record_failure(self, name: str, error: str) -> None:
        """Record a failed request. Triggers backoff if threshold exceeded."""
        entry = self._providers.get(name)
        if not entry:
            return

        now = datetime.now(timezone.utc)

        entry.error_count += 1
        entry.total_requests += 1
        entry.consecutive_failures += 1
        entry.last_error = error
        entry.last_request = now

        # Calculate backoff if threshold exceeded
        max_fails = self._degradation.max_consecutive_failures
        if entry.consecutive_failures >= max_fails:
            # Exponential backoff: 2^(fails - max_fails) capped at max_backoff
            backoff_exponent = entry.consecutive_failures - max_fails
            backoff_seconds = min(
                (2 ** backoff_exponent),
                self._degradation.max_backoff_seconds,
            )
            entry.backoff_until = now + timedelta(seconds=backoff_seconds)

        self._update_status(entry)

    def _update_status(self, entry: ProviderEntry) -> None:
        """Update provider status based on current state."""
        now = datetime.now(timezone.utc)

        # Check if backoff expired → auto-recovery
        if (entry.backoff_until and now >= entry.backoff_until):
            entry.backoff_until = None
            entry.consecutive_failures = 0
            entry.status = ProviderStatus.HEALTHY
            logger.info(f"Provider '{entry.name}' auto-recovered")
            return

        # Determine status
        if entry.backoff_until:
            entry.status = ProviderStatus.DOWN
        elif entry.consecutive_failures > 0:
            entry.status = ProviderStatus.DEGRADED
        else:
            entry.status = ProviderStatus.HEALTHY

    def get_best_provider(self) -> ProviderEntry | None:
        """Get the best available provider (healthy, not in backoff).

        Providers are sorted by priority (lower number = higher priority).
        """
        now = datetime.now(timezone.utc)

        available = []
        for entry in self._providers.values():
            self._update_status(entry)  # Check for auto-recovery

            # Skip if in backoff
            if entry.backoff_until and now < entry.backoff_until:
                continue

            available.append(entry)

        if not available:
            return None

        # Sort by priority (lower = better), then by EWMA latency (lower = better)
        available.sort(key=lambda e: (e.priority, e.ewma_latency))
        return available[0]

    def is_degraded_mode(self) -> bool:
        """Check if ALL providers are down (degraded mode)."""
        if not self._providers:
            return False
        now = datetime.now(timezone.utc)
        for entry in self._providers.values():
            self._update_status(entry)
            if not (entry.backoff_until and now < entry.backoff_until):
                return False  # At least one provider available
        return True  # All providers are in backoff

    def retry_provider(self, name: str) -> bool:
        """Manually retry a provider by clearing its backoff.

        Returns True if the provider exists and was reset.
        """
        entry = self._providers.get(name)
        if not entry:
            return False
        entry.backoff_until = None
        entry.consecutive_failures = 0
        entry.status = ProviderStatus.HEALTHY
        return True

    def get_all_status(self) -> list[dict[str, Any]]:
        """Return health status of all providers for API/UI consumption."""
        now = datetime.now(timezone.utc)
        result = []
        for entry in self._providers.values():
            self._update_status(entry)
            result.append({
                "name": entry.name,
                "model": entry.model,
                "api_base": entry.api_base,
                "priority": entry.priority,
                "status": entry.status.value,
                "ewma_latency": round(entry.ewma_latency, 3),
                "error_count": entry.error_count,
                "success_count": entry.success_count,
                "total_requests": entry.total_requests,
                "consecutive_failures": entry.consecutive_failures,
                "last_error": entry.last_error,
                "last_success": entry.last_success.isoformat() if entry.last_success else None,
                "backoff_until": entry.backoff_until.isoformat() if entry.backoff_until else None,
                "in_backoff": bool(entry.backoff_until and now < entry.backoff_until),
            })
        return result

    def get_summary(self) -> dict[str, Any]:
        """Return a summary dict for agent status."""
        statuses = self.get_all_status()
        healthy = sum(1 for s in statuses if s["status"] == "healthy")
        degraded = sum(1 for s in statuses if s["status"] == "degraded")
        down = sum(1 for s in statuses if s["status"] == "down")

        best = self.get_best_provider()

        return {
            "total_providers": len(statuses),
            "healthy": healthy,
            "degraded": degraded,
            "down": down,
            "degraded_mode": self.is_degraded_mode(),
            "current_provider": best.name if best else None,
            "current_model": best.model if best else self._fallback_model,
            "providers": statuses,
        }

    @classmethod
    def from_config(cls, config: dict | None) -> "ProviderHealthTracker":
        """Build tracker from full Lumen config dict.

        Looks for:
          config["providers"]["chain"] — list of provider entries
          config["providers"]["degradation"] — degradation settings
          config["model"] — legacy single-model (becomes default provider)
          config["api_key_env"] — for the default provider
          config["api_base"] — for the default provider
        """
        tracker = cls()

        if not config or not isinstance(config, dict):
            return tracker

        # Parse degradation config
        providers_cfg = config.get("providers", {})
        tracker._degradation = DegradationConfig.from_config(config)

        # Parse provider chain
        chain = providers_cfg.get("chain", []) if isinstance(providers_cfg, dict) else []

        if chain and isinstance(chain, list):
            for entry in chain:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name", "")
                if not name:
                    continue
                tracker.register(
                    name=name,
                    model=entry.get("model", ""),
                    api_base=entry.get("api_base", ""),
                    api_key_env=entry.get("api_key_env", ""),
                    priority=int(entry.get("priority", 50)),
                )
        else:
            # Legacy: register single provider from config root
            model = config.get("model", "")
            if model:
                tracker.register(
                    name=config.get("provider", "default"),
                    model=model,
                    api_base=config.get("api_base", ""),
                    api_key_env=config.get("api_key_env", ""),
                    priority=1,
                )

        # Set fallback from model router config if available
        models_cfg = config.get("models", {})
        if isinstance(models_cfg, dict):
            tracker._fallback_model = models_cfg.get("fallback", "")

        return tracker
