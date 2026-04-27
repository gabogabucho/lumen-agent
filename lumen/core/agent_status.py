"""Agent status — consolidated observable state of the Lumen runtime.

Provides a single entry point to understand "what's happening" with the agent:
current model, provider health, active channels, loaded modules, registered
tools, memory stats, and any warnings. Designed for both API responses and
CLI output.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ChannelStatus:
    """Status of a single communication channel."""
    name: str
    status: str = "disconnected"  # connected, disconnected, error
    message_count: int = 0
    last_activity: str | None = None
    error: str | None = None
    is_internal: bool = False  # True for web channel (bypasses inbox)


@dataclass
class ModuleStatus:
    """Status of a single module."""
    name: str
    status: str = "inactive"  # active, inactive, error
    version: str = ""
    capabilities: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class MemoryStats:
    """Memory system statistics."""
    total_memories: int = 0
    total_sessions: int = 0
    categories: dict[str, int] = field(default_factory=dict)
    database_size_kb: float = 0.0


@dataclass
class AgentStatusSnapshot:
    """Complete snapshot of the agent's current state."""
    version: str = "unknown"
    uptime_seconds: float = 0.0
    model: str = ""
    provider: str = ""
    provider_status: str = "unknown"  # healthy, degraded, down
    degraded_mode: bool = False
    channels: list[ChannelStatus] = field(default_factory=list)
    modules: list[ModuleStatus] = field(default_factory=list)
    tools_count: int = 0
    active_tools: list[str] = field(default_factory=list)
    memory_stats: MemoryStats = field(default_factory=MemoryStats)
    sessions_count: int = 0
    warnings: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to API-friendly dict."""
        return {
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "model": self.model,
            "provider": self.provider,
            "provider_status": self.provider_status,
            "degraded_mode": self.degraded_mode,
            "channels": [
                {"name": c.name, "status": c.status, "message_count": c.message_count,
                 "last_activity": c.last_activity, "error": c.error}
                for c in self.channels
            ],
            "modules": [
                {"name": m.name, "status": m.status, "version": m.version,
                 "capabilities": m.capabilities, "error": m.error}
                for m in self.modules
            ],
            "tools_count": self.tools_count,
            "active_tools": self.active_tools[:20],  # Limit for API response
            "memory_stats": {
                "total_memories": self.memory_stats.total_memories,
                "total_sessions": self.memory_stats.total_sessions,
                "categories": self.memory_stats.categories,
                "database_size_kb": round(self.memory_stats.database_size_kb, 1),
            },
            "sessions_count": self.sessions_count,
            "warnings": self.warnings,
        }


class AgentStatusCollector:
    """Collects status from all runtime components into a unified snapshot.
    
    This is the single source of truth for "what's happening" with the agent.
    Components are gathered via callbacks registered at runtime.
    
    Usage:
        collector = AgentStatusCollector(version="0.8.0")
        
        # Register callbacks (done once at bootstrap)
        collector.register_model_callback(lambda: brain.model)
        collector.register_provider_callback(lambda: "deepseek")
        collector.register_tools_callback(lambda: list(registry.as_tools().keys()))
        
        # Get status anytime
        status = collector.snapshot()
        print(status.to_dict())
    """

    def __init__(self, version: str = "unknown"):
        self._version = version
        self._start_time = time.monotonic()
        
        # Callbacks registered by runtime components
        self._model_cb: Any = None
        self._provider_cb: Any = None
        self._provider_status_cb: Any = None
        self._degraded_mode_cb: Any = None
        self._channels_cb: Any = None
        self._modules_cb: Any = None
        self._tools_cb: Any = None
        self._memory_cb: Any = None
        self._sessions_cb: Any = None
        self._warnings_cb: Any = None

    def register_model_callback(self, cb) -> None:
        """Register callback that returns current model string."""
        self._model_cb = cb

    def register_provider_callback(self, cb) -> None:
        """Register callback that returns current provider name string."""
        self._provider_cb = cb

    def register_provider_status_callback(self, cb) -> None:
        """Register callback that returns provider status string."""
        self._provider_status_cb = cb

    def register_degraded_mode_callback(self, cb) -> None:
        """Register callback that returns bool (is degraded mode)."""
        self._degraded_mode_cb = cb

    def register_channels_callback(self, cb) -> None:
        """Register callback that returns list of ChannelStatus."""
        self._channels_cb = cb

    def register_modules_callback(self, cb) -> None:
        """Register callback that returns list of ModuleStatus."""
        self._modules_cb = cb

    def register_tools_callback(self, cb) -> None:
        """Register callback that returns list[str] of active tool names."""
        self._tools_cb = cb

    def register_memory_callback(self, cb) -> None:
        """Register callback that returns MemoryStats."""
        self._memory_cb = cb

    def register_sessions_callback(self, cb) -> None:
        """Register callback that returns int (session count)."""
        self._sessions_cb = cb

    def register_warnings_callback(self, cb) -> None:
        """Register callback that returns list[str] of current warnings."""
        self._warnings_cb = cb

    def snapshot(self) -> AgentStatusSnapshot:
        """Collect and return a complete status snapshot."""
        warnings = []
        
        # Safely call callbacks, catching errors
        def safe_call(cb, default=None):
            if cb is None:
                return default
            try:
                return cb()
            except Exception as e:
                warnings.append(f"Status callback error: {e}")
                return default
        
        model = safe_call(self._model_cb, "")
        provider = safe_call(self._provider_cb, "")
        provider_status = safe_call(self._provider_status_cb, "unknown")
        degraded_mode = safe_call(self._degraded_mode_cb, False)
        channels = safe_call(self._channels_cb, [])
        modules = safe_call(self._modules_cb, [])
        tools = safe_call(self._tools_cb, [])
        memory_stats = safe_call(self._memory_cb, MemoryStats())
        sessions_count = safe_call(self._sessions_cb, 0)
        
        # Auto-detect warnings
        if degraded_mode:
            warnings.append("All providers are down — running in degraded mode")
        if provider_status == "down":
            warnings.append(f"Current provider '{provider}' is down")
        elif provider_status == "degraded":
            warnings.append(f"Current provider '{provider}' is degraded")
        
        # Add callback warnings
        callback_warnings = safe_call(self._warnings_cb, [])
        if callback_warnings:
            warnings.extend(callback_warnings)
        
        # Deduplicate warnings
        seen = set()
        unique_warnings = []
        for w in warnings:
            if w not in seen:
                seen.add(w)
                unique_warnings.append(w)
        
        tools_list = tools if isinstance(tools, list) else []
        
        return AgentStatusSnapshot(
            version=self._version,
            uptime_seconds=time.monotonic() - self._start_time,
            model=model,
            provider=provider,
            provider_status=provider_status,
            degraded_mode=bool(degraded_mode),
            channels=channels if isinstance(channels, list) else [],
            modules=modules if isinstance(modules, list) else [],
            tools_count=len(tools_list),
            active_tools=tools_list,
            memory_stats=memory_stats if isinstance(memory_stats, MemoryStats) else MemoryStats(),
            sessions_count=int(sessions_count) if sessions_count else 0,
            warnings=unique_warnings,
        )

    def health_check(self) -> dict[str, Any]:
        """Lightweight health check for /health endpoint.
        
        Returns minimal info suitable for load balancers and monitoring.
        """
        snap = self.snapshot()
        return {
            "status": "degraded" if snap.degraded_mode else "ok",
            "version": snap.version,
            "model": snap.model,
            "provider": snap.provider,
            "provider_status": snap.provider_status,
            "uptime_seconds": round(snap.uptime_seconds, 1),
        }
