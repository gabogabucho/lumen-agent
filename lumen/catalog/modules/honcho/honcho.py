"""HonchoClient — persistent cross-session memory client.

Wraps the honcho-ai SDK. Provides session management, semantic search,
context fetching, conclusion writing, and memory storage.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Module-level singleton state
_honcho_client: "HonchoClient | None" = None
_honcho_ready: bool = False


@dataclass
class HonchoConfig:
    """Configuration for Honcho SDK client."""

    workspace_id: str
    api_key: str | None = None
    base_url: str | None = None
    session_strategy: str = "per-session"
    recall_mode: str = "hybrid"
    host: str | None = None
    environment: str = "production"

    @staticmethod
    def from_config(config: dict) -> "HonchoConfig":
        """Build HonchoConfig from Lumen config dict.

        Reads flat keys from config["honcho"] for plain config values,
        and config["secrets"]["honcho"]["api_key"] for the API key secret.
        """
        honcho_config = config.get("honcho", {}) or {}
        if not isinstance(honcho_config, dict):
            honcho_config = {}

        workspace_id = str(honcho_config.get("workspace_id", "")).strip()
        base_url = str(honcho_config.get("base_url", "")).strip() or None
        session_strategy = str(honcho_config.get("session_strategy", "per-session")).strip() or "per-session"
        recall_mode = str(honcho_config.get("recall_mode", "hybrid")).strip() or "hybrid"
        host = None

        # Auto-detect environment from host config
        lumen_config = config.get("lumen", {}) or {}
        if isinstance(lumen_config, dict):
            host = str(lumen_config.get("host", "")).strip() or None

        # Detect production vs local from host
        environment = "local" if host and host != "0.0.0.0" else "production"

        # Read API key from secrets store
        api_key: str | None = None
        secrets = config.get("secrets", {}) or {}
        if isinstance(secrets, dict):
            honcho_secrets = secrets.get("honcho", {}) or {}
            if isinstance(honcho_secrets, dict):
                api_key = str(honcho_secrets.get("api_key", "")).strip() or None

        return HonchoConfig(
            workspace_id=workspace_id,
            api_key=api_key,
            base_url=base_url,
            session_strategy=session_strategy,
            recall_mode=recall_mode,
            host=host,
            environment=environment,
        )


class HonchoClient:
    """Client for Honcho persistent memory.

    Wraps the honcho-ai SDK with graceful error handling, timeouts,
    and session management.
    """

    def __init__(self, config: HonchoConfig):
        self.config = config
        self._client = None  # honcho-ai Honcho instance
        self._peer_id: str | None = None
        self._connected_at: float = 0.0

    def connect(self) -> "Honcho | None":
        """Connect to Honcho API.

        Returns the Honcho SDK instance on success, None on failure.
        Raises ImportError if honcho-ai is not installed.
        """
        try:
            from honcho import Honcho
        except ImportError as exc:
            raise ImportError(
                "honcho-ai package not installed. "
                "Run: pip install honcho-ai"
            ) from exc

        # Auto-detect cloud vs self-hosted: if base_url is set/non-empty, route there
        base_url = self.config.base_url if self.config.base_url else None
        api_key = self.config.api_key

        try:
            self._client = Honcho(
                workspace_id=self.config.workspace_id,
                api_key=api_key,
                environment=self.config.environment,
                base_url=base_url,
            )
            self._connected_at = time.time()
            return self._client
        except Exception as e:
            logger.warning("honcho: connect failed — %s", e)
            self._client = None
            return None

    def disconnect(self) -> None:
        """Close the Honcho connection."""
        self._client = None
        self._connected_at = 0.0
        self._peer_id = None

    def is_connected(self) -> bool:
        """Check if client is connected to Honcho."""
        return self._client is not None

    def register_peer(self) -> str | None:
        """Register this Lumen instance as a peer with Honcho.

        Returns the peer_id string, or None if registration fails.
        On failure, sets _honcho_ready = False via module-level state.
        """
        global _honcho_ready
        if not self.is_connected():
            return None

        try:
            result = self._client.register_peer()
            if isinstance(result, dict):
                self._peer_id = result.get("peer_id") or result.get(
                    "id", f"peer_lumen_{int(time.time())}"
                )
                self._connected_at = time.time()
                return self._peer_id
            elif isinstance(result, str):
                self._peer_id = result
                self._connected_at = time.time()
                return self._peer_id
            else:
                self._peer_id = f"peer_lumen_{int(time.time())}"
                self._connected_at = time.time()
                return self._peer_id
        except Exception as e:
            logger.warning("honcho: peer registration failed — %s", e)
            _honcho_ready = False
            return None

    def search(
        self, session_key: str, query: str, max_tokens: int = 800
    ) -> dict:
        """Search memory semantically.

        Args:
            session_key: The session key to search within.
            query: Search query string.
            max_tokens: Maximum tokens in response (default 800, capped at 2000).

        Returns:
            Dict with "result" and "sessions" keys.
        """
        try:
            result = self._call_with_timeout(
                self._client.search, session_key, query, max_tokens=max_tokens
            )

            # Truncate result to max_tokens if needed
            if isinstance(result, dict):
                response_text = result.get("result", "") or ""
                if isinstance(response_text, str) and len(response_text) > max_tokens:
                    result["result"] = response_text[:max_tokens]

            if not isinstance(result, dict):
                return {"result": str(result if result else ""), "sessions": []}

            return {
                "result": result.get("result", ""),
                "sessions": result.get("sessions", result.get("results", [])),
            }
        except Exception as e:
            logger.warning("honcho: search failed — %s", e)
            return {"result": "", "sessions": []}

    def fetch_context(self, session_key: str) -> dict:
        """Fetch full context for a session.

        Returns:
            Dict with context block, summary, card, representation, recent.
        """
        try:
            result = self._call_with_timeout(self._client.fetch_context, session_key)

            if isinstance(result, dict):
                return {
                    "context": result.get("context", result.get("context_block", result.get("block", ""))),
                    "summary": result.get("summary", ""),
                    "card": result.get("card", result.get("peer_card", {})),
                    "representation": result.get("representation", result.get("peer_profile", {})),
                    "recent": result.get("recent", result.get("recent_messages", [])),
                }

            return {"context": "", "summary": "", "card": {}, "representation": {}, "recent": []}
        except Exception as e:
            logger.warning("honcho: fetch_context failed — %s", e)
            return {"context": "", "summary": "", "card": {}, "representation": {}, "recent": []}

    def write_conclusion(self, session_key: str, content: str, peer: str = "user") -> dict:
        """Write a conclusion to Honcho.

        Fire-and-forget semantics — never raises, always returns a dict.

        Args:
            session_key: The session key.
            content: Conclusion text (truncated to 25000 chars).
            peer: Peer identifier (default "user").

        Returns:
            Dict with "success" key and optional "error".
        """
        try:
            # Truncate content to 25000 chars
            if len(content) > 25000:
                content = content[:25000]

            result = self._call_with_timeout(
                self._client.write_conclusion,
                session_key,
                content,
                peer=peer,
            )

            if isinstance(result, dict):
                success = result.get("ok", result.get("success", True))
                if success:
                    return {"success": True}
                return {"success": False, "error": result.get("error", "write failed")}

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def store_memory(self, session: str, content: str, memory_type: str = "fact") -> dict:
        """Store arbitrary memory in Honcho.

        Args:
            session: Session ID (may be empty/None for session-less memory).
            content: Memory content string.
            memory_type: Type of memory (e.g., "fact", "preference", "conclusion").

        Returns:
            Dict with "id" and "success" keys.
        """
        try:
            result = self._call_with_timeout(
                self._client.remember,
                content=content,
                session=session if session else None,
                memory_type=memory_type,
            )

            if isinstance(result, dict):
                mem_id = result.get("id", result.get("memory_id", ""))
                success = result.get("ok", result.get("success", True))
                return {"id": str(mem_id), "success": success}

            return {"id": "", "success": True}
        except Exception as e:
            logger.warning("honcho: store_memory failed — %s", e)
            return {"id": "", "success": False, "error": str(e)}

    def build_context_block(self, session_key: str) -> str:
        """Build a formatted context block for injection.

        Fetches context and wraps it in <context-block>...</context-block> XML.

        Args:
            session_key: The session key to fetch context for.

        Returns:
            Formatted string with <context-block> wrapper, or empty string on failure.
        """
        ctx = self.fetch_context(session_key)
        context = ctx.get("context", "")
        if not context:
            # Build from components
            summary = ctx.get("summary", "")
            card = ctx.get("card", {})
            representation = ctx.get("representation", {})
            recent = ctx.get("recent", [])

            parts = []
            if summary:
                parts.append(f"Summary: {summary}")
            if card:
                card_id = card.get("id", card.get("name", ""))
                parts.append(f"Peer: {card_id}")
            if representation:
                traits = representation.get("traits", [])
                if traits:
                    parts.append(f"Profile: {', '.join(traits)}")
            if recent:
                recent_str = "; ".join(
                    str(r.get("content", str(r)))[:200] for r in recent[:5]
                )
                parts.append(f"Recent: {recent_str}")

            if parts:
                context = "\n".join(parts)

        if context:
            return f"<context-block>\n{context}\n</context-block>"
        return ""

    def inject_context(self, session_key: str, brain) -> None:
        """Inject context into the brain before session thought.

        Fire-and-forget: if the SDK call fails, log warning but don't raise.

        Args:
            session_key: The session key.
            brain: The Lumen brain instance.
        """
        global _honcho_ready
        if not _honcho_ready:
            return

        try:
            context_block = self.build_context_block(session_key)
            if context_block and hasattr(brain, "set_context"):
                brain.set_context(context_block)
            elif context_block:
                # If brain doesn't have set_context, log but don't fail
                logger.info("honcho: context block available but brain has no set_context — skipped")
        except Exception as e:
            logger.warning("honcho: inject_context failed — %s", e)

    def save_conclusions(self, session_key: str, session_data) -> None:
        """Write session conclusions to Honcho after session end.

        Fire-and-forget: if the SDK call fails, log warning but don't raise.

        Args:
            session_key: The session key.
            session_data: Session data (dict with "summary", "interactions", etc.)
        """
        global _honcho_ready
        if not _honcho_ready:
            return

        try:
            # Extract conclusion content from session_data
            if isinstance(session_data, dict):
                summary = session_data.get("summary", "")
                interactions = session_data.get("interactions", [])
                tool_usage = session_data.get("tool_usage", [])
            else:
                summary = str(session_data)
                interactions = []
                tool_usage = []

            # Build conclusion from available data
            parts = []
            if summary:
                parts.append(f"Session summary: {summary}")
            for interaction in interactions[:10]:
                if isinstance(interaction, dict):
                    content = interaction.get("content", "")
                    if content:
                        parts.append(f"Interaction: {content[:500]}")
                elif isinstance(interaction, str) and interaction.strip():
                    parts.append(f"Interaction: {interaction[:500]}")
            for tool in tool_usage[:5]:
                if isinstance(tool, dict):
                    tool_name = tool.get("name", "")
                    result = tool.get("result", "")
                    if tool_name and result:
                        parts.append(f"Tool {tool_name}: {result[:300]}")

            conclusion = "\n".join(parts) if parts else str(session_data)

            # Write the conclusion — fire-and-forget
            self.write_conclusion(session_key, conclusion, peer="lumen")
            logger.info("honcho: conclusion saved for session %s", session_key)
        except Exception as e:
            logger.warning("honcho: conclusion write failed, continuing — %s", e)

    def list_sessions(self) -> dict:
        """Test connectivity by listing sessions.

        Returns:
            Dict with session list or error info.
        """
        try:
            result = self._call_with_timeout(self._client.list_sessions)
            if isinstance(result, dict):
                return result
            return {"sessions": []}
        except Exception as e:
            logger.warning("honcho: list_sessions failed — %s", e)
            return {"error": str(e), "sessions": []}

    def _call_with_timeout(self, method, *args, timeout: float = 30.0, **kwargs) -> dict:
        """Execute an SDK call with timeout.

        Wraps synchronous SDK calls for use in async context with a
        30-second default timeout. Converts TimeoutException → timeout dict.

        Args:
            method: The callable to execute.
            *args: Positional arguments for the method.
            timeout: Timeout in seconds (default 30).
            **kwargs: Keyword arguments for the method.

        Returns:
            The result dict from the SDK call, or timeout error dict.
        """
        try:
            # Execute synchronously in a thread pool
            loop = asyncio.get_running_loop()
            result = loop.run_in_executor(None, lambda: method(*args, **kwargs))
            # run_in_executor returns a coroutine-like, need to await
            # Actually run_in_executor returns a Future; we need asyncio.wait_for
            return loop.run_until_complete(
                asyncio.wait_for(result, timeout=timeout)
            )
        except asyncio.TimeoutError:
            error_dict = {
                "error": "gateway_timeout",
                "message": "Honcho API did not respond in time (30s)",
            }
            logger.warning("honcho: call timed out after %ss", timeout)
            return error_dict
        except Exception as e:
            logger.warning("honcho: SDK call failed — %s", e)
            return {"error": str(e), "ok": False}


def get_honcho_client() -> "HonchoClient | None":
    """Lazy singleton access to HonchoClient.

    Creates HonchoClient from config on first call. Returns None if
    configuration is incomplete or connection fails.

    Sets _honcho_ready = True if successfully connected.
    """
    global _honcho_client, _honcho_ready

    if _honcho_client is not None and _honcho_client.is_connected():
        return _honcho_client

    config = _load_honcho_config()
    if not config or not config.workspace_id:
        return None

    try:
        client = HonchoClient(config)
        sdk = client.connect()
        if sdk is not None:
            _honcho_client = client
            _honcho_ready = True
            return client
    except ImportError as e:
        logger.warning("honcho: SDK not available — %s", e)
        _honcho_ready = False
    except Exception as e:
        logger.warning("honcho: client initialization failed — %s", e)
        _honcho_ready = False

    return None


def reset_honcho_client() -> None:
    """Explicit teardown — called on reconfigure or module uninstall."""
    global _honcho_client, _honcho_ready

    if _honcho_client is not None:
        _honcho_client.disconnect()
    _honcho_client = None
    _honcho_ready = False


def _load_honcho_config() -> HonchoConfig | None:
    """Load and validate honcho configuration."""
    try:
        import os
        import yaml

        # Try to load from Lumen's config path
        lumen_dir = Path.home() / ".lumen"
        config_path = lumen_dir / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if isinstance(config, dict):
                return HonchoConfig.from_config(config)
    except Exception:
        pass
    return None


# Import Path at module level for config loading
from pathlib import Path
