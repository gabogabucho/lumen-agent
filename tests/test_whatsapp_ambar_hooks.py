import asyncio
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from lumen.channels import web
from lumen.core.module_runtime import (
    HookResult,
    InboundHookEvent,
    ModuleRuntimeContext,
    normalize_hook_result,
)


CONNECTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "lumen"
    / "catalog"
    / "modules"
    / "x-lumen-comunicacion-whatsapp"
    / "connector.py"
)


def load_connector():
    spec = importlib.util.spec_from_file_location("whatsapp_connector_test", CONNECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeContext:
    def __init__(self, runtime_dir: Path, settings: dict | None = None):
        self.runtime_dir = runtime_dir
        self.settings = settings or {}
        self.memory = None
        self.sent_events = []
        self.hooks = []

    def resolve_setting(self, key, env_name=None):
        return self.settings.get(key) or self.settings.get(env_name)

    def read_runtime_state(self):
        return {}

    def write_runtime_state(self, payload):
        self.last_state = payload

    def register_inbound_hook(self, channel, handler, *, timeout_seconds=None):
        self.hooks.append((channel, handler, timeout_seconds))
        return lambda: None

    async def invoke_inbound_hooks(self, event, *, timeout_seconds=None):
        for channel, handler, hook_timeout in self.hooks:
            if channel != event.channel:
                continue
            result = handler(event)
            if asyncio.iscoroutine(result):
                result = await result
            normalized = normalize_hook_result(result)
            if normalized.action == "consume":
                return normalized
        return HookResult(action="pass_through")


class HookContractTests(unittest.TestCase):
    def test_hook_result_normalization_defaults_to_pass_through(self):
        assert normalize_hook_result(None).action == "pass_through"
        assert normalize_hook_result({"action": "unknown"}).action == "pass_through"
        assert normalize_hook_result("bad").action == "pass_through"

    def test_context_hook_timeout_and_exception_pass_through(self):
        ctx = ModuleRuntimeContext(
            name="mod",
            module_dir=Path("."),
            runtime_dir=Path("."),
            manifest={},
            config={},
        )

        async def slow(_event):
            await asyncio.sleep(0.05)
            return {"action": "consume"}

        def boom(_event):
            raise RuntimeError("bad hook")

        ctx.register_inbound_hook("whatsapp", slow, timeout_seconds=0.001)
        ctx.register_inbound_hook("whatsapp", boom)
        event = InboundHookEvent(
            channel="whatsapp",
            sender_id="u1",
            text="hello",
            metadata={},
        )

        async def run():
            result = await ctx.invoke_inbound_hooks(event, timeout_seconds=0.001)
            assert result.action == "pass_through"

        asyncio.run(run())


class WhatsAppConnectorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name)
        self.connector = load_connector()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_text_message_pass_through_writes_metadata(self):
        ctx = FakeContext(self.runtime_dir)
        runtime = self.connector.WhatsAppRuntime(ctx)

        async def run():
            await runtime._handle_message({"chatId": "c1", "senderId": "s1", "body": "hola", "id": "m1"})

        asyncio.run(run())
        rows = (self.runtime_dir / "inbox.jsonl").read_text(encoding="utf-8").splitlines()
        entry = json.loads(rows[0])
        assert entry["chat_id"] == "c1"
        assert entry["text"] == "hola"
        assert entry["metadata"]["message_id"] == "m1"
        assert entry["message_type"] == "text"
        assert entry["source"] == "whatsapp"

    def test_consume_hook_sends_response_and_skips_inbox(self):
        ctx = FakeContext(self.runtime_dir)
        ctx.register_inbound_hook(
            "whatsapp",
            lambda event: {"action": "consume", "response": "handled", "metadata": {"hook": "ambar"}},
        )
        runtime = self.connector.WhatsAppRuntime(ctx)

        async def run():
            with patch.object(runtime, "send", new=AsyncMock()) as send_mock:
                await runtime._handle_message({"chatId": "c1", "senderId": "s1", "body": "hola"})
                send_mock.assert_awaited_once_with("c1", "handled")

        asyncio.run(run())
        assert not (self.runtime_dir / "inbox.jsonl").exists()

    def test_audio_transcription_routes_transcript_with_audio_metadata(self):
        ctx = FakeContext(self.runtime_dir, {"stt_model": "whisper"})
        runtime = self.connector.WhatsAppRuntime(ctx)

        async def run():
            with patch.object(self.connector, "_download_audio", AsyncMock(return_value="voice.ogg")):
                with patch.object(self.connector, "_transcribe_audio", AsyncMock(return_value="transcript")):
                    await runtime._handle_message(
                        {"chatId": "c1", "senderId": "s1", "mediaType": "ptt", "mediaUrls": ["http://audio"]}
                    )

        asyncio.run(run())
        entry = json.loads((self.runtime_dir / "inbox.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert entry["text"] == "transcript"
        assert entry["message_type"] == "audio"
        assert entry["metadata"]["type"] == "audio_transcription"
        assert entry["metadata"]["audio"]["media_type"] == "ptt"

    def test_tts_reply_modes_and_text_fallback(self):
        async def run():
            ctx = FakeContext(self.runtime_dir, {"reply_mode": "both", "tts_model": "tts"})
            runtime = self.connector.WhatsAppRuntime(ctx)
            with patch.object(self.connector, "_synthesize_speech", AsyncMock(return_value="a.ogg")):
                with patch.object(self.connector, "_bridge_post", return_value={}) as post:
                    await runtime.send("c1", "hello")
            paths = [call.args[1] for call in post.call_args_list]
            assert "/send" in paths
            assert "/send-media" in paths

            ctx = FakeContext(self.runtime_dir, {"reply_mode": "voice", "tts_model": "tts"})
            runtime = self.connector.WhatsAppRuntime(ctx)
            with patch.object(self.connector, "_synthesize_speech", AsyncMock(return_value=None)):
                with patch.object(self.connector, "_bridge_post", return_value={}) as post:
                    await runtime.send("c1", "hello")
            assert [call.args[1] for call in post.call_args_list].count("/send") == 1

        asyncio.run(run())


class WhatsAppSendApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lumen_dir = Path(self.temp_dir.name)
        self.original_lumen_dir = web.LUMEN_DIR
        self.original_config_path = web.CONFIG_PATH
        self.original_brain = web._brain
        self.original_config = web._config
        self.original_access_mode = web._access_mode
        web.LUMEN_DIR = self.lumen_dir
        web.CONFIG_PATH = self.lumen_dir / "config.yaml"
        web._config = {"model": "test", "api": {"rest_key": "secret"}}
        web.CONFIG_PATH.write_text(json.dumps(web._config), encoding="utf-8")

    def tearDown(self):
        web.LUMEN_DIR = self.original_lumen_dir
        web.CONFIG_PATH = self.original_config_path
        web._brain = self.original_brain
        web._config = self.original_config
        web._access_mode = self.original_access_mode
        self.temp_dir.cleanup()

    def test_send_api_auth_validation_unavailable_and_dispatch(self):
        web.configure_access_mode("serve")
        web._brain = MagicMock()
        web._brain.module_manager = MagicMock()
        web._brain.module_manager._loaded = {}
        client = TestClient(web.app)

        assert client.post("/api/whatsapp/send", json={"to": "c", "message": "m"}).status_code == 401
        response = client.post("/api/whatsapp/send", json={"to": "", "message": "m"}, headers={"Authorization": "Bearer secret"})
        assert response.status_code == 400
        response = client.post("/api/whatsapp/send", json={"to": "c", "message": "m"}, headers={"Authorization": "Bearer secret"})
        assert response.status_code == 503

        runtime = MagicMock()
        runtime.send_message = AsyncMock(return_value={"status": "ok", "chat_id": "c", "message_id": "mid"})
        web._brain.module_manager._loaded = {"x-lumen-comunicacion-whatsapp": MagicMock(state=runtime)}
        response = client.post("/api/whatsapp/send", json={"to": "c", "message": "m"}, headers={"Authorization": "Bearer secret"})
        assert response.status_code == 200
        assert response.json()["message_id"] == "mid"
        runtime.send_message.assert_awaited_once()


class WhatsAppMemoryDuplicationTests(unittest.TestCase):
    """Issue #22: inbound messages were persisted twice — once here as
    'whatsapp_message' and once by Brain.think as 'conversation:{session}'.
    The brain owns conversation persistence; the connector must not save."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name)
        self.connector = load_connector()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_inbound_message_is_not_saved_to_memory_by_connector(self):
        ctx = FakeContext(self.runtime_dir)
        ctx.memory = MagicMock()
        ctx.memory._db = object()
        ctx.memory.remember = AsyncMock()
        runtime = self.connector.WhatsAppRuntime(ctx)

        async def run():
            await runtime._handle_message(
                {"chatId": "c1", "senderId": "s1", "body": "Hola! Muy bien", "id": "m1"}
            )

        asyncio.run(run())
        ctx.memory.remember.assert_not_awaited()
        # The jsonl inbox archive still captures the message
        rows = (self.runtime_dir / "inbox.jsonl").read_text(encoding="utf-8").splitlines()
        assert json.loads(rows[0])["text"] == "Hola! Muy bien"
