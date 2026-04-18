import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.installer import Installer
from lumen.core.memory import Memory
from lumen.core.module_runtime import ModuleRuntimeManager


REPO_ROOT = Path(__file__).resolve().parents[1]
TELEGRAM_KIT_DIR = (
    REPO_ROOT / "lumen" / "catalog" / "modules" / "x-lumen-comunicacion-telegram"
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TelegramModuleInstallTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.pkg_dir = self.tmp_path / "pkg"
        self.lumen_dir = self.tmp_path / "lumen-home"
        (self.pkg_dir / "catalog" / "kits").mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            TELEGRAM_KIT_DIR,
            self.pkg_dir / "catalog" / "modules" / "x-lumen-comunicacion-telegram",
        )
        (self.pkg_dir / "connectors").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            REPO_ROOT / "lumen" / "connectors" / "built-in.yaml",
            self.pkg_dir / "connectors" / "built-in.yaml",
        )
        (self.pkg_dir / "catalog" / "index.yaml").write_text(
            yaml.dump(
                {
                    "modules": [
                        {
                            "name": "x-lumen-comunicacion-telegram",
                            "display_name": "Telegram",
                            "description": "Telegram channel",
                            "version": "1.0.0",
                            "author": "Lumen Team",
                            "price": "free",
                            "tags": ["x-lumen", "comunicacion", "telegram"],
                            "provides": ["message.send_telegram"],
                            "path": "modules/x-lumen-comunicacion-telegram",
                        }
                    ]
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _installer(self) -> Installer:
        return Installer(
            pkg_dir=self.pkg_dir,
            connectors=ConnectorRegistry(),
            memory=None,
            catalog=Catalog(self.pkg_dir / "catalog" / "index.yaml"),
            lumen_dir=self.lumen_dir,
        )

    def test_install_from_catalog_creates_runtime_files(self):
        result = self._installer().install_from_catalog("x-lumen-comunicacion-telegram")

        self.assertEqual(result["status"], "installed")
        runtime_dir = self.lumen_dir / "modules" / "x-lumen-comunicacion-telegram"
        self.assertTrue((runtime_dir / "config.yaml").exists())
        self.assertTrue((runtime_dir / "runtime.json").exists())

    def test_uninstall_removes_runtime_files(self):
        installer = self._installer()
        installer.install_from_catalog("x-lumen-comunicacion-telegram")

        runtime_dir = self.lumen_dir / "modules" / "x-lumen-comunicacion-telegram"
        (runtime_dir / "inbox.jsonl").write_text("{}\n", encoding="utf-8")

        result = installer.uninstall("x-lumen-comunicacion-telegram")

        self.assertEqual(result["status"], "uninstalled")
        self.assertFalse(runtime_dir.exists())
        self.assertFalse(
            (self.pkg_dir / "modules" / "x-lumen-comunicacion-telegram").exists()
        )


class TelegramModuleRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)
        self.pkg_dir = self.tmp_path / "pkg"
        self.lumen_dir = self.tmp_path / "lumen-home"
        module_dir = self.pkg_dir / "modules" / "x-lumen-comunicacion-telegram"
        module_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(TELEGRAM_KIT_DIR, module_dir)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_sync_registers_send_tool_and_send_message_works(self):
        connectors = ConnectorRegistry()
        memory = Memory(self.lumen_dir / "memory.db")
        requests_seen: list[tuple[str, dict | None]] = []

        def fake_urlopen(req, timeout=0):
            payload = None
            if req.data:
                payload = json.loads(req.data.decode("utf-8"))
            requests_seen.append((req.full_url, payload))

            if req.full_url.endswith("/deleteWebhook"):
                return _FakeResponse({"ok": True, "result": True})
            if req.full_url.endswith("/getUpdates"):
                return _FakeResponse({"ok": True, "result": []})
            if req.full_url.endswith("/sendMessage"):
                return _FakeResponse({"ok": True, "result": {"message_id": 99}})
            raise AssertionError(f"Unexpected URL: {req.full_url}")

        manager = ModuleRuntimeManager(
            pkg_dir=self.pkg_dir,
            lumen_dir=self.lumen_dir,
            config={},
            connectors=connectors,
            memory=memory,
        )

        with patch.dict(
            "os.environ", {"TELEGRAM_BOT_TOKEN": "test-token"}, clear=False
        ):
            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                await manager.sync()
                self.assertTrue(connectors.has_tool("message.send_telegram"))

                result = await connectors.execute_tool(
                    "message.send_telegram",
                    {"chat_id": "12345", "text": "hola"},
                )

                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["message_id"], 99)
                self.assertTrue(
                    any(url.endswith("/sendMessage") for url, _ in requests_seen)
                )

                await manager.close()

        self.assertFalse(connectors.has_tool("message.send_telegram"))
