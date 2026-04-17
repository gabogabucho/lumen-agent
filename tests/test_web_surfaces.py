import json
import tempfile
import unittest
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from lumen.channels import web
from lumen.core.catalog import Catalog
from lumen.core.connectors import Connector, ConnectorRegistry
from lumen.core.discovery import discover_all
from lumen.core.marketplace import Marketplace
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry


async def _noop(**_kwargs):
    return {"ok": True}


class MemoryStub:
    def __init__(self, messages=None, *, should_fail=False):
        self.messages = list(messages or [])
        self.should_fail = should_fail
        self.calls = []

    async def load_conversation(self, session_id: str, limit: int = 50):
        self.calls.append((session_id, limit))
        if self.should_fail:
            raise RuntimeError("memory unavailable")
        return list(self.messages)


class BrainStub:
    def __init__(self, *, registry=None, connectors=None, flows=None, memory=None):
        self.registry = registry or Registry()
        self.connectors = connectors or ConnectorRegistry()
        self.flows = list(flows or [])
        self.memory = memory or MemoryStub()
        self.last_think = None

    async def think(self, user_text, session):
        self.last_think = {
            "user_text": user_text,
            "session_id": session.session_id,
            "history": list(session.history),
        }
        return {"message": f"Echo: {user_text}"}


class StubMarketplace(Marketplace):
    def __init__(self, *args, payloads=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.payloads = payloads or {}

    def _fetch_json(self, url: str):
        return self.payloads[url]


class WebSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.original_brain = web._brain
        self.original_config = web._config
        self.original_locale = web._locale
        self.original_sessions = dict(web.session_manager._sessions)
        web._brain = None
        web._config = {}
        web._locale = {}
        web.session_manager._sessions.clear()
        self.client = TestClient(web.app)

    def tearDown(self):
        web._brain = self.original_brain
        web._config = self.original_config
        web._locale = self.original_locale
        web.session_manager._sessions.clear()
        web.session_manager._sessions.update(self.original_sessions)

    def test_api_status_reports_truthful_payload_shape_and_counts(self):
        connectors = ConnectorRegistry()
        task = Connector("task", "Tasks", ["create"])
        task.register_handler("create", _noop)
        connectors.register(task)
        registry = Registry()
        registry.register(
            Capability(
                kind=CapabilityKind.CHANNEL,
                name="web",
                description="Web dashboard",
                status=CapabilityStatus.READY,
            )
        )
        registry.register(
            Capability(
                kind=CapabilityKind.SKILL,
                name="faq",
                description="Answers FAQs",
                status=CapabilityStatus.READY,
                provides=["faq.answer"],
                min_capability="tier-1",
            )
        )
        registry.register(
            Capability(
                kind=CapabilityKind.MCP,
                name="docs",
                description="Docs MCP",
                status=CapabilityStatus.ERROR,
                provides=["docs.search"],
                min_capability="tier-2",
            )
        )
        web._brain = BrainStub(
            registry=registry,
            connectors=connectors,
            flows=[
                {
                    "intent": "book_demo",
                    "triggers": ["book", "demo"],
                    "slots": {"email": {"required": True}, "date": {"required": True}},
                }
            ],
        )
        web._config = {"model": "demo-model", "language": "es"}

        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "active")
        self.assertEqual(payload["model"], "demo-model")
        self.assertEqual(payload["language"], "es")
        self.assertEqual(len(payload["capabilities"]), 3)
        self.assertEqual(payload["ready"], 2)
        self.assertEqual(payload["gaps"], 1)
        self.assertEqual(payload["summary"]["channel"]["ready"], 1)
        self.assertEqual(payload["summary"]["skill"]["ready"], 1)
        self.assertEqual(payload["summary"]["mcp"]["error"], 1)
        self.assertEqual(payload["flows"][0]["intent"], "book_demo")
        self.assertEqual(payload["flows"][0]["slots"], ["email", "date"])
        self.assertEqual(payload["capabilities"][2]["min_capability"], "tier-2")

    def test_api_status_defaults_to_not_configured_when_brain_missing(self):
        response = self.client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "not_configured",
                "version": "0.1.0",
                "model": "not configured",
                "language": "en",
                "capabilities": [],
                "summary": {},
                "flows": [],
                "ready": 0,
                "gaps": 0,
            },
        )

    def test_api_history_returns_persisted_messages(self):
        web._brain = BrainStub(
            memory=MemoryStub(
                [
                    {"role": "user", "content": "hola"},
                    {"role": "assistant", "content": "buenas"},
                ]
            )
        )

        response = self.client.get("/api/history/session-123")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["messages"],
            [
                {"role": "user", "content": "hola"},
                {"role": "assistant", "content": "buenas"},
            ],
        )
        self.assertEqual(web._brain.memory.calls, [("session-123", 50)])

    def test_websocket_hydrates_session_sends_response_and_cleans_up(self):
        web._brain = BrainStub(
            memory=MemoryStub(
                [
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                ]
            )
        )

        with self.client.websocket_connect("/ws/session-abc") as websocket:
            websocket.send_text(json.dumps({"content": "new question"}))

            typing_on = json.loads(websocket.receive_text())
            assistant = json.loads(websocket.receive_text())
            typing_off = json.loads(websocket.receive_text())

            self.assertEqual(typing_on, {"type": "typing", "status": True})
            self.assertEqual(assistant["type"], "message")
            self.assertEqual(assistant["role"], "assistant")
            self.assertEqual(assistant["content"], "Echo: new question")
            self.assertEqual(typing_off, {"type": "typing", "status": False})
            self.assertEqual(
                web._brain.last_think["history"],
                [
                    {"role": "user", "content": "previous question"},
                    {"role": "assistant", "content": "previous answer"},
                ],
            )
            self.assertIsNotNone(web.session_manager.get("session-abc"))

        self.assertEqual(web._brain.memory.calls, [("session-abc", 50)])
        self.assertIsNone(web.session_manager.get("session-abc"))


class CapabilityPropagationTests(unittest.TestCase):
    def test_discovery_catalog_and_marketplace_preserve_min_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            (pkg_dir / "modules" / "tiered-module").mkdir(parents=True)
            (pkg_dir / "modules" / "tiered-module" / "module.yaml").write_text(
                yaml.dump(
                    {
                        "name": "tiered-module",
                        "display_name": "Tiered Module",
                        "description": "Installed runtime module",
                        "version": "1.0.0",
                        "min_capability": "tier-2",
                        "tags": ["personality"],
                    }
                ),
                encoding="utf-8",
            )
            (pkg_dir / "modules" / "tiered-module" / "SKILL.md").write_text(
                "---\nname: tiered-module\ndescription: runtime skill\n---\n",
                encoding="utf-8",
            )
            catalog_path = pkg_dir / "index.yaml"
            catalog_path.write_text(
                yaml.dump(
                    {
                        "modules": [
                            {
                                "name": "catalog-tiered",
                                "display_name": "Catalog Tiered",
                                "description": "Catalog entry",
                                "min_capability": "tier-2",
                                "path": "kits/catalog-tiered",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            connectors = ConnectorRegistry()
            task = Connector("task", "Tasks", ["create"])
            task.register_handler("create", _noop)
            connectors.register(task)
            registry = discover_all(
                Registry(), pkg_dir, connectors, active_channels=["web"]
            )
            catalog = Catalog(catalog_path)
            marketplace = StubMarketplace(
                catalog=catalog,
                registry=registry,
                connectors=connectors,
                config={
                    "marketplace": {
                        "feeds": [
                            {
                                "name": "OpenClaw",
                                "url": "https://example.test/feed.json",
                            }
                        ]
                    }
                },
                payloads={
                    "https://example.test/feed.json": {
                        "skills": [
                            {
                                "name": "remote-planner",
                                "description": "Remote planning skill",
                                "connectors_required": ["task"],
                                "min_capability": "tier-3",
                            }
                        ],
                        "mcps": [
                            {
                                "name": "remote-docs",
                                "description": "Remote docs mcp",
                                "tools": ["docs.search"],
                                "min_capability": "tier-2",
                            }
                        ],
                    }
                },
            )

            catalog_items = catalog.list_all()
            snapshot = marketplace.snapshot()

        self.assertEqual(
            registry.get(CapabilityKind.MODULE, "tiered-module").metadata[
                "min_capability"
            ],
            "tier-2",
        )
        self.assertEqual(catalog_items[0]["min_capability"], "tier-2")
        self.assertEqual(snapshot["skills"]["available"][0]["min_capability"], "tier-3")
        self.assertEqual(snapshot["mcps"]["available"][0]["min_capability"], "tier-2")
        self.assertEqual(
            snapshot["kits_lumen"]["installed"][0]["min_capability"],
            "tier-2",
        )


if __name__ == "__main__":
    unittest.main()
