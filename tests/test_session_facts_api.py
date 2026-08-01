"""Tests for the session facts HTTP surface.

`save_session_fact` and its siblings already existed on the memory layer; what
was missing was a door from outside the process. These cover that door.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from lumen.channels import web
from lumen.core.connectors import ConnectorRegistry
from lumen.core.registry import Registry


class FactsMemoryStub:
    """Just enough memory to see what the endpoint hands down."""

    def __init__(self, fail_with=None):
        self._db = object()          # already initialised
        self.saved = []
        self.listed = []
        self._fail_with = fail_with
        self._next_id = 1

    async def init(self):
        self._db = object()

    async def save_session_fact(self, session_id, fact, category="general",
                                importance=0.5):
        if self._fail_with:
            raise self._fail_with
        self.saved.append({
            "session_id": session_id, "fact": fact,
            "category": category, "importance": importance,
        })
        fact_id, self._next_id = self._next_id, self._next_id + 1
        return fact_id

    async def list_session_facts(self, query="", limit=10, session_prefix=None,
                                 session_prefixes=None):
        self.listed.append({"query": query, "limit": limit,
                            "session_prefix": session_prefix})
        return [{"id": 1, "session_id": "s1", "fact": "likes the balcony plants",
                 "category": "general", "importance": 0.9, "created_at": 0.0}]


class BrainStub:
    def __init__(self, memory=None):
        self.registry = Registry()
        self.connectors = ConnectorRegistry()
        self.flows = []
        self.memory = memory or FactsMemoryStub()


class SessionFactsAPITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_lumen_dir = web.LUMEN_DIR
        self.original_config_path = web.CONFIG_PATH
        self.original_brain = web._brain
        self.original_config = web._config
        web.LUMEN_DIR = Path(self.temp_dir.name)
        web.CONFIG_PATH = web.LUMEN_DIR / "config.yaml"
        web._brain = None
        web._config = {}
        web.session_manager._sessions.clear()
        self.client = TestClient(web.app)
        self.memory = FactsMemoryStub()
        os.environ["LUMEN_API_KEY"] = "test-key-123"
        self.auth = {"Authorization": "Bearer test-key-123"}

    def tearDown(self):
        self.temp_dir.cleanup()
        web.LUMEN_DIR = self.original_lumen_dir
        web.CONFIG_PATH = self.original_config_path
        web._brain = self.original_brain
        web._config = self.original_config
        web.session_manager._sessions.clear()
        os.environ.pop("LUMEN_API_KEY", None)

    # --- Auth: same Bearer as /api/chat ---

    def test_create_rejects_no_auth(self):
        response = self.client.post("/api/session/facts",
                                    json={"session_id": "s1", "fact": "x"})
        assert response.status_code == 401

    def test_create_rejects_wrong_key(self):
        response = self.client.post(
            "/api/session/facts",
            json={"session_id": "s1", "fact": "x"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert response.status_code == 401

    def test_list_rejects_no_auth(self):
        assert self.client.get("/api/session/facts").status_code == 401

    # --- Writing ---

    def test_seeds_a_fact_and_returns_its_id(self):
        web._brain = BrainStub(self.memory)
        response = self.client.post(
            "/api/session/facts",
            json={"session_id": "s1", "fact": "Lives alone since her husband died",
                  "category": "family", "importance": 0.9},
            headers=self.auth,
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"] == 1
        assert self.memory.saved == [{
            "session_id": "s1",
            "fact": "Lives alone since her husband died",
            "category": "family",
            "importance": 0.9,
        }]

    def test_category_and_importance_have_defaults(self):
        web._brain = BrainStub(self.memory)
        self.client.post("/api/session/facts",
                         json={"session_id": "s1", "fact": "x"}, headers=self.auth)
        assert self.memory.saved[0]["category"] == "general"
        assert self.memory.saved[0]["importance"] == 0.5

    def test_session_id_is_required(self):
        web._brain = BrainStub(self.memory)
        r = self.client.post("/api/session/facts", json={"fact": "x"},
                             headers=self.auth)
        assert r.status_code == 400
        assert "session_id" in r.json()["error"]

    def test_fact_is_required(self):
        web._brain = BrainStub(self.memory)
        r = self.client.post("/api/session/facts", json={"session_id": "s1"},
                             headers=self.auth)
        assert r.status_code == 400
        assert "fact" in r.json()["error"]

    def test_blank_fact_is_not_a_fact(self):
        """A whitespace-only fact would occupy a top-N slot and say nothing."""
        web._brain = BrainStub(self.memory)
        r = self.client.post("/api/session/facts",
                             json={"session_id": "s1", "fact": "   "},
                             headers=self.auth)
        assert r.status_code == 400
        assert self.memory.saved == []

    def test_importance_outside_zero_to_one_is_rejected(self):
        """Importance orders what continuity injects every turn. Out-of-range
        values silently outrank everything else forever."""
        web._brain = BrainStub(self.memory)
        for bad in (1.5, -0.1):
            r = self.client.post(
                "/api/session/facts",
                json={"session_id": "s1", "fact": "x", "importance": bad},
                headers=self.auth,
            )
            assert r.status_code == 400, bad
        assert self.memory.saved == []

    def test_non_numeric_importance_is_rejected(self):
        web._brain = BrainStub(self.memory)
        r = self.client.post(
            "/api/session/facts",
            json={"session_id": "s1", "fact": "x", "importance": "very"},
            headers=self.auth,
        )
        assert r.status_code == 400

    def test_invalid_json_is_rejected(self):
        web._brain = BrainStub(self.memory)
        r = self.client.post("/api/session/facts", content=b"not json",
                             headers=self.auth)
        assert r.status_code == 400

    def test_without_a_brain_it_says_so(self):
        web._brain = None
        r = self.client.post("/api/session/facts",
                             json={"session_id": "s1", "fact": "x"},
                             headers=self.auth)
        assert r.status_code == 503

    def test_a_memory_failure_is_reported_not_swallowed(self):
        web._brain = BrainStub(FactsMemoryStub(fail_with=RuntimeError("disk full")))
        r = self.client.post("/api/session/facts",
                             json={"session_id": "s1", "fact": "x"},
                             headers=self.auth)
        assert r.status_code == 500
        assert "disk full" in r.json()["error"]

    def test_memory_is_initialised_if_it_was_not(self):
        memory = FactsMemoryStub()
        memory._db = None
        web._brain = BrainStub(memory)
        r = self.client.post("/api/session/facts",
                             json={"session_id": "s1", "fact": "x"},
                             headers=self.auth)
        assert r.status_code == 200, r.text

    # --- Reading ---

    def test_lists_facts(self):
        web._brain = BrainStub(self.memory)
        r = self.client.get("/api/session/facts?session_id=s1", headers=self.auth)
        assert r.status_code == 200
        assert r.json()["facts"][0]["fact"] == "likes the balcony plants"
        assert self.memory.listed[0]["session_prefix"] == "s1"

    def test_listing_without_a_session_does_not_filter(self):
        web._brain = BrainStub(self.memory)
        self.client.get("/api/session/facts", headers=self.auth)
        assert self.memory.listed[0]["session_prefix"] is None

    def test_limit_is_clamped(self):
        """An unbounded limit turns a seeding endpoint into a full table dump."""
        web._brain = BrainStub(self.memory)
        self.client.get("/api/session/facts?limit=100000", headers=self.auth)
        assert self.memory.listed[0]["limit"] == 100

    def test_non_integer_limit_is_rejected(self):
        web._brain = BrainStub(self.memory)
        r = self.client.get("/api/session/facts?limit=many", headers=self.auth)
        assert r.status_code == 400


if __name__ == "__main__":
    unittest.main()


class SessionFactsAgainstRealMemoryTests(unittest.TestCase):
    """The endpoint against the real Memory, not a stub.

    The stub above proves the HTTP shape; it cannot prove the call actually
    lands in SQLite. A seeded fact that 200s and stores nothing looks exactly
    like success from the caller's side.
    """

    def setUp(self):
        from lumen.core.memory import Memory

        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_lumen_dir = web.LUMEN_DIR
        self.original_config_path = web.CONFIG_PATH
        self.original_brain = web._brain
        self.original_config = web._config
        web.LUMEN_DIR = Path(self.temp_dir.name)
        web.CONFIG_PATH = web.LUMEN_DIR / "config.yaml"
        web._config = {}
        web.session_manager._sessions.clear()

        self.memory = Memory(db_path=Path(self.temp_dir.name) / "memory.db")
        web._brain = BrainStub(self.memory)
        self.client = TestClient(web.app)
        os.environ["LUMEN_API_KEY"] = "test-key-123"
        self.auth = {"Authorization": "Bearer test-key-123"}

    def tearDown(self):
        # Close before cleanup: on Windows an open SQLite handle makes the temp
        # directory undeletable, and the test fails in teardown having passed.
        asyncio.run(self.memory.close())
        self.temp_dir.cleanup()
        web.LUMEN_DIR = self.original_lumen_dir
        web.CONFIG_PATH = self.original_config_path
        web._brain = self.original_brain
        web._config = self.original_config
        web.session_manager._sessions.clear()
        os.environ.pop("LUMEN_API_KEY", None)

    def test_a_seeded_fact_survives_to_sqlite_and_reads_back(self):
        posted = self.client.post(
            "/api/session/facts",
            json={"session_id": "elder-42",
                  "fact": "Do not talk to her like she is old, she notices",
                  "category": "style", "importance": 0.95},
            headers=self.auth,
        )
        assert posted.status_code == 200, posted.text

        got = self.client.get("/api/session/facts?session_id=elder-42",
                              headers=self.auth)
        assert got.status_code == 200, got.text
        facts = got.json()["facts"]
        assert len(facts) == 1, facts
        assert facts[0]["fact"] == "Do not talk to her like she is old, she notices"
        assert facts[0]["category"] == "style"
        assert facts[0]["importance"] == 0.95

    def test_importance_orders_what_continuity_would_inject(self):
        """The reason importance is validated at all: it decides which facts
        stay in the top-N injected every turn."""
        for fact, importance in (("minor detail", 0.1), ("takes her pill at 6", 0.9)):
            self.client.post(
                "/api/session/facts",
                json={"session_id": "elder-42", "fact": fact,
                      "importance": importance},
                headers=self.auth,
            )

        facts = self.client.get("/api/session/facts?session_id=elder-42&limit=1",
                                headers=self.auth).json()["facts"]
        assert facts[0]["fact"] == "takes her pill at 6"

    def test_facts_of_one_session_do_not_leak_into_another(self):
        self.client.post("/api/session/facts",
                         json={"session_id": "elder-42", "fact": "hers"},
                         headers=self.auth)
        facts = self.client.get("/api/session/facts?session_id=elder-99",
                                headers=self.auth).json()["facts"]
        assert facts == []
