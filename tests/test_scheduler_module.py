"""Tests for the functional scheduler module (issue #29).

The scheduler must actually FIRE reminders: persist jobs, check due times,
deliver through the originating channel tool, and survive restarts.
"""

import asyncio
import importlib.util
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

CONNECTOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "lumen"
    / "catalog"
    / "modules"
    / "scheduler"
    / "connector.py"
)


def load_connector():
    spec = importlib.util.spec_from_file_location("scheduler_connector_test", CONNECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeConnectors:
    def __init__(self):
        self.registered = {}
        self.executed = []
        self.known_tools = {"message.send_whatsapp"}

    def register_tool(self, name, description, parameters, handler, metadata=None):
        self.registered[name] = {"handler": handler, "parameters": parameters}

    def unregister_tool(self, name):
        self.registered.pop(name, None)

    def _resolve_tool_name(self, name):
        return name

    def has_tool(self, name):
        return name in self.known_tools

    async def execute_tool(self, name, params=None):
        if name not in self.known_tools:
            raise ValueError(f"Unknown tool: {name}")
        self.executed.append({"tool": name, "params": params})
        return {"ok": True}


class FakeContext:
    def __init__(self, runtime_dir: Path, settings: dict | None = None):
        self.name = "scheduler"
        self.runtime_dir = runtime_dir
        self.module_dir = runtime_dir
        self.manifest = {}
        self.config = {"locale": {"timezone": "America/Argentina/Buenos_Aires"}}
        self.connectors = FakeConnectors()
        self.settings = settings or {}
        self.registered_tools = []

    def ensure_runtime_dir(self):
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        return self.runtime_dir

    def resolve_setting(self, key, env_name=None):
        return self.settings.get(key)

    def register_tool(self, name, description, parameters, handler, metadata=None):
        self.connectors.register_tool(name, description, parameters, handler, metadata)
        self.registered_tools.append(name)

    def read_runtime_state(self):
        return {}

    def write_runtime_state(self, payload):
        pass


class SchedulerModuleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.runtime_dir = Path(self.temp_dir.name)
        self.connector = load_connector()
        self.ctx = FakeContext(self.runtime_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _runtime(self):
        return self.connector.SchedulerRuntime(self.ctx)

    def test_activate_registers_tools(self):
        async def run():
            runtime = await self.connector.activate(self.ctx)
            await self.connector.deactivate(self.ctx, runtime)

        asyncio.run(run())
        for tool in ("scheduler__create", "scheduler__list", "scheduler__cancel"):
            assert tool in self.ctx.connectors.registered, tool

    def test_create_and_list_job(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            created = await runtime.create_job(
                text="llamar a Maria",
                when=(datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
                channel="whatsapp",
                chat_id="549111234",
            )
            assert created["id"]
            jobs = await runtime.list_jobs()
            assert len(jobs) == 1
            assert jobs[0]["text"] == "llamar a Maria"
            assert jobs[0]["status"] == "pending"
            await runtime.close()

        asyncio.run(run())

    def test_create_rejects_past_time(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            result = await runtime.create_job(
                text="x",
                when=(datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
                channel="whatsapp",
                chat_id="1",
            )
            assert result.get("error")
            await runtime.close()

        asyncio.run(run())

    def test_cancel_job(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            created = await runtime.create_job(
                text="cita medica",
                when=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d %H:%M"),
                channel="whatsapp",
                chat_id="1",
            )
            result = await runtime.cancel_job(job_id=created["id"])
            assert result["status"] == "cancelled"
            jobs = await runtime.list_jobs()
            assert jobs == []
            await runtime.close()

        asyncio.run(run())

    def test_due_job_fires_through_channel_tool(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            await runtime.create_job(
                text="tomar agua",
                when=(datetime.now() + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                channel="whatsapp",
                chat_id="549111234",
            )
            await asyncio.sleep(1.2)
            fired = await runtime.fire_due_jobs()
            assert fired == 1
            sent = self.ctx.connectors.executed
            assert sent[0]["tool"] == "message.send_whatsapp"
            assert "tomar agua" in sent[0]["params"]["text"]
            assert sent[0]["params"]["chat_id"] == "549111234"
            jobs = await runtime.list_jobs()
            assert jobs == []  # done jobs are not pending
            await runtime.close()

        asyncio.run(run())

    def test_recurring_job_reschedules_after_fire(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            await runtime.create_job(
                text="pastilla",
                when=(datetime.now() + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                channel="whatsapp",
                chat_id="1",
                recurrence="daily",
            )
            await asyncio.sleep(1.2)
            fired = await runtime.fire_due_jobs()
            assert fired == 1
            jobs = await runtime.list_jobs()
            assert len(jobs) == 1  # rescheduled for tomorrow
            next_due = datetime.fromisoformat(jobs[0]["due_at"])
            assert next_due > datetime.now(next_due.tzinfo) + timedelta(hours=20)
            await runtime.close()

        asyncio.run(run())

    def test_jobs_survive_restart(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            await runtime.create_job(
                text="persistente",
                when=(datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d %H:%M"),
                channel="whatsapp",
                chat_id="1",
            )
            await runtime.close()

            # New runtime over the same runtime_dir = container restart
            runtime2 = self._runtime()
            await runtime2.init_store()
            jobs = await runtime2.list_jobs()
            assert len(jobs) == 1
            assert jobs[0]["text"] == "persistente"
            await runtime2.close()

        asyncio.run(run())

    def test_fire_failure_keeps_job_pending(self):
        async def run():
            runtime = self._runtime()
            await runtime.init_store()
            await runtime.create_job(
                text="canal roto",
                when=(datetime.now() + timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S"),
                channel="telegram",  # not in known_tools → delivery fails
                chat_id="1",
            )
            await asyncio.sleep(1.2)
            fired = await runtime.fire_due_jobs()
            assert fired == 0
            jobs = await runtime.list_jobs()
            assert len(jobs) == 1  # still pending, retried next sweep
            await runtime.close()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
