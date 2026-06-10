"""Scheduler module — reminders and recurring tasks that actually fire.

The LLM creates jobs with scheduler__create during conversation; a background
loop checks due jobs every SWEEP_INTERVAL seconds and delivers them through
the originating channel's message.send_* tool. Jobs persist in SQLite under
the module runtime dir, so they survive restarts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger("lumen.module.scheduler")

MODULE_NAME = "x-lumen-scheduler"
SWEEP_INTERVAL_SECONDS = 30
KNOWN_CHANNELS = ("whatsapp", "telegram", "discord", "email")
WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


class SchedulerRuntime:
    def __init__(self, context):
        self.context = context
        self.db_path = Path(context.ensure_runtime_dir()) / "scheduler.db"
        self._db: sqlite3.Connection | None = None
        self._loop_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # ── persistence ────────────────────────────────────────────────

    async def init_store(self):
        def _open():
            # to_thread may use a different worker thread per call; access is
            # serialized with self._lock, so cross-thread use is safe.
            db = sqlite3.connect(str(self.db_path), check_same_thread=False)
            db.row_factory = sqlite3.Row
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_jobs (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    recurrence TEXT DEFAULT '',
                    channel TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    fired_at TEXT,
                    metadata TEXT DEFAULT '{}'
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status_due ON scheduled_jobs(status, due_at)"
            )
            db.commit()
            return db

        self._db = await asyncio.to_thread(_open)

    async def close(self):
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._db:
            await asyncio.to_thread(self._db.close)
            self._db = None

    def _tz(self):
        locale = (self.context.config or {}).get("locale", {})
        tz_name = locale.get("timezone") if isinstance(locale, dict) else None
        if tz_name:
            try:
                return ZoneInfo(str(tz_name))
            except ZoneInfoNotFoundError:
                logger.warning("Invalid scheduler timezone %s; using system local", tz_name)
        return datetime.now().astimezone().tzinfo

    def _now(self) -> datetime:
        return datetime.now(self._tz())

    def _parse_when(self, when: str) -> datetime | None:
        raw = str(when or "").strip().replace("T", " ")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = datetime.strptime(raw, fmt)
                return parsed.replace(tzinfo=self._tz())
            except ValueError:
                continue
        try:
            parsed = datetime.fromisoformat(str(when))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=self._tz())
            return parsed
        except ValueError:
            return None

    # ── job operations (also the LLM tool handlers) ────────────────

    async def create_job(
        self,
        text: str = "",
        when: str = "",
        channel: str = "",
        chat_id: str = "",
        recurrence: str = "",
        **_ignored,
    ) -> dict:
        text = str(text or "").strip()
        if not text:
            return {"error": "text_required"}
        due = self._parse_when(when)
        if due is None:
            return {"error": "invalid_when", "hint": "Use 'YYYY-MM-DD HH:MM' in the user's local time."}
        if due <= self._now():
            return {"error": "when_in_past", "hint": "The reminder time must be in the future."}
        channel = str(channel or "").strip().lower()
        chat_id = str(chat_id or "").strip()
        if not channel or not chat_id:
            resolved = self._resolve_default_target()
            channel = channel or resolved[0]
            chat_id = chat_id or resolved[1]
        if not channel or not chat_id:
            return {
                "error": "no_delivery_target",
                "hint": "No channel/chat_id given and no default could be resolved.",
            }
        recurrence = str(recurrence or "").strip().lower()
        if recurrence and not self._valid_recurrence(recurrence):
            return {"error": "invalid_recurrence", "hint": "Use 'daily' or 'weekly:mon'..'weekly:sun'."}

        job = {
            "id": f"job-{secrets.token_hex(6)}",
            "text": text,
            "due_at": due.isoformat(),
            "recurrence": recurrence,
            "channel": channel,
            "chat_id": chat_id,
            "status": "pending",
            "created_at": self._now().isoformat(),
        }

        def _insert():
            self._db.execute(
                "INSERT INTO scheduled_jobs (id, text, due_at, recurrence, channel, chat_id, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (job["id"], job["text"], job["due_at"], job["recurrence"],
                 job["channel"], job["chat_id"], job["status"], job["created_at"]),
            )
            self._db.commit()

        async with self._lock:
            await asyncio.to_thread(_insert)
        logger.info("Scheduler job created: %s at %s (%s)", job["id"], job["due_at"], recurrence or "once")
        return job

    async def list_jobs(self, **_ignored) -> list[dict]:
        def _query():
            rows = self._db.execute(
                "SELECT id, text, due_at, recurrence, channel, chat_id, status, created_at "
                "FROM scheduled_jobs WHERE status = 'pending' ORDER BY due_at ASC LIMIT 50"
            ).fetchall()
            return [dict(r) for r in rows]

        async with self._lock:
            return await asyncio.to_thread(_query)

    async def cancel_job(self, job_id: str = "", **_ignored) -> dict:
        job_id = str(job_id or "").strip()
        if not job_id:
            return {"error": "job_id_required"}

        def _cancel():
            cursor = self._db.execute(
                "UPDATE scheduled_jobs SET status = 'cancelled' WHERE id = ? AND status = 'pending'",
                (job_id,),
            )
            self._db.commit()
            return cursor.rowcount

        async with self._lock:
            updated = await asyncio.to_thread(_cancel)
        if not updated:
            return {"error": "not_found", "job_id": job_id}
        return {"status": "cancelled", "job_id": job_id}

    # ── firing ─────────────────────────────────────────────────────

    async def fire_due_jobs(self) -> int:
        """Deliver every pending job whose time has come. Returns fired count.

        Delivery failures leave the job pending so the next sweep retries —
        a reminder that never arrives is worse than one a minute late.
        """
        now_iso = self._now().isoformat()

        def _due():
            rows = self._db.execute(
                "SELECT id, text, due_at, recurrence, channel, chat_id "
                "FROM scheduled_jobs WHERE status = 'pending' AND due_at <= ? ORDER BY due_at ASC",
                (now_iso,),
            ).fetchall()
            return [dict(r) for r in rows]

        async with self._lock:
            due_jobs = await asyncio.to_thread(_due)

        fired = 0
        for job in due_jobs:
            tool = f"message.send_{job['channel']}"
            message = f"🔔 Recordatorio: {job['text']}"
            try:
                await self.context.connectors.execute_tool(
                    tool, {"chat_id": job["chat_id"], "text": message}
                )
            except Exception as exc:
                logger.warning("Scheduler delivery failed for %s via %s: %s", job["id"], tool, exc)
                continue

            next_due = self._next_occurrence(job)

            def _mark(job=job, next_due=next_due):
                if next_due is not None:
                    self._db.execute(
                        "UPDATE scheduled_jobs SET due_at = ?, fired_at = ? WHERE id = ?",
                        (next_due.isoformat(), self._now().isoformat(), job["id"]),
                    )
                else:
                    self._db.execute(
                        "UPDATE scheduled_jobs SET status = 'done', fired_at = ? WHERE id = ?",
                        (self._now().isoformat(), job["id"]),
                    )
                self._db.commit()

            async with self._lock:
                await asyncio.to_thread(_mark)
            fired += 1
            logger.info("Scheduler job fired: %s via %s", job["id"], tool)
        return fired

    @staticmethod
    def _valid_recurrence(recurrence: str) -> bool:
        if recurrence == "daily":
            return True
        if recurrence.startswith("weekly:"):
            return recurrence.split(":", 1)[1] in WEEKDAYS
        return False

    def _next_occurrence(self, job: dict) -> datetime | None:
        recurrence = str(job.get("recurrence") or "").strip().lower()
        if not recurrence:
            return None
        base = self._parse_when(job["due_at"]) or self._now()
        now = self._now()
        if recurrence == "daily":
            nxt = base + timedelta(days=1)
            while nxt <= now:
                nxt += timedelta(days=1)
            return nxt
        if recurrence.startswith("weekly:"):
            nxt = base + timedelta(days=7)
            while nxt <= now:
                nxt += timedelta(days=7)
            return nxt
        return None

    def _resolve_default_target(self) -> tuple[str, str]:
        """Fall back to module settings, then to the most recent chat of any
        active communication channel (single-user instances)."""
        channel = str(self.context.resolve_setting("default_channel") or "").strip().lower()
        chat_id = str(self.context.resolve_setting("default_chat_id") or "").strip()
        if channel and chat_id:
            return channel, chat_id
        runtime_root = Path(self.context.runtime_dir).parent
        for candidate in KNOWN_CHANNELS:
            for module_dir in (f"x-lumen-comunicacion-{candidate}", candidate):
                state_path = runtime_root / module_dir / "runtime.json"
                if not state_path.exists():
                    continue
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                last_chat = str(state.get("last_chat_id") or "").strip()
                if last_chat:
                    return channel or candidate, chat_id or last_chat
        return channel, chat_id

    # ── background loop ────────────────────────────────────────────

    async def start(self):
        self._loop_task = asyncio.create_task(self._sweep_loop())

    async def _sweep_loop(self):
        while True:
            try:
                await self.fire_due_jobs()
            except Exception:
                logger.exception("Scheduler sweep failed; retrying next interval")
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


async def activate(context):
    runtime = SchedulerRuntime(context)
    await runtime.init_store()

    context.register_tool(
        "scheduler__create",
        "Schedule a reminder for the user. The message is delivered automatically "
        "at the given time through the conversation channel. Use for any request "
        "like 'remind me to X at Y' or recurring reminders.",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remind, in the user's words."},
                "when": {
                    "type": "string",
                    "description": "Local date-time 'YYYY-MM-DD HH:MM'. Resolve relative times ('in 2 hours', 'tomorrow 10am') to absolute before calling.",
                },
                "recurrence": {
                    "type": "string",
                    "description": "Optional: 'daily' or 'weekly:mon'..'weekly:sun' for recurring reminders.",
                },
                "channel": {"type": "string", "description": "Optional delivery channel (whatsapp, telegram...). Defaults to the active conversation channel."},
                "chat_id": {"type": "string", "description": "Optional chat id. Defaults to the active conversation."},
            },
            "required": ["text", "when"],
        },
        runtime.create_job,
        metadata={"kind": "module", "module": MODULE_NAME},
    )
    context.register_tool(
        "scheduler__list",
        "List the user's pending scheduled reminders.",
        {"type": "object", "properties": {}},
        runtime.list_jobs,
        metadata={"kind": "module", "module": MODULE_NAME},
    )
    context.register_tool(
        "scheduler__cancel",
        "Cancel a scheduled reminder by its job id (see scheduler__list).",
        {
            "type": "object",
            "properties": {"job_id": {"type": "string", "description": "Job id to cancel."}},
            "required": ["job_id"],
        },
        runtime.cancel_job,
        metadata={"kind": "module", "module": MODULE_NAME},
    )

    await runtime.start()
    return runtime


async def deactivate(context, runtime):
    if runtime is not None:
        await runtime.close()
