"""Persistent memory — SQLite with FTS5 full-text search."""

import json
import time
from pathlib import Path

import aiosqlite


class Memory:
    """Lumen's persistent memory. Stores and recalls information using SQLite + FTS5.

    Used for: task tracking, notes, conversation facts, anything Lumen should
    remember across sessions.
    """

    def __init__(self, db_path: str | Path = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: aiosqlite.Connection | None = None

    async def init(self):
        """Initialize database and create tables."""
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content=memories,
                content_rowid=id
            );

            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories
            BEGIN
                INSERT INTO memories_fts(rowid, content)
                VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories
            BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;

            CREATE TABLE IF NOT EXISTS session_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                importance REAL DEFAULT 0.5,
                source_turn INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                summary TEXT DEFAULT '',
                fact_count INTEGER DEFAULT 0,
                turn_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                source TEXT DEFAULT '',
                confidence REAL DEFAULT 0.8,
                pinned INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_triggered REAL,
                trigger_count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_session_facts_session ON session_facts(session_id);
            CREATE INDEX IF NOT EXISTS idx_session_facts_category ON session_facts(category);
            CREATE INDEX IF NOT EXISTS idx_lessons_category ON lessons(category);
            CREATE INDEX IF NOT EXISTS idx_lessons_pinned ON lessons(pinned);

            CREATE TABLE IF NOT EXISTS outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                output_id TEXT UNIQUE NOT NULL,
                session_id TEXT DEFAULT '',
                type TEXT NOT NULL DEFAULT 'text',
                content TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_outputs_session ON outputs(session_id);
            CREATE INDEX IF NOT EXISTS idx_outputs_type ON outputs(type);
            CREATE INDEX IF NOT EXISTS idx_outputs_created ON outputs(created_at);
            """
        )
        await self._db.commit()

    async def remember(
        self,
        content: str,
        category: str = "general",
        metadata: dict | None = None,
    ) -> int:
        """Store something in memory. Returns the memory ID."""
        cursor = await self._db.execute(
            "INSERT INTO memories (content, category, metadata, created_at) "
            "VALUES (?, ?, ?, ?)",
            (content, category, json.dumps(metadata or {}), time.time()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def recall(self, query: str, limit: int = 5) -> list[dict]:
        """Search memory using FTS5. Returns matching memories ranked by relevance."""
        safe_query = " ".join(f'"{term}"' for term in query.split() if term.strip())
        if not safe_query:
            return []

        try:
            rows = await self._db.execute_fetchall(
                """
                SELECT m.id, m.content, m.category, m.metadata, m.created_at
                FROM memories_fts f
                JOIN memories m ON f.rowid = m.id
                WHERE memories_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (safe_query, limit),
            )
        except Exception:
            # Fallback to LIKE search if FTS fails
            rows = await self._db.execute_fetchall(
                """
                SELECT id, content, category, metadata, created_at
                FROM memories
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )

        return [
            {
                "id": row[0],
                "content": row[1],
                "category": row[2],
                "metadata": json.loads(row[3]),
                "created_at": row[4],
            }
            for row in rows
        ]

    async def list_by_category(
        self, category: str, limit: int = 20
    ) -> list[dict]:
        """List memories in a category, newest first."""
        rows = await self._db.execute_fetchall(
            "SELECT id, content, category, metadata, created_at "
            "FROM memories WHERE category = ? ORDER BY created_at DESC LIMIT ?",
            (category, limit),
        )
        return [
            {
                "id": row[0],
                "content": row[1],
                "category": row[2],
                "metadata": json.loads(row[3]),
                "created_at": row[4],
            }
            for row in rows
        ]

    async def forget(self, memory_id: int):
        """Delete a memory by ID."""
        await self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        await self._db.commit()

    async def save_conversation_turn(
        self, session_id: str, role: str, content: str
    ):
        """Save a conversation turn to persistent storage."""
        await self.remember(
            content,
            category=f"conversation:{session_id}",
            metadata={"role": role, "session_id": session_id},
        )

    async def load_conversation(
        self, session_id: str, limit: int = 50
    ) -> list[dict]:
        """Load conversation history for a session from persistent storage."""
        rows = await self._db.execute_fetchall(
            "SELECT content, metadata, created_at FROM memories "
            "WHERE category = ? ORDER BY created_at ASC LIMIT ?",
            (f"conversation:{session_id}", limit),
        )
        return [
            {
                "role": json.loads(row[1]).get("role", "user"),
                "content": row[0],
            }
            for row in rows
        ]

    async def close(self):
        """Close the database connection."""
        if self._db:
            await self._db.close()

    # --- Session Facts ---

    async def save_session_fact(
        self, session_id: str, fact: str, category: str = "general", importance: float = 0.5
    ) -> int:
        """Save a distilled fact from a session."""
        cursor = await self._db.execute(
            "INSERT INTO session_facts (session_id, fact, category, importance, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, fact, category, importance, time.time()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def list_session_facts(self, query: str = "", limit: int = 10) -> list[dict]:
        """List session facts, optionally filtered by query."""
        if query:
            rows = await self._db.execute_fetchall(
                "SELECT id, session_id, fact, category, importance, created_at "
                "FROM session_facts WHERE fact LIKE ? ORDER BY importance DESC LIMIT ?",
                (f"%{query}%", limit),
            )
        else:
            rows = await self._db.execute_fetchall(
                "SELECT id, session_id, fact, category, importance, created_at "
                "FROM session_facts ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        return [
            {"id": r[0], "session_id": r[1], "fact": r[2], "category": r[3],
             "importance": r[4], "created_at": r[5]}
            for r in rows
        ]

    async def list_session_summaries(self, limit: int = 20) -> list[dict]:
        """List session summaries."""
        rows = await self._db.execute_fetchall(
            "SELECT id, session_id, summary, fact_count, turn_count, created_at "
            "FROM session_summaries ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "session_id": r[1], "summary": r[2],
             "fact_count": r[3], "turn_count": r[4], "created_at": r[5]}
            for r in rows
        ]

    async def save_session_summary(
        self, session_id: str, summary: str, fact_count: int = 0, turn_count: int = 0
    ) -> int:
        """Save or update a session summary."""
        # Use INSERT OR REPLACE for upsert behavior
        cursor = await self._db.execute(
            "INSERT OR REPLACE INTO session_summaries (session_id, summary, fact_count, turn_count, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, summary, fact_count, turn_count, time.time()),
        )
        await self._db.commit()
        return cursor.lastrowid

    # --- Lessons ---

    async def save_lesson(
        self, rule: str, category: str = "general", source: str = "", confidence: float = 0.8
    ) -> int:
        """Save a new lesson."""
        cursor = await self._db.execute(
            "INSERT INTO lessons (rule, category, source, confidence, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (rule, category, source, confidence, time.time()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_lesson(self, lesson_id: int) -> dict | None:
        """Get a lesson by ID."""
        rows = await self._db.execute_fetchall(
            "SELECT id, rule, category, source, confidence, pinned, created_at, last_triggered, trigger_count "
            "FROM lessons WHERE id = ?",
            (lesson_id,),
        )
        if not rows:
            return None
        r = rows[0]
        return {
            "id": r[0], "rule": r[1], "category": r[2], "source": r[3],
            "confidence": r[4], "pinned": bool(r[5]), "created_at": r[6],
            "last_triggered": r[7], "trigger_count": r[8],
        }

    async def list_lessons(self, limit: int = 50) -> list[dict]:
        """List lessons, pinned first, then by confidence desc."""
        rows = await self._db.execute_fetchall(
            "SELECT id, rule, category, source, confidence, pinned, created_at, last_triggered, trigger_count "
            "FROM lessons ORDER BY pinned DESC, confidence DESC LIMIT ?",
            (limit,),
        )
        return [
            {"id": r[0], "rule": r[1], "category": r[2], "source": r[3],
             "confidence": r[4], "pinned": bool(r[5]), "created_at": r[6],
             "last_triggered": r[7], "trigger_count": r[8]}
            for r in rows
        ]

    async def delete_lesson(self, lesson_id: int):
        """Delete a lesson by ID."""
        await self._db.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
        await self._db.commit()

    async def update_lesson(
        self,
        lesson_id: int,
        rule: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        pinned: int | None = None,
        last_triggered: float | None = None,
        trigger_count_inc: int = 0,
    ):
        """Update lesson fields. Only provided fields are changed."""
        updates = []
        params = []

        if rule is not None:
            updates.append("rule = ?")
            params.append(rule)
        if category is not None:
            updates.append("category = ?")
            params.append(category)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if pinned is not None:
            updates.append("pinned = ?")
            params.append(pinned)
        if last_triggered is not None:
            updates.append("last_triggered = ?")
            params.append(last_triggered)
        if trigger_count_inc > 0:
            updates.append("trigger_count = trigger_count + ?")
            params.append(trigger_count_inc)

        if not updates:
            return

        params.append(lesson_id)
        await self._db.execute(
            f"UPDATE lessons SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        await self._db.commit()

    # ─── Structured Outputs ───

    async def save_output(self, output: "StructuredOutput") -> int:
        """Persist a structured output. Returns the row ID."""
        from lumen.core.output_types import StructuredOutput

        cursor = await self._db.execute(
            "INSERT OR REPLACE INTO outputs (output_id, session_id, type, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                output.output_id,
                output.session_id,
                output.type.value,
                output.content,
                json.dumps(output.metadata),
                output.timestamp,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_outputs(
        self,
        session_id: str | None = None,
        output_type: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Retrieve structured outputs, optionally filtered."""
        conditions = []
        params: list = []

        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if output_type:
            conditions.append("type = ?")
            params.append(output_type)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.extend([limit, offset])

        rows = await self._db.execute_fetchall(
            f"SELECT id, output_id, session_id, type, content, metadata, created_at "
            f"FROM outputs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        return [
            {
                "id": r[0],
                "output_id": r[1],
                "session_id": r[2],
                "type": r[3],
                "content": r[4],
                "metadata": json.loads(r[5]) if r[5] else {},
                "created_at": r[6],
            }
            for r in rows
        ]

    async def count_outputs(self, session_id: str | None = None) -> int:
        """Count outputs, optionally filtered by session."""
        if session_id:
            row = await self._db.execute_fetchall(
                "SELECT COUNT(*) FROM outputs WHERE session_id = ?", (session_id,)
            )
        else:
            row = await self._db.execute_fetchall("SELECT COUNT(*) FROM outputs")
        return row[0][0] if row else 0

    async def delete_output(self, output_id: str) -> bool:
        """Delete an output by its output_id. Returns True if deleted."""
        cursor = await self._db.execute(
            "DELETE FROM outputs WHERE output_id = ?", (output_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def get_stats(self) -> dict:
        """Get memory system statistics."""
        # Count memories by category
        rows = await self._db.execute_fetchall(
            "SELECT category, COUNT(*) as cnt FROM memories GROUP BY category"
        )
        categories = {r[0]: r[1] for r in rows}

        total = sum(categories.values())
        sessions = sum(v for k, v in categories.items() if k.startswith("conversation:"))

        # Count facts and lessons
        fact_rows = await self._db.execute_fetchall("SELECT COUNT(*) FROM session_facts")
        lesson_rows = await self._db.execute_fetchall("SELECT COUNT(*) FROM lessons")

        return {
            "total_memories": total,
            "total_sessions": sessions,
            "categories": categories,
            "total_facts": fact_rows[0][0] if fact_rows else 0,
            "total_lessons": lesson_rows[0][0] if lesson_rows else 0,
        }

    async def purge_old_conversations(self, days: int = 30) -> int:
        """Delete conversation memories older than N days.

        Returns the number of rows deleted.
        """
        cutoff = time.time() - (days * 86400)
        cursor = await self._db.execute(
            "DELETE FROM memories WHERE category LIKE 'conversation:%' AND created_at < ?",
            (cutoff,),
        )
        await self._db.commit()
        return cursor.rowcount
