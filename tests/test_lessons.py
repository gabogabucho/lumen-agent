import asyncio
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, MagicMock
import unittest

import pytest


class TestLessonStore(unittest.TestCase):
    """Test the LessonStore."""

    def _make_memory(self):
        memory = MagicMock()
        memory.save_lesson = AsyncMock(return_value=1)
        memory.get_lesson = AsyncMock(return_value=None)
        memory.list_lessons = AsyncMock(return_value=[])
        memory.delete_lesson = AsyncMock()
        memory.update_lesson = AsyncMock()
        return memory

    def test_add_lesson(self):
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        lid = asyncio.run(store.add_lesson("Always say hello"))
        assert lid == 1

    def test_add_lesson_validates_category(self):
        """Invalid category falls back to 'general'."""
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        store = LessonStore(memory)
        asyncio.run(store.add_lesson("test", category="invalid_cat"))
        # Check that 'general' was passed, not 'invalid_cat'
        call_args = memory.save_lesson.call_args
        assert call_args[1].get("category") == "general" or call_args[0][1] == "general"

    def test_get_lesson_not_found(self):
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        result = asyncio.run(store.get_lesson(999))
        assert result is None

    def test_get_lesson_found(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.8, "pinned": False, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        result = asyncio.run(store.get_lesson(1))
        assert result is not None
        assert result.rule == "test"

    def test_delete_lesson_not_found(self):
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        result = asyncio.run(store.delete_lesson(999))
        assert result is False

    def test_delete_lesson_found(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.8, "pinned": False, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        result = asyncio.run(store.delete_lesson(1))
        assert result is True
        memory.delete_lesson.assert_called_once_with(1)

    def test_pin_lesson(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.8, "pinned": False, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        result = asyncio.run(store.pin_lesson(1))
        assert result is True
        memory.update_lesson.assert_called_once_with(1, pinned=1)

    def test_unpin_lesson(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.8, "pinned": True, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        result = asyncio.run(store.unpin_lesson(1))
        assert result is True
        memory.update_lesson.assert_called_once_with(1, pinned=0)

    def test_format_for_prompt_empty(self):
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        result = store.format_for_prompt([])
        assert result == ""

    def test_format_for_prompt_with_lessons(self):
        from lumen.core.lessons import LessonStore, Lesson
        store = LessonStore(self._make_memory())
        lessons = [
            Lesson(id=1, rule="Always be polite", category="general", pinned=False),
            Lesson(id=2, rule="Never delete without asking", category="safety", pinned=True),
        ]
        result = store.format_for_prompt(lessons)
        assert "Always be polite" in result
        assert "[PINNED]" in result
        assert "Never delete without asking" in result
        assert "Learned Rules" in result

    def test_decay_confidence(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.8, "pinned": False, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        result = asyncio.run(store.decay_confidence(1, amount=0.2))
        assert result is True
        memory.update_lesson.assert_called_once()
        # Check confidence was decreased
        call_kwargs = memory.update_lesson.call_args[1]
        assert call_kwargs["confidence"] == pytest.approx(0.6)

    def test_decay_confidence_floor(self):
        """Confidence never goes below 0."""
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.get_lesson = AsyncMock(return_value={
            "id": 1, "rule": "test", "category": "general", "source": "manual",
            "confidence": 0.05, "pinned": False, "created_at": 1000.0,
            "last_triggered": None, "trigger_count": 0,
        })
        store = LessonStore(memory)
        asyncio.run(store.decay_confidence(1, amount=0.5))
        call_kwargs = memory.update_lesson.call_args[1]
        assert call_kwargs["confidence"] == 0.0

    def test_auto_lesson_below_threshold(self):
        """Auto-lesson not created if occurrence count < 3."""
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        result = asyncio.run(store.check_auto_lesson("some error", "sess-1", occurrence_count=2))
        assert result is None
        store.memory.save_lesson.assert_not_called()

    def test_auto_lesson_above_threshold(self):
        """Auto-lesson created when occurrence count >= 3."""
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        result = asyncio.run(store.check_auto_lesson("some error", "sess-1", occurrence_count=3))
        assert result is not None
        store.memory.save_lesson.assert_called_once()

    def test_auto_lesson_duplicate_prevented(self):
        """Duplicate auto-lessons are prevented."""
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.list_lessons = AsyncMock(return_value=[
            {"rule": "When encountering this error pattern, handle it correctly: some error", "category": "tool_usage"}
        ])
        store = LessonStore(memory)
        result = asyncio.run(store.check_auto_lesson("some error", "sess-1", occurrence_count=5))
        assert result is None
        memory.save_lesson.assert_not_called()

    def test_get_stats(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.list_lessons = AsyncMock(return_value=[
            {"category": "safety", "pinned": True, "confidence": 0.9},
            {"category": "safety", "pinned": False, "confidence": 0.7},
            {"category": "preference", "pinned": False, "confidence": 0.5},
        ])
        store = LessonStore(memory)
        stats = asyncio.run(store.get_stats())
        assert stats["total"] == 3
        assert stats["pinned"] == 1
        assert stats["categories"]["safety"] == 2
        assert stats["avg_confidence"] == 0.7

    def test_trigger_lesson(self):
        from lumen.core.lessons import LessonStore
        store = LessonStore(self._make_memory())
        asyncio.run(store.trigger_lesson(1))
        store.memory.update_lesson.assert_called_once()
        call_kwargs = store.memory.update_lesson.call_args[1]
        assert call_kwargs["trigger_count_inc"] == 1
        assert call_kwargs["last_triggered"] is not None

    def test_get_active_lessons(self):
        from lumen.core.lessons import LessonStore
        memory = self._make_memory()
        memory.list_lessons = AsyncMock(return_value=[
            {"id": 1, "rule": "test", "category": "general", "source": "manual",
             "confidence": 0.8, "pinned": False, "created_at": 1000.0,
             "last_triggered": None, "trigger_count": 0}
        ])
        store = LessonStore(memory)
        lessons = asyncio.run(store.get_active_lessons())
        assert len(lessons) == 1
        assert lessons[0].rule == "test"


class TestMemoryExtension(unittest.TestCase):
    """Test new memory methods (session_facts, summaries, lessons)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_init_creates_new_tables(self):
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        # Verify tables exist by inserting and querying
        asyncio.run(memory.save_session_fact("s1", "test fact", "general", 0.5))
        asyncio.run(memory.save_lesson("test rule", "safety", "manual"))
        
        facts = asyncio.run(memory.list_session_facts())
        assert len(facts) == 1
        assert facts[0]["fact"] == "test fact"
        
        lessons = asyncio.run(memory.list_lessons())
        assert len(lessons) == 1
        assert lessons[0]["rule"] == "test rule"
        
        asyncio.run(memory.close())

    def test_session_facts_crud(self):
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        # Create
        fid = asyncio.run(memory.save_session_fact("s1", "fact1", "preference", 0.9))
        assert fid is not None
        
        # List
        facts = asyncio.run(memory.list_session_facts())
        assert len(facts) == 1
        
        # Search
        results = asyncio.run(memory.list_session_facts(query="fact1"))
        assert len(results) == 1

    def test_session_summaries(self):
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        asyncio.run(memory.save_session_summary("s1", "Summary text", 3, 10))
        asyncio.run(memory.save_session_summary("s2", "Another summary", 5, 20))
        
        summaries = asyncio.run(memory.list_session_summaries())
        assert len(summaries) == 2

    def test_session_summary_upsert(self):
        """Saving same session_id twice replaces (upsert)."""
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        asyncio.run(memory.save_session_summary("s1", "First", 1, 5))
        asyncio.run(memory.save_session_summary("s1", "Updated", 3, 10))
        
        summaries = asyncio.run(memory.list_session_summaries())
        assert len(summaries) == 1
        assert summaries[0]["summary"] == "Updated"
        assert summaries[0]["fact_count"] == 3

    def test_lessons_crud(self):
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        # Create
        lid = asyncio.run(memory.save_lesson("rule1", "safety", "manual", 0.9))
        
        # Read
        lesson = asyncio.run(memory.get_lesson(lid))
        assert lesson["rule"] == "rule1"
        assert lesson["pinned"] is False
        
        # Update
        asyncio.run(memory.update_lesson(lid, pinned=1))
        lesson = asyncio.run(memory.get_lesson(lid))
        assert lesson["pinned"] is True
        
        # Trigger
        asyncio.run(memory.update_lesson(lid, trigger_count_inc=1))
        lesson = asyncio.run(memory.get_lesson(lid))
        assert lesson["trigger_count"] == 1
        
        # Delete
        asyncio.run(memory.delete_lesson(lid))
        lesson = asyncio.run(memory.get_lesson(lid))
        assert lesson is None

    def test_lessons_ordered_pinned_first(self):
        """Pinned lessons appear before unpinned."""
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        asyncio.run(memory.save_lesson("unpinned", "general", "manual", 0.5))
        asyncio.run(memory.save_lesson("pinned_rule", "safety", "manual", 0.5))
        asyncio.run(memory.update_lesson(2, pinned=1))
        
        lessons = asyncio.run(memory.list_lessons())
        assert lessons[0]["rule"] == "pinned_rule"
        assert lessons[0]["pinned"] is True

    def test_get_stats(self):
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        asyncio.run(memory.remember("test memory", "general"))
        asyncio.run(memory.save_session_fact("s1", "fact"))
        asyncio.run(memory.save_lesson("rule"))
        
        stats = asyncio.run(memory.get_stats())
        assert stats["total_memories"] >= 1
        assert stats["total_facts"] == 1
        assert stats["total_lessons"] == 1
        
        asyncio.run(memory.close())

    def test_existing_memories_still_work(self):
        """Existing memory methods are not broken by new tables."""
        from lumen.core.memory import Memory
        memory = Memory(db_path=self.db_path)
        asyncio.run(memory.init())
        
        mid = asyncio.run(memory.remember("hello world", "general"))
        results = asyncio.run(memory.recall("hello"))
        assert len(results) >= 1
        
        asyncio.run(memory.forget(mid))
        results = asyncio.run(memory.recall("hello"))
        assert len(results) == 0
        
        asyncio.run(memory.close())


if __name__ == "__main__":
    unittest.main()
