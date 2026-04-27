"""Persistent lessons — learn from errors and user corrections.

Lessons are durable rules that Lumen follows across sessions to avoid
repeating mistakes. They can be auto-generated (when the same error
occurs 3+ times) or manually created by the user.

Inspired by Aiden's LESSONS.md pattern, but stored in SQLite for
queryability and structured management.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class LessonCategory(str, Enum):
    SAFETY = "safety"
    PREFERENCE = "preference"
    TOOL_USAGE = "tool_usage"
    FORMAT = "format"
    GENERAL = "general"


VALID_CATEGORIES = {c.value for c in LessonCategory}


@dataclass
class Lesson:
    """A persistent lesson learned by the agent."""
    id: int = 0
    rule: str = ""
    category: str = "general"
    source: str = ""           # "session:abc", "user:manual", "system:auto"
    confidence: float = 0.8    # 0.0-1.0
    pinned: bool = False
    created_at: float = 0.0
    last_triggered: float | None = None
    trigger_count: int = 0


LESSON_INJECTION_PROMPT = """## Learned Rules (never violate these):
{lessons}

If a user request conflicts with a learned rule, follow the rule and explain why."""


class LessonStore:
    """Manages persistent lessons in SQLite.
    
    Usage:
        store = LessonStore(memory)
        
        # Manual lesson
        lesson_id = await store.add_lesson(
            rule="Always confirm before deleting files",
            category="safety",
            source="user:manual"
        )
        
        # Get lessons for prompt injection
        lessons = await store.get_active_lessons()
        injection = store.format_for_prompt(lessons)
    """

    def __init__(self, memory: Any):
        self.memory = memory

    async def add_lesson(
        self,
        rule: str,
        category: str = "general",
        source: str = "user:manual",
        confidence: float = 0.8,
    ) -> int:
        """Add a new lesson. Returns the lesson ID."""
        if category not in VALID_CATEGORIES:
            category = "general"
        return await self.memory.save_lesson(
            rule=rule,
            category=category,
            source=source,
            confidence=confidence,
        )

    async def get_lesson(self, lesson_id: int) -> Lesson | None:
        """Get a lesson by ID."""
        row = await self.memory.get_lesson(lesson_id)
        if not row:
            return None
        return Lesson(**row)

    async def get_active_lessons(self, limit: int = 20) -> list[Lesson]:
        """Get all active lessons, pinned first, then by confidence."""
        rows = await self.memory.list_lessons(limit=limit)
        return [Lesson(**r) for r in rows]

    async def delete_lesson(self, lesson_id: int) -> bool:
        """Delete a lesson. Returns True if found."""
        existing = await self.memory.get_lesson(lesson_id)
        if not existing:
            return False
        await self.memory.delete_lesson(lesson_id)
        return True

    async def pin_lesson(self, lesson_id: int) -> bool:
        """Pin a lesson (prevent auto-deletion). Returns True if found."""
        existing = await self.memory.get_lesson(lesson_id)
        if not existing:
            return False
        await self.memory.update_lesson(lesson_id, pinned=1)
        return True

    async def unpin_lesson(self, lesson_id: int) -> bool:
        """Unpin a lesson. Returns True if found."""
        existing = await self.memory.get_lesson(lesson_id)
        if not existing:
            return False
        await self.memory.update_lesson(lesson_id, pinned=0)
        return True

    async def trigger_lesson(self, lesson_id: int) -> None:
        """Record that a lesson was triggered (used in a response)."""
        await self.memory.update_lesson(
            lesson_id,
            last_triggered=time.time(),
            trigger_count_inc=1,
        )

    async def decay_confidence(self, lesson_id: int, amount: float = 0.1) -> bool:
        """Decrease confidence (e.g., when lesson was violated)."""
        existing = await self.memory.get_lesson(lesson_id)
        if not existing:
            return False
        current = existing.confidence if hasattr(existing, 'confidence') else existing.get("confidence", 0.0)
        new_confidence = max(0.0, current - amount)
        await self.memory.update_lesson(lesson_id, confidence=new_confidence)
        return True

    async def check_auto_lesson(self, error_pattern: str, session_id: str, occurrence_count: int = 3) -> int | None:
        """Check if an error pattern should auto-generate a lesson.
        
        Returns lesson ID if created, None if not.
        """
        if occurrence_count < 3:
            return None
        
        # Check if a similar lesson already exists
        existing = await self.memory.list_lessons(limit=100)
        for lesson_data in existing:
            if error_pattern.lower() in lesson_data["rule"].lower():
                return None  # Already have this lesson
        
        rule = f"When encountering this error pattern, handle it correctly: {error_pattern}"
        lesson_id = await self.add_lesson(
            rule=rule,
            category="tool_usage",
            source=f"system:auto:{session_id}",
            confidence=0.6,  # Lower confidence for auto-generated
        )
        logger.info(f"Auto-generated lesson {lesson_id} from error pattern in session {session_id}")
        return lesson_id

    def format_for_prompt(self, lessons: list[Lesson]) -> str:
        """Format lessons for injection into system prompt."""
        if not lessons:
            return ""
        
        lines = []
        for lesson in lessons:
            pin_marker = "[PINNED] " if lesson.pinned else ""
            lines.append(f"- {pin_marker}{lesson.rule}")
        
        return LESSON_INJECTION_PROMPT.format(lessons="\n".join(lines))

    async def get_stats(self) -> dict[str, Any]:
        """Get lesson statistics."""
        rows = await self.memory.list_lessons(limit=1000)
        categories = {}
        pinned_count = 0
        total_confidence = 0.0
        for r in rows:
            cat = r.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1
            if r.get("pinned"):
                pinned_count += 1
            total_confidence += r.get("confidence", 0.0)
        
        return {
            "total": len(rows),
            "pinned": pinned_count,
            "categories": categories,
            "avg_confidence": round(total_confidence / len(rows), 2) if rows else 0.0,
        }
