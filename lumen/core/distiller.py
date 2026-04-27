"""Session distillation — extract durable facts from conversations.

After a session ends (or reaches a turn threshold), the distiller calls the
LLM to extract 5-15 durable facts from the conversation. These facts are
stored in the session_facts table and can be recalled in future sessions
to provide context continuity.

Pattern inspired by Aiden's session-end memory distillation.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from litellm import acompletion

logger = logging.getLogger(__name__)


@dataclass
class DistilledFact:
    """A single fact extracted from a conversation."""
    fact: str
    category: str = "general"  # preference, fact, decision, context, project
    importance: float = 0.5    # 0.0-1.0
    source_turn: int = 0


@dataclass
class SessionSummary:
    """Summary of a distilled session."""
    session_id: str
    summary: str
    fact_count: int
    turn_count: int


DISTILL_PROMPT = """Analyze this conversation and extract durable facts that would be useful to remember for future interactions with this user.

For each fact:
- Must be a standalone, specific, actionable statement
- Include user preferences, decisions made, project details, context clues
- Rate importance: 1.0 = critical (identity, auth keys, core preferences), 0.5 = useful, 0.1 = minor
- Categorize: preference, fact, decision, context, project

Return ONLY a JSON array of objects with keys: fact, category, importance (float 0.0-1.0).
Extract 5-15 facts. If the conversation is trivial (greetings, small talk), extract fewer.

Conversation:
{conversation}"""


class SessionDistiller:
    """Distills conversations into durable facts.
    
    Usage:
        distiller = SessionDistiller(memory=memory, model="deepseek/deepseek-chat")
        
        # After session ends:
        facts = await distiller.distill_session("session-abc")
        # facts is a list of DistilledFact objects, also stored in DB
    """

    def __init__(
        self,
        memory: Any,  # Memory instance
        model: str = "deepseek/deepseek-chat",
        min_turns: int = 4,  # Don't distill very short sessions
        max_turns: int = 50,  # Limit context sent to LLM
    ):
        self.memory = memory
        self.model = model
        self.min_turns = min_turns
        self.max_turns = max_turns

    async def distill_session(self, session_id: str) -> list[DistilledFact]:
        """Distill a session into durable facts.
        
        Returns list of extracted facts (also stored in DB).
        """
        # Load conversation
        turns = await self.memory.load_conversation(session_id, limit=self.max_turns)
        
        if len(turns) < self.min_turns:
            logger.debug(f"Session {session_id} too short ({len(turns)} turns), skipping distillation")
            return []
        
        # Format conversation for LLM
        conversation_text = self._format_conversation(turns)
        
        # Call LLM to extract facts
        raw_facts = await self._extract_facts(conversation_text)
        
        if not raw_facts:
            return []
        
        # Store facts in DB
        stored_facts = []
        for i, fact_data in enumerate(raw_facts):
            fact = DistilledFact(
                fact=fact_data.get("fact", ""),
                category=fact_data.get("category", "general"),
                importance=float(fact_data.get("importance", 0.5)),
                source_turn=fact_data.get("source_turn", 0),
            )
            if fact.fact:  # Skip empty facts
                await self.memory.save_session_fact(
                    session_id=session_id,
                    fact=fact.fact,
                    category=fact.category,
                    importance=fact.importance,
                )
                stored_facts.append(fact)
        
        # Generate and store summary
        summary_text = self._generate_summary(stored_facts, len(turns))
        await self.memory.save_session_summary(
            session_id=session_id,
            summary=summary_text,
            fact_count=len(stored_facts),
            turn_count=len(turns),
        )
        
        logger.info(f"Distilled session {session_id}: {len(stored_facts)} facts from {len(turns)} turns")
        return stored_facts

    def _format_conversation(self, turns: list[dict]) -> str:
        """Format turns into readable conversation text."""
        lines = []
        for i, turn in enumerate(turns):
            role = turn.get("role", "unknown").upper()
            content = turn.get("content", "")
            # Truncate very long turns
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    async def _extract_facts(self, conversation: str) -> list[dict]:
        """Call LLM to extract facts from conversation."""
        prompt = DISTILL_PROMPT.format(conversation=conversation)
        
        try:
            response = await acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            content = response.choices[0].message.content if response.choices else ""
            return self._parse_facts_response(content)
        except Exception as e:
            logger.warning(f"Distillation LLM call failed: {e}")
            return []

    def _parse_facts_response(self, content: str) -> list[dict]:
        """Parse LLM response into list of fact dicts."""
        if not content:
            return []
        
        # Try to extract JSON array from response
        # Handle markdown code blocks
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                facts = json.loads(json_match.group())
                if isinstance(facts, list):
                    return [f for f in facts if isinstance(f, dict) and f.get("fact")]
            except json.JSONDecodeError:
                pass
        
        # Fallback: try parsing the whole content
        try:
            facts = json.loads(content.strip())
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, dict) and f.get("fact")]
        except json.JSONDecodeError:
            pass
        
        return []

    def _generate_summary(self, facts: list[DistilledFact], turn_count: int) -> str:
        """Generate a text summary from extracted facts."""
        if not facts:
            return f"Session with {turn_count} turns — no significant facts extracted."
        
        top_facts = sorted(facts, key=lambda f: f.importance, reverse=True)[:5]
        fact_lines = [f"- {f.fact}" for f in top_facts]
        return f"Session summary ({turn_count} turns, {len(facts)} facts):\n" + "\n".join(fact_lines)

    async def recall_facts(self, query: str = "", limit: int = 10) -> list[dict]:
        """Recall previously distilled facts, optionally filtered by query."""
        return await self.memory.list_session_facts(query=query, limit=limit)

    async def list_summaries(self, limit: int = 20) -> list[dict]:
        """List session summaries."""
        return await self.memory.list_session_summaries(limit=limit)
