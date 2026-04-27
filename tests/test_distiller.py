import asyncio
import json
from unittest.mock import AsyncMock, patch, MagicMock
import unittest


class TestDistillSession(unittest.TestCase):
    """Test session distillation."""

    def _make_memory(self, turns=None):
        memory = MagicMock()
        memory.load_conversation = AsyncMock(return_value=turns or [])
        memory.save_session_fact = AsyncMock(return_value=1)
        memory.save_session_summary = AsyncMock(return_value=1)
        return memory

    def test_skips_short_sessions(self):
        """Sessions with fewer than min_turns are skipped."""
        from lumen.core.distiller import SessionDistiller
        memory = self._make_memory(turns=[{"role": "user", "content": "hi"}])
        distiller = SessionDistiller(memory=memory, min_turns=4)
        result = asyncio.run(distiller.distill_session("sess-1"))
        assert result == []
        memory.save_session_fact.assert_not_called()

    def test_extracts_facts_from_conversation(self):
        """Valid conversation produces extracted facts."""
        from lumen.core.distiller import SessionDistiller
        turns = [
            {"role": "user", "content": "My name is Gabo"},
            {"role": "assistant", "content": "Nice to meet you Gabo!"},
            {"role": "user", "content": "I prefer Python over JavaScript"},
            {"role": "assistant", "content": "Good choice!"},
            {"role": "user", "content": "My project is called Lumen"},
            {"role": "assistant", "content": "Great name!"},
        ]
        facts_response = json.dumps([
            {"fact": "User's name is Gabo", "category": "fact", "importance": 0.9},
            {"fact": "User prefers Python over JavaScript", "category": "preference", "importance": 0.7},
        ])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=facts_response))]
        
        memory = self._make_memory(turns=turns)
        distiller = SessionDistiller(memory=memory, min_turns=4)
        
        with patch("lumen.core.distiller.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(distiller.distill_session("sess-2"))
        
        assert len(result) == 2
        assert result[0].fact == "User's name is Gabo"
        assert memory.save_session_fact.call_count == 2
        memory.save_session_summary.assert_called_once()

    def test_handles_llm_error_gracefully(self):
        """LLM failure returns empty list, doesn't crash."""
        from lumen.core.distiller import SessionDistiller
        turns = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        
        memory = self._make_memory(turns=turns)
        distiller = SessionDistiller(memory=memory, min_turns=4)
        
        with patch("lumen.core.distiller.acompletion", new_callable=AsyncMock, side_effect=Exception("API down")):
            result = asyncio.run(distiller.distill_session("sess-3"))
        
        assert result == []
        memory.save_session_fact.assert_not_called()

    def test_handles_malformed_llm_response(self):
        """Non-JSON LLM response returns empty list."""
        from lumen.core.distiller import SessionDistiller
        turns = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Not JSON at all"))]
        
        memory = self._make_memory(turns=turns)
        distiller = SessionDistiller(memory=memory, min_turns=4)
        
        with patch("lumen.core.distiller.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(distiller.distill_session("sess-4"))
        
        assert result == []

    def test_skips_empty_facts(self):
        """Facts with empty 'fact' field are filtered out."""
        from lumen.core.distiller import SessionDistiller
        turns = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        facts_response = json.dumps([
            {"fact": "Valid fact", "category": "fact", "importance": 0.5},
            {"fact": "", "category": "fact", "importance": 0.5},
            {"fact": "Another valid", "category": "preference", "importance": 0.3},
        ])
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=facts_response))]
        
        memory = self._make_memory(turns=turns)
        distiller = SessionDistiller(memory=memory, min_turns=4)
        
        with patch("lumen.core.distiller.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(distiller.distill_session("sess-5"))
        
        assert len(result) == 2

    def test_parses_markdown_code_block(self):
        """JSON in markdown code block is parsed correctly."""
        from lumen.core.distiller import SessionDistiller
        turns = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        facts_response = '```json\n[{"fact": "Parsed from code block", "category": "fact", "importance": 0.5}]\n```'
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=facts_response))]
        
        memory = self._make_memory(turns=turns)
        distiller = SessionDistiller(memory=memory, min_turns=4)
        
        with patch("lumen.core.distiller.acompletion", new_callable=AsyncMock, return_value=mock_response):
            result = asyncio.run(distiller.distill_session("sess-6"))
        
        assert len(result) == 1
        assert result[0].fact == "Parsed from code block"

    def test_truncates_long_turns(self):
        """Very long conversation turns are truncated to 500 chars."""
        from lumen.core.distiller import SessionDistiller
        long_content = "x" * 1000
        turns = [{"role": "user", "content": long_content} for _ in range(5)]
        
        distiller = SessionDistiller(memory=MagicMock(), min_turns=4)
        formatted = distiller._format_conversation(turns)
        
        for line in formatted.split("\n"):
            # Content part (after "]: ") should be <= 503 (500 + "...")
            content_part = line.split("]: ", 1)[1] if "]: " in line else line
            assert len(content_part) <= 503

    def test_generates_summary(self):
        """Summary is generated from top facts."""
        from lumen.core.distiller import SessionDistiller
        facts = [
            MagicMock(fact="Important fact", importance=0.9),
            MagicMock(fact="Less important", importance=0.3),
        ]
        distiller = SessionDistiller(memory=MagicMock())
        summary = distiller._generate_summary(facts, 10)
        
        assert "10 turns" in summary
        assert "2 facts" in summary
        assert "Important fact" in summary

    def test_recall_facts(self):
        """recall_facts delegates to memory.list_session_facts."""
        from lumen.core.distiller import SessionDistiller
        memory = MagicMock()
        memory.list_session_facts = AsyncMock(return_value=[{"fact": "test"}])
        distiller = SessionDistiller(memory=memory)
        
        result = asyncio.run(distiller.recall_facts("test query"))
        assert result == [{"fact": "test"}]
        memory.list_session_facts.assert_called_once_with(query="test query", limit=10)

    def test_list_summaries(self):
        """list_summaries delegates to memory.list_session_summaries."""
        from lumen.core.distiller import SessionDistiller
        memory = MagicMock()
        memory.list_session_summaries = AsyncMock(return_value=[{"session_id": "s1"}])
        distiller = SessionDistiller(memory=memory)
        
        result = asyncio.run(distiller.list_summaries())
        assert len(result) == 1


if __name__ == "__main__":
    unittest.main()
