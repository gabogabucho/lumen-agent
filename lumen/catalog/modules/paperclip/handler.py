"""Paperclip module — endpoint handlers.

Integration with Paperclip for multi-agent orchestration.
Processes tasks from Paperclip through Lumen's brain and reports status.
"""

import time
from datetime import datetime, timedelta


async def get_skill_content(skill_name: str) -> str | None:
    """Read a skill's SKILL.md from the paperclip module directory."""
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent.parent.parent  # lumen/
    skill_path = pkg_dir / "catalog" / "modules" / "paperclip" / "skills" / skill_name / "SKILL.md"
    if skill_path.exists():
        return skill_path.read_text(encoding="utf-8")

    # Fallback: try from ~/.lumen/modules (runtime install location)
    home = Path.home() / ".lumen" / "modules" / "paperclip" / "skills" / skill_name / "SKILL.md"
    if home.exists():
        return home.read_text(encoding="utf-8")
    return None


async def handle_task(request: dict, config: dict, brain, memory, awareness=None, **kwargs) -> dict:
    """POST /paperclip/task — Receives a task from Paperclip, processes it.

    Args:
        request: Dict with path, query_params, headers, body (parsed JSON).
        config: Lumen runtime config dict.
        brain: The Brain instance (core/brain.py).
        memory: The Memory instance (core/memory.py).
        awareness: Optional CapabilityAwareness instance.
        **kwargs: For future extensibility (e.g., session_id, request_obj).

    Returns:
        JSON-serializable task result dict.
    """
    task_id = request.get("body", {}).get("task_id", "unknown")
    data = request.get("body", {})
    start = time.time()

    # Build the prompt with skill context
    skill_content = await get_skill_content("paperclip-awareness")
    skill_prefix = skill_content or _default_skill_prefix()

    prompt = f"""{skill_prefix}

TASK: {data.get('title', 'No title')}

DESCRIPTION:
{data.get('description', 'No description')}

GOAL:
{data.get('goal', 'Not specified')}

PRIORITY: {data.get('priority', 'normal')}

CONTEXT:
{data.get('context', 'Not provided')}

Complete this task. Return your result in this exact format:
SUMMARY: one sentence describing what you did or decided
OUTPUT: your full work product or analysis
ACTIONS: list any concrete actions you took (one per line, or "none")
BLOCKERS: list any blockers encountered (one per line, or "none")
NEXT STEP: one concrete suggested next action"""

    try:
        # Create a dedicated session for this task
        session_id = f"paperclip-task-{task_id}"

        # Call brain directly — Lumen's brain thinks with full context
        if brain:
            response = await brain.think(
                message=prompt,
                session=_make_session(session_id)
            )
            output = response.get("content", "") if isinstance(response, dict) else str(response)
        else:
            output = "Brain not available. Cannot process task."

        # Parse structured response
        result = _parse_task_response(output)

        # Store in memory
        title = data.get("title", "unnamed task")
        await memory.remember(
            content=f"Paperclip task '{title}' completed. Result: {result['summary']}",
            category="paperclip"
        )

        # Update capability awareness
        if awareness:
            try:
                awareness.set_status(
                    capability="paperclip.task.receiver",
                    status="active",
                    metadata={"last_task_id": task_id, "last_run": datetime.now().isoformat()}
                )
            except Exception:
                pass

        duration_ms = int((time.time() - start) * 1000)

        response_data = {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }

        return response_data

    except Exception as e:
        error_str = str(e)
        # Don't leak internal errors to Paperclip
        safe_error = "error processing task" if "error" in error_str.lower() else error_str[:200]

        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": safe_error,
            "timestamp": datetime.now().isoformat()
        }


async def handle_report(request: dict, config: dict, brain, memory, awareness=None, **kwargs) -> dict:
    """GET /paperclip/report — Returns current state for Paperclip CEO.

    Args:
        request: Dict with query_params and body.
        config: Lumen runtime config dict.
        brain: The Brain instance.
        memory: The Memory instance.
        awareness: Optional CapabilityAwareness instance.

    Returns:
        JSON-serializable report dict.
    """
    try:
        period = request.get("query_params", {}).get("period", "7d")
        days = {"1d": 1, "7d": 7, "30d": 30}.get(period, 7)
        since = datetime.now() - timedelta(days=days)

        # Agent info
        agent_id = config.get("paperclip.agent_id", "unregistered")
        agent_role = config.get("paperclip.agent_role", "Lumen Agent")
        personality_name = "default"
        lumen_version = "0.0.0"

        if brain:
            # Get personality name from brain
            try:
                personality = brain.personality.current() if hasattr(brain, "personality") else {}
                personality_name = (personality.get("identity") or {}).get("name", "default")
            except Exception:
                personality_name = "default"

            # Get version from config
            lumen_version = config.get("model", "0.0.0").split("/")[0] if "/" in str(config.get("model", "")) else "0.0.0"
            try:
                import lumen as lumen_pkg
                lumen_version = getattr(lumen_pkg, "__version__", "0.0.0")
            except Exception:
                pass

        # Task stats from memory
        tasks = await _get_task_stats(memory, since)

        # Memory stats
        memory_stats = await _get_memory_stats(memory, since)

        # Custom stats — hook for personalities/modules
        try:
            custom = {}
            if hasattr(brain, "personality") and brain.personality:
                stats_method = getattr(brain.personality, "paperclip_stats", None)
                if callable(stats_method):
                    custom_result = stats_method()
                    if isinstance(custom_result, dict):
                        custom.update(custom_result)
            # Allow modules to add custom stats
            if hasattr(memory, "list_by_category"):
                paperclip_facts = await memory.list_by_category("paperclip", limit=10) or []
                if paperclip_facts:
                    custom["paperclip_tasks"] = len(paperclip_facts)
        except Exception:
            custom = {}

        report = {
            "ok": True,
            "agent": {
                "id": agent_id,
                "role": agent_role,
                "lumen_version": lumen_version,
                "personality": str(personality_name),
                "status": "online",
            },
            "tasks": tasks,
            "memory": memory_stats,
            "custom": custom,
            "period": period,
            "timestamp": datetime.now().isoformat()
        }

        return report

    except Exception as e:
        return {
            "ok": False,
            "error": f"report generation failed: {e}",
            "timestamp": datetime.now().isoformat()
        }


async def handle_heartbeat(request: dict, config: dict, brain, memory, awareness=None, **kwargs) -> dict:
    """POST /paperclip/heartbeat — Receives heartbeat and directives from Paperclip.

    Args:
        request: Dict with body (directives array).
        config: Lumen runtime config dict.
        brain: The Brain instance.
        memory: The Memory instance.
        awareness: Optional CapabilityAwareness instance.

    Returns:
        JSON-serializable heartbeat response.
    """
    try:
        data = request.get("body", {})
        directives = data.get("directives", [])

        for directive in directives:
            directive_type = directive.get("type", "")
            directive_content = directive.get("content", "")

            if directive_type == "goal_update":
                await memory.remember(
                    content=f"[Paperclip directive] {directive_content}",
                    category="paperclip"
                )
            elif directive_type == "context_update":
                await memory.remember(
                    content=f"[Paperclip context] {directive_content}",
                    category="paperclip"
                )

        return {
            "ok": True,
            "agent_id": config.get("paperclip.agent_id", "unregistered"),
            "status": "online",
            "directive_count": len(directives),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_resume(request: dict, config: dict, memory) -> dict:
    """POST /paperclip/resume — Resume a previously interrupted task.

    Paperclip calls this when a task was interrupted (timeout, error)
    and wants to try again with updated context.

    Args:
        request: Dict with task_id and new context.
        config: Lumen runtime config dict.
        memory: The Memory instance.

    Returns:
        JSON-serializable resume response.
    """
    task_id = request.get("body", {}).get("task_id", "unknown")
    data = request.get("body", {})

    # Find previous task result in memory
    try:
        results = await memory.recall(query="paperclip task", limit=5)
        recent = [r for r in results if task_id in r.get("content", "")]

        if recent:
            last_result = recent[0].get("content", "")
        else:
            last_result = "No previous result found."
    except Exception:
        last_result = "No previous result found."

    return {
        "ok": True,
        "task_id": task_id,
        "previous_result_summary": last_result[:200],
        "ready_to_resume": True,
        "timestamp": datetime.now().isoformat()
    }


# ─── Internal helpers ───


def _make_session(session_id: str):
    """Create a lightweight Session-like object for paperclip tasks."""
    import time
    import uuid

    class _PaperclipSession:
        """Minimal session compatible with brain.think() signature."""
        def __init__(self, sid: str):
            self.session_id = sid
            self.history = []
            self.last_seen = time.time()

        def touch(self):
            self.last_seen = time.time()

        def add_message(self, role, content):
            self.history.append({"role": role, "content": content})

    return _PaperclipSession(session_id)


def _default_skill_prefix() -> str:
    """Fallback skill content if the SKILL.md file can't be read."""
    return (
        "You are a Lumen agent operating as part of a Paperclip-managed company.\n"
        "Have a role with specific responsibilities.\n"
        "A CEO assigns tasks and reads your reports.\n"
        "Company goals guide your work. Use tokens efficiently.\n"
    )


def _parse_task_response(text: str) -> dict:
    """Parse the structured brain output into result fields."""
    result = {
        "summary": "",
        "output": "",
        "actions_taken": [],
        "blockers": [],
        "next_step": ""
    }

    current_section = None
    lines = text.strip().split("\n")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("SUMMARY:"):
            result["summary"] = stripped[len("SUMMARY:"):].strip()
            current_section = None
        elif stripped.startswith("OUTPUT:"):
            current_section = "output"
            rest = stripped[len("OUTPUT:"):]
            if rest.strip():
                result["output"] = rest.strip()
        elif stripped.startswith("ACTIONS:"):
            current_section = "actions"
        elif stripped.startswith("BLOCKERS:"):
            current_section = "blockers"
        elif stripped.startswith("NEXT STEP:"):
            result["next_step"] = stripped[len("NEXT STEP:"):].strip()
            current_section = None

        elif current_section == "output" and line and not stripped.startswith(("SUMMARY:", "OUTPUT:", "ACTIONS:", "BLOCKERS:", "NEXT STEP:")):
            if result["output"] and not result["output"].endswith("\n"):
                result["output"] += "\n"
            result["output"] += line

        elif current_section == "actions" and stripped and not stripped.startswith(("SUMMARY:", "OUTPUT:", "ACTIONS:", "BLOCKERS:", "NEXT STEP:")):
            if stripped.lower() != "none":
                item = stripped.lstrip("- ").strip()
                if item:
                    result["actions_taken"].append(item)

        elif current_section == "blockers" and stripped and not stripped.startswith(("SUMMARY:", "OUTPUT:", "ACTIONS:", "BLOCKERS:", "NEXT STEP:")):
            if stripped.lower() != "none":
                item = stripped.lstrip("- ").strip()
                if item:
                    result["blockers"].append(item)

    return result


async def _get_task_stats(memory, since: datetime) -> dict:
    """Read task history from Lumen memory."""
    try:
        tasks = await memory.list_by_category("paperclip", limit=50) or []
        return {
            "completed_in_period": len(tasks),
            "last_task_at": tasks[-1].get("created_at") if tasks else None
        }
    except Exception:
        return {"completed_in_period": 0, "last_task_at": None}


async def _get_memory_stats(memory, since: datetime) -> dict:
    """Read recent memory activity."""
    try:
        recent = await memory.recall(query="", limit=5)
        total_facts = 0
        try:
            stats = await memory.get_stats()
            total_facts = stats.get("total_memories", 0)
        except Exception:
            pass

        return {
            "total_facts": total_facts,
            "recent_activity": [m.get("content", "")[:120] if isinstance(m, dict) else str(m)[:120] for m in (recent or [])]
        }
    except Exception:
        return {"total_facts": 0, "recent_activity": []}


if __name__ == "__main__":
    """Self-test: verify handler functions are importable."""
    import asyncio

    test_request = {
        "body": {
            "task_id": "test_1",
            "title": "Test task",
            "description": "A test",
            "goal": "Test goal",
            "priority": "low",
        }
    }

    print("handler.py loaded successfully, handlers:", [
        "handle_task",
        "handle_report",
        "handle_heartbeat",
        "handle_resume",
    ])
