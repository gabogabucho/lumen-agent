# lumen-paperclip
## Integration module spec — Lumen Agent × Paperclip

**Status:** RFC (ready for implementation)
**Target repo:** gabogabucho/lumen-agent
**Module path:** `lumen/catalog/modules/paperclip/`

---

## Problem this solves

Paperclip orchestrates agents but cannot hold a conversation, remember context,
or interact with humans through messaging channels.

Lumen converses, remembers, and acts — but has no org chart, no governance,
and no place in a larger company structure.

This module bridges both systems. Once installed, a Lumen instance becomes
a first-class employee in any Paperclip-managed company: it receives tasks,
reports status, and operates within the company's goals and budget.

```
Without this module:
  Paperclip ──── (silence) ────▶ Lumen
  Lumen     ──── (silence) ────▶ Paperclip

With this module:
  Paperclip ──── POST /paperclip/task ────▶ Lumen (executes)
  Lumen     ──── GET  /paperclip/report ──▶ Paperclip CEO (reads)
  Lumen     ────  heartbeat adapter   ────▶ Paperclip (registered agent)
```

---

## Module identity

```yaml
name: paperclip
version: 1.0.0
description: >
  Connect this Lumen instance to a Paperclip company as a registered agent.
  Receive tasks from the org chart, report status to the CEO,
  and operate within company goals and budgets.
tags: [integration, orchestration, enterprise, paperclip]
provides:
  - paperclip.agent
  - paperclip.task.receiver
  - paperclip.report.endpoint
requires:
  skills:
    - paperclip-awareness
  modules: []
x-lumen:
  tier: tier-2
  endpoints:
    - path: /paperclip/task
      method: POST
      auth: bearer
    - path: /paperclip/report
      method: GET
      auth: bearer
    - path: /paperclip/heartbeat
      method: POST
      auth: bearer
  on_install: on_install.py
  on_configure: on_configure.py
```

---

## Configuration (set via `lumen config set`)

```yaml
# Required
paperclip.url: "http://localhost:3100"     # Paperclip server URL
paperclip.api_key: ""                      # Paperclip API key
paperclip.company_id: ""                   # Company this agent belongs to
paperclip.agent_id: ""                     # This Lumen's agent ID in Paperclip
paperclip.agent_role: ""                   # Human-readable role (e.g. "Sales Agent")

# Optional
paperclip.report_fields: "all"             # Which fields to include in /report
paperclip.heartbeat_interval: 86400        # Seconds between self-reported heartbeats (default: 24h)
paperclip.task_timeout: 300                # Max seconds to process a task before timing out
```

**CLI setup example:**

```bash
lumen config set paperclip.url http://localhost:3100
lumen config set paperclip.api_key sk-paperclip-xxxxx
lumen config set paperclip.company_id company_abc123
lumen config set paperclip.agent_id agent_xyz789
lumen config set paperclip.agent_role "Sales Agent"
```

---

## Endpoints

### POST /paperclip/task

Paperclip sends a task to this Lumen instance. Lumen processes it through
its brain (with full personality + memory + skills context) and returns
a structured result.

**Request:**

```json
{
  "task_id": "task_abc123",
  "issue_id": "issue_xyz789",
  "title": "Analyze this week's pipeline and suggest next action",
  "description": "Full task description with context...",
  "goal": "Increase conversion rate from leads to paid customers",
  "priority": "high",
  "due": "2026-05-08T18:00:00Z",
  "context": {
    "company_goal": "Reach $10k MRR",
    "agent_role": "Sales Agent",
    "custom_data": {}
  }
}
```

**Processing flow:**

```
POST /paperclip/task
    ↓
Validate bearer token
    ↓
Build prompt:
  [paperclip-awareness skill]
  + task title + description + goal + context
  + Lumen's own memory (relevant facts)
    ↓
Send to brain (LLM call)
    ↓
Parse structured response
    ↓
Store task result in Lumen memory
    ↓
Return result to Paperclip
```

**Response:**

```json
{
  "ok": true,
  "task_id": "task_abc123",
  "status": "completed",
  "result": {
    "summary": "Brief summary of what was done or decided",
    "output": "Full response text from Lumen",
    "actions_taken": ["action 1", "action 2"],
    "blockers": [],
    "next_step": "Suggested next action for the CEO or team"
  },
  "tokens_used": 1240,
  "duration_ms": 3200,
  "timestamp": "2026-05-07T14:30:00Z"
}
```

**Error response:**

```json
{
  "ok": false,
  "task_id": "task_abc123",
  "status": "failed",
  "error": "description of what went wrong",
  "timestamp": "2026-05-07T14:30:00Z"
}
```

---

### GET /paperclip/report

The Paperclip CEO agent calls this endpoint on its daily heartbeat
to read the current state of this Lumen instance.

**Request:** GET with Bearer auth, no body.

**Optional query params:**

```
?period=7d        # Stats for last N days (default: 7d, options: 1d, 7d, 30d, all)
?fields=tasks,memory,status   # Comma-separated fields to include
```

**Response:**

```json
{
  "ok": true,
  "agent": {
    "id": "agent_xyz789",
    "role": "Sales Agent",
    "lumen_version": "0.3.0",
    "personality": "vendedor-editorial",
    "status": "online",
    "uptime_hours": 168
  },
  "tasks": {
    "received_total": 24,
    "completed": 22,
    "failed": 1,
    "pending": 1,
    "last_completed_at": "2026-05-07T12:00:00Z"
  },
  "memory": {
    "total_facts": 47,
    "total_notes": 12,
    "recent_activity": [
      "Lead María García classified as HOT - 2026-05-07",
      "Sent payment link to José Rodríguez - 2026-05-06"
    ]
  },
  "custom": {},
  "period": "7d",
  "timestamp": "2026-05-07T14:30:00Z"
}
```

The `custom` field is populated by the active personality or installed modules.
For example, a sales agent personality would add:

```json
"custom": {
  "leads_total": 47,
  "leads_hot": 8,
  "leads_closed": 6,
  "conversion_rate": "12.8%"
}
```

This allows any Lumen personality to expose domain-specific metrics
to Paperclip without changing the module code.

---

### POST /paperclip/heartbeat

Paperclip sends a periodic heartbeat to confirm the agent is alive
and to pass any pending directives.

**Request:**

```json
{
  "company_id": "company_abc123",
  "directives": [
    {
      "type": "goal_update",
      "content": "Focus on authors with completed manuscripts this week"
    }
  ]
}
```

**Response:**

```json
{
  "ok": true,
  "agent_id": "agent_xyz789",
  "status": "online",
  "directive_count": 1,
  "timestamp": "2026-05-07T14:30:00Z"
}
```

Directives are stored in Lumen memory so the brain can reference them
in subsequent conversations and tasks.

---

## SKILL: paperclip-awareness

**File:** `skills/paperclip-awareness/SKILL.md`

This skill is injected into the brain context whenever Lumen processes
a Paperclip task. It teaches Lumen how to behave as part of a company.

```markdown
# Paperclip awareness

## You are part of a company

You are a Lumen agent operating as an employee in a Paperclip-managed company.
This means you have:
- A role with specific responsibilities
- A CEO who assigns tasks and reads your reports
- Company goals that your work must serve
- A budget — your responses cost tokens, use them efficiently

## When you receive a task from Paperclip

1. Read the task title, description, and goal carefully
2. Check your memory for relevant context about this topic
3. Do the work — analyze, decide, write, or act as needed
4. Return a clear, structured result:
   - What you did or decided (summary)
   - Full output (your actual work product)
   - Any blockers you encountered
   - Suggested next step for the CEO or team

## How to write your report output

- Be direct. The CEO reads dozens of reports.
- Lead with the most important finding or decision.
- Flag blockers clearly — do not bury them.
- Suggest a concrete next action when you have one.
- Do not pad the response. Quality over length.

## What you do NOT do

- Do not make decisions outside your role without flagging it
- Do not spend tokens explaining obvious things
- Do not report "I completed the task" without actual output
- Do not fabricate data — if you don't know, say so
```

---

## on_install.py

Runs once when the module is installed. Registers this Lumen instance
as an agent in the Paperclip company.

```python
import requests

def on_install(config, lumen):
    """
    Called once after module installation.
    Registers this Lumen instance as an agent in Paperclip.
    """
    paperclip_url = config.get("paperclip.url")
    api_key = config.get("paperclip.api_key")
    company_id = config.get("paperclip.company_id")

    if not all([paperclip_url, api_key, company_id]):
        lumen.log("paperclip: skipping registration — config incomplete.")
        lumen.log("Run: lumen config set paperclip.url / api_key / company_id")
        return

    try:
        response = requests.post(
            f"{paperclip_url}/api/agents/register",
            json={
                "name": lumen.identity.name,
                "role": config.get("paperclip.agent_role", "Lumen Agent"),
                "company_id": company_id,
                "adapter": "http",
                "heartbeat_url": f"{lumen.base_url}/paperclip/heartbeat",
                "task_url": f"{lumen.base_url}/paperclip/task",
                "report_url": f"{lumen.base_url}/paperclip/report",
                "capabilities": lumen.registry.list_provides()
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )

        if response.ok:
            agent_id = response.json().get("agent_id")
            config.set("paperclip.agent_id", agent_id)
            lumen.log(f"paperclip: registered as agent {agent_id}")
        else:
            lumen.log(f"paperclip: registration failed — {response.status_code}")
            lumen.log("You can register manually via Paperclip dashboard.")

    except Exception as e:
        lumen.log(f"paperclip: could not reach Paperclip server — {e}")
        lumen.log("Configure paperclip.url and try: lumen reload")
```

---

## handler.py

Core request handlers for the three endpoints.

```python
import time
from datetime import datetime, timedelta

async def handle_task(request, config, lumen):
    """
    POST /paperclip/task
    Receives a task from Paperclip, processes it through the brain,
    returns structured result.
    """
    data = await request.json()
    task_id = data.get("task_id", "unknown")
    start = time.time()

    # Build the prompt
    prompt = f"""You are operating as part of a Paperclip-managed company.

TASK: {data.get('title', '')}

DESCRIPTION:
{data.get('description', '')}

COMPANY GOAL: {data.get('goal', 'Not specified')}

CONTEXT:
{data.get('context', {})}

Complete this task. Return your result in this exact format:
SUMMARY: one sentence describing what you did or decided
OUTPUT: your full work product or analysis
ACTIONS: list any concrete actions you took (one per line, or "none")
BLOCKERS: list any blockers encountered (one per line, or "none")
NEXT STEP: one concrete suggested next action"""

    try:
        response = await lumen.brain.process(
            message=prompt,
            session_id=f"paperclip-task-{task_id}",
            inject_skill="paperclip-awareness"
        )

        # Parse structured response
        result = parse_task_response(response)

        # Store in memory
        await lumen.memory.add_fact(
            f"Paperclip task '{data.get('title')}' completed. Result: {result['summary']}"
        )

        duration_ms = int((time.time() - start) * 1000)

        return {
            "ok": True,
            "task_id": task_id,
            "status": "completed",
            "result": result,
            "duration_ms": duration_ms,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {
            "ok": False,
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def handle_report(request, config, lumen):
    """
    GET /paperclip/report
    Returns current state of this Lumen instance to the Paperclip CEO.
    """
    period = request.query_params.get("period", "7d")
    days = {"1d": 1, "7d": 7, "30d": 30}.get(period, 7)
    since = datetime.now() - timedelta(days=days)

    # Base report
    report = {
        "ok": True,
        "agent": {
            "id": config.get("paperclip.agent_id", "unregistered"),
            "role": config.get("paperclip.agent_role", "Lumen Agent"),
            "lumen_version": lumen.version,
            "personality": lumen.personality.name if lumen.personality else "default",
            "status": "online",
        },
        "tasks": await get_task_stats(lumen, since),
        "memory": await get_memory_stats(lumen, since),
        "custom": await get_custom_stats(lumen),
        "period": period,
        "timestamp": datetime.now().isoformat()
    }

    return report


async def handle_heartbeat(request, config, lumen):
    """
    POST /paperclip/heartbeat
    Receives heartbeat from Paperclip. Stores any directives in memory.
    """
    data = await request.json()
    directives = data.get("directives", [])

    for directive in directives:
        if directive.get("type") == "goal_update":
            await lumen.memory.add_fact(
                f"[Paperclip directive] {directive.get('content')}",
                tags=["paperclip", "directive"]
            )

    return {
        "ok": True,
        "agent_id": config.get("paperclip.agent_id"),
        "status": "online",
        "directive_count": len(directives),
        "timestamp": datetime.now().isoformat()
    }


def parse_task_response(text):
    """Parse the structured brain output into result fields."""
    result = {
        "summary": "",
        "output": "",
        "actions_taken": [],
        "blockers": [],
        "next_step": ""
    }

    current = None
    lines = text.strip().split("\n")

    for line in lines:
        if line.startswith("SUMMARY:"):
            result["summary"] = line.replace("SUMMARY:", "").strip()
        elif line.startswith("OUTPUT:"):
            current = "output"
            result["output"] = line.replace("OUTPUT:", "").strip()
        elif line.startswith("ACTIONS:"):
            current = "actions"
        elif line.startswith("BLOCKERS:"):
            current = "blockers"
        elif line.startswith("NEXT STEP:"):
            result["next_step"] = line.replace("NEXT STEP:", "").strip()
            current = None
        elif current == "output" and line:
            result["output"] += "\n" + line
        elif current == "actions" and line and line != "none":
            result["actions_taken"].append(line.strip("- ").strip())
        elif current == "blockers" and line and line != "none":
            result["blockers"].append(line.strip("- ").strip())

    return result


async def get_task_stats(lumen, since):
    """Read task history from Lumen memory."""
    # Adapt to actual Lumen memory API
    facts = await lumen.memory.search("Paperclip task", since=since)
    return {
        "completed_in_period": len(facts),
        "last_activity": facts[-1].get("created_at") if facts else None
    }


async def get_memory_stats(lumen, since):
    """Read recent memory activity."""
    recent = await lumen.memory.recent(limit=5, since=since)
    return {
        "total_facts": await lumen.memory.count("facts"),
        "total_notes": await lumen.memory.count("notes"),
        "recent_activity": [f.get("content", "")[:80] for f in recent]
    }


async def get_custom_stats(lumen):
    """
    Hook for personality/modules to inject domain-specific metrics.
    Personalities can implement a paperclip_stats() method to populate this.
    """
    if hasattr(lumen.personality, "paperclip_stats"):
        try:
            return await lumen.personality.paperclip_stats()
        except Exception:
            pass
    return {}
```

---

## File structure

```
lumen/catalog/modules/paperclip/
├── module.yaml              # Module manifest
├── handler.py               # Endpoint handlers
├── on_install.py            # Registration hook
├── on_configure.py          # Re-registration on config change
├── README.md                # User-facing docs
└── skills/
    └── paperclip-awareness/
        └── SKILL.md         # Injected into brain on task processing
```

---

## README.md (user-facing)

```markdown
# Paperclip module for Lumen

Connect this Lumen instance to a Paperclip company.
Once installed, Lumen becomes a registered agent in your org chart —
it receives tasks, reports to the CEO, and operates within company goals.

## Install

lumen module install paperclip

## Configure

lumen config set paperclip.url http://your-paperclip-server:3100
lumen config set paperclip.api_key YOUR_PAPERCLIP_API_KEY
lumen config set paperclip.company_id YOUR_COMPANY_ID
lumen config set paperclip.agent_role "Your Agent Role"

## What you get

- POST /paperclip/task — Paperclip sends tasks, Lumen executes them
- GET  /paperclip/report — Paperclip CEO reads Lumen's status
- POST /paperclip/heartbeat — keeps the connection alive, delivers directives

## Custom metrics

If your active personality implements paperclip_stats(), those metrics
appear in the CEO's daily report under the "custom" field.
This lets any Lumen personality expose domain-specific data
(leads, sales, tasks completed, etc.) without changing this module.

## Requirements

- Lumen 0.3.0+
- A running Paperclip instance (paperclipai/paperclip)
- A Paperclip API key with agent registration permissions
```

---

## Open questions for implementation

These need resolution during development — not blocking the spec but
worth noting before writing the first line of code:

1. **Paperclip agent registration API** — does Paperclip have a
   `POST /api/agents/register` endpoint or does registration happen
   only through the dashboard? If no API exists, `on_install.py`
   should print manual instructions instead of auto-registering.

2. **Lumen brain API for skill injection** — `inject_skill` parameter
   in `lumen.brain.process()` may not exist yet. If not, the skill
   content should be prepended to the prompt directly.

3. **Memory API surface** — `lumen.memory.search()`, `.recent()`,
   and `.count()` need to match the actual memory module API.
   Check `lumen/core/memory.py` before implementing `handler.py`.

4. **lumen.base_url** — needed in `on_install.py` to pass the callback
   URLs to Paperclip. Confirm this is available in the install context
   or read it from config.

---

*lumen-paperclip — module spec v1.0*
*Ready to implement — give this doc to an agent and let it build*
