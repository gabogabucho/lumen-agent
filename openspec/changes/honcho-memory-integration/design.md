# Design: Honcho Persistent Memory Integration

## Technical Approach

Integrate Honcho.dev as Lumen's cross-session persistent memory layer by creating a new `honcho` module under `lumen/catalog/modules/honcho/` that wraps the `honcho-ai` SDK. The module installs `honcho-ai`, registers Lumen as an AI peer, and exposes 4 REST endpoints (`/honcho/search`, `/honcho/context`, `/honcho/conclude`, `/honcho/memory`) following the paperclip pattern in `lumen/channels/web.py`. Memory sync hooks inject context before session thought and write conclusions after session end. An awareness skill instructs the Lumen agent on when to recall, search, and persist knowledge.

Cloud vs self-hosted mode is auto-detected from config — if `honcho.base_url` is set, client routes there; otherwise defaults to cloud `honcho.dev`. All endpoints guard with bearer token auth + module installed check + brain readiness, returning 503/401/504 per spec.

## Architecture Decisions

### Decision: HonchoClient as module-level singleton with lazy init

**Choice**: A `HonchoClient` class stored in `honcho.py` module globals (`_honcho_client`, `_honcho_ready`) exposed via `get_honcho_client()`. Lazy init on first endpoint hit, re-created on reconfigure.

**Alternatives considered**: Separate client module (`honcho/client.py`), or shared state in `lumen/core/`.

**Rationale**: Paperclip follows the same pattern (`handler.py` with standalone module). Keeps honcho isolated in its own module directory. Lazy init avoids SDK import cost at startup if honcho is never installed. `reset_honcho_client()` for explicit teardown.

### Decision: Module-level honcho.py as both client and API handler

**Choice**: `honcho.py` exports `HonchoClient` for direct use AND `get_honcho_client()` for lazy singleton. Endpoints in `web.py` do a dynamic import: `from lumen.catalog.modules.honcho import honcho`.

**Alternatives considered**: Separate `honcho.py` (client) + `honcho/handler.py` (endpoints), like paperclip's split.

**Rationale**: Honcho endpoints are simpler (search, context, conclude, store — no brain pipeline). A single file reduces coupling. Paperclip's `handler.py` complexity comes from brain interaction; honcho is a straight SDK wrapper + API endpoints.

### Decision: Config namespacing under `honcho.*` dotkeys

**Choice**: `honcho.workspace_id`, `honcho.api_key` (secret), `honcho.base_url`, `honcho.session_strategy`, `honcho.recall_mode`. Secrets stored in `_config["secrets"]["honcho"]["api_key"]`.

**Alternatives considered**: HONCHO_ prefixed environment variables only, or flat `workspace_id` / `api_key` without namespace.

**Rationale**: Matches existing paperclip pattern (`paperclip.url`, `paperclip.agent_id`, `secrets.paperc`lip`). Namespace avoids collisions. Dotkey convention is established throughout lumen.

### Decision: Sync endpoint calls — no async queue for Honcho writes

**Choice**: Search and store endpoints use synchronous HTTP calls with `asyncio.wait_for` + 30s timeout. No background worker needed for the MVP.

**Alternatives considered**: Background write queue with exponential backoff, fire-and-forget for conclusion writes.

**Rationale**: Spec REQ-011 says timeouts, 503, 504 — which implies synchronous handling. The awareness skill instructs the Lumen agent to call `/honcho/conclude` as a tool, which is an active request — not a background process. Queue adds unnecessary complexity for v1; can be added later if needed.

### Decision: Awareness skill as LLM prompt instructions, not executable code

**Choice**: `skills/honcho-awareness/SKILL.md` contains natural-language instructions for the Lumen brain: when to search, how to write conclusions, what memory retention policies apply. No Python agent for the skill.

**Alternatives considered**: Python-based skill agent, or a separate process.

**Rationale**: Consistent with all other Lumen skills (paperclip-awareness, scheduler, etc.). Skills are injected as system prompt context — the brain reads the SKILL.md and follows instructions.

## Data Flow

```
External Client ──→  /honcho/search ──→ web.py guard ──→ honcho.search() ──→ honcho-ai SDK ──→ Honcho Cloud
External Client ──→  /honcho/context ──→ web.py guard ──→ honcho.fetch_context() ──→ Honcho Cloud
External Client ──→  /honcho/conclude ──→ web.py guard ──→ honcho.write_conclusion() ──→ Honcho Cloud
External Client ──→  /honcho/memory ──→ web.py guard ──→ honcho.store_memory() ──→ Honcho Cloud

Lumen Brain ──→ (awareness skill) ──→ agent calls /honcho/context ──→ context injected into brain.think()
Lumen Brain ──→ (awareness skill) ──→ agent calls /honcho/conclude ──→ conclusion written to Honcho
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `lumen/catalog/modules/honcho/__init__.py` | Create | Package init — imports and exports `honcho` submodule, exposes `__all__` |
| `lumen/catalog/modules/honcho/honcho.py` | Create | HonchoClient class, config, module-level singleton (`get_honcho_client()`, `reset_honcho_client()`) |
| `lumen/catalog/modules/honcho/module.yaml` | Create | Module manifest — name, provides (honcho.memory, honcho.search, honcho.remember), requires, env specs |
| `lumen/catalog/modules/honcho/SKILL.md` | Create | Module-level documentation describing the integration |
| `lumen/catalog/modules/honcho/README.md` | Create | Setup instructions for users |
| `lumen/catalog/modules/honcho/on_install.py` | Create | Install hook — pip install honcho-ai, register peer, store peer_id, validate connection |
| `lumen/catalog/modules/honcho/on_configure.py` | Create | Reconfigure hook — reconnect with updated credentials, validate, update peer_id |
| `lumen/catalog/modules/honcho/skills/honcho-awareness/SKILL.md` | Create | Awareness skill — natural-language instructions for the Lumen agent |
| `lumen/channels/web.py` | Modify | 4 new REST endpoints following paperclip pattern |
| `lumen/catalog/index.yaml` | Modify | Add honcho-memory catalog entry |

## Interfaces / Contracts

### honcho.py — Client API

```python
# Module-level state (honcho.py)
_honcho_client: HonchoClient | None = None
_honcho_ready: bool = False

def get_honcho_client() -> HonchoClient | None:
    """Lazy singleton access. Creates HonchoClient from config on first call."""

def reset_honcho_client():
    """Explicit teardown — called on reconfigure or module uninstall."""

# Configuration
class HonchoConfig:
    workspace_id: str
    api_key: str | None          # secret
    base_url: str | None         # self-hosted (None → cloud)
    session_strategy: str        # per-session | per-directory | per-repo | global
    recall_mode: str             # hybrid | context | tools
    host: str | None             # auto-detected from _config["lumen"]["host"]
    environment: str             # "production" or "local"

    @staticmethod
    def from_config(config: dict) -> HonchoConfig

# Client class
class HonchoClient:
    config: HonchoConfig
    _client: Honcho | None       # SDK instance

    def connect(self) -> Honcho        # honcho-ai Honcho(workspace_id, api_key, environment, base_url)
    def disconnect(self) -> None
    def is_connected(self) -> bool
    def register_peer(self) -> str     # Returns peer_id string
    def search(self, session_key: str, query: str, max_tokens: int = 800) -> dict
    def fetch_context(self, session_key: str) -> dict
    def write_conclusion(self, session_key: str, content: str, peer: str) -> dict
    def store_memory(self, session: str, content: str, memory_type: str) -> dict
    def build_context_block(self, session_key: str) -> str  # <context-block>...</context-block>
    def inject_context(self, session_key: str, brain) -> None  # REQ-004
    def save_conclusions(self, session_key: str, session_data) -> None  # REQ-005
```

### Endpoint Request/Response Contracts

| Endpoint | Method | Auth | Request | Response |
|----------|--------|------|---------|----------|
| `/honcho/search` | POST | Bearer + module | `{"query": "...", "max_tokens": 800}` | `{"result": "...", "sessions": [...]}` |
| `/honcho/context` | GET | Bearer + module | Query: `session_id`, `peer` | `{"context": "...", "summary": ..., "card": ..., "representation": ..., "recent": [...]}` |
| `/honcho/conclude` | POST | Bearer + module | `{"content": "...", "peer": "user"}` | `{"success": true}` or `{"success": false, "error": "..."}` |
| `/honcho/memory` | POST | Bearer + module | `{"content": "...", "session": "sess123", "type": "fact"}` | `{"id": "...", "success": true}` |

### web.py Guard Chain (paperclip pattern)

```python
@app.post("/honcho/search")
async def honcho_search(request):
    # 1. Module guard — check honcho installed & ready
    # 2. Bearer auth — _validate_honcho_bearer_token(request)
    # 3. Brain ready check — if not _brain: return 503
    # 4. Dynamic import — from lumen.catalog.modules.honcho.honcho import get_honcho_client
    # 5. Client call — client.search(query, max_tokens)
    # 6. JSONResponse
```

### module.yaml Manifest

```yaml
name: honcho-memory
display_name: "Honcho Persistent Memory"
description: "Cross-session persistent memory via Honcho.dev — semantic search, recall, and learn from past interactions."
version: 1.0.0
author: "Lumen Team"
price: free
min_capability: tier-2
provides:
  - honcho.memory
  - honcho.search
  - honcho.remember
requires:
  skills:
    - honcho-awareness
  modules: []
x-lumen:
  runtime:
    env:
      # Plain config
      - name: workspace_id
        secret: false
        pattern: ".+"
      - name: base_url          # optional — self-hosted
        secret: false
      - name: session_strategy  # per-session|per-directory|per-repo|global
        secret: false
      - name: recall_mode       # hybrid|context|tools
        secret: false
      # Secret
      - name: api_key
        secret: true
```

### index.yaml Entry

```yaml
- name: honcho-memory
  display_name: "Honcho Persistent Memory"
  description: "Integraci\u00f3n con Honcho.dev para memoria persistente cross-session \u2014 recuperar hechos, aprender de interacciones pasadas"
  version: 1.0.0
  author: "Lumen Team"
  price: free
  min_capability: tier-2
  provides:
    - honcho.memory
    - honcho.search
    - honcho.remember
  requires:
    skills:
      - honcho-awareness
    modules: []
  tags: [integration, memory, cross-session, honcho]
```

### Config Storage

```yaml
# Plain config keys (_config["honcho"]):
honcho:
  workspace_id: "ws_abc123"
  session_strategy: "per-session"
  recall_mode: "hybrid"
  base_url: ""                    # empty → cloud mode

# Secrets store (_config["secrets"]["honcho"]):
secrets:
  honcho:
    api_key: "hk_live_xxxxxx"

# Module setup (_config["module_setup"]["honcho"]):
module_setup:
  honcho:
    peer_id: "peer_lumen_xyz"
    status: "connected"           # "connected" | "error" | "pending"
    connected: true
```

### Awareness Skill (honcho-awareness/SKILL.md) Structure

The SKILL.md contains these sections:
1. **What is Honcho** — persistent cross-session memory, semantic search
2. **When to search** — before answering, when user asks about past interactions
3. **How to use honcho_search** — call `POST /honcho/search`, interpret results
4. **How to use honcho_conclude** — after completing tasks, write learned facts
5. **Context injection** — use `GET /honcho/context` to get past state before thinking
6. **Memory store** — direct memory writes with `POST /honcho/memory`
7. **Memory retention best practices** — what to store, what to skip, summary vs detail tradeoffs
8. **Error awareness** — if Honcho is down, continue without memory; don't fail the task

## Error Handling Matrix

| Scenario | Status | Response |
|----------|--------|----------|
| Module not installed | 503 | `{"error": "honcho_module_not_installed", "message": "..."}` |
| Invalid bearer token | 401 | `{"error": "bearer_token_required"}` |
| Honcho API unreachable | 503 | `{"error": "service_unavailable", "message": "Honcho API unreachable"}` |
| Honcho API >30s | 504 | `{"error": "gateway_timeout", "message": "Honcho API did not respond in time"}` |
| Auth error (module guard fails first) | 403 (via paperclip bearer guard) | `{"error": "invalid_bearer_token"}` |
| Graceful degradation | N/A | Module available but `_honcho_ready=False` → endpoints return 503 with "not connected" message |

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `HonchoConfig.from_config()` with valid/invalid config | Mock config dict, assert field validation |
| Unit | `HonchoClient.connect()` with mock SDK | Patch `honcho-ai Honcho`, verify constructor args |
| Unit | `HonchoClient.is_connected()` state transitions | Test None → connected → disconnected |
| Unit | `HonchoClient.search()` truncation to max_tokens | Verify text slicing |
| Unit | `HonchoClient.build_context_block()` format | Verify XML-like `<context-block>` wrapper |
| Unit | `on_install.py` with missing config | Assert "config incomplete" message, no exception |
| Unit | `on_configure.py` with invalid credentials | Assert error message, no crash |
| Integration | Endpoint with valid token → mock HonchoClient | Mock `get_honcho_client()` returns fake client, verify JSONResponse |
| Integration | Endpoint with invalid token → 401 | Send request without Bearer, verify 401 |
| Integration | Endpoint with module not installed → 503 | Set `_honcho_modules_installed=False`, verify guard fires |
| Integration | Endpoint with _brain=None → 503 | Remove _brain, verify guard fires before import |
| E2E | Full install flow: `lumen module install honcho` → config → endpoint call | Spin up test Lumen instance, verify module appears in `/api/modules/installed` |
| E2E | Cloud mode → connect to honcho.dev | Set `base_url=""`, verify cloud URL used |
| E2E | Self-hosted mode → connect to custom URL | Set `base_url="https://honcho.internal"`, verify routing |

## Migration / Rollout

No migration required. The module is additive — existing Lumen functionality is unaffected. Rollout:

1. Ship honcho module and endpoints in same release
2. Endpoints are inert until module is installed AND `get_honcho_client()` returns a connected client
3. If `honcho-ai` package is not found on import, endpoints return 503 `module_not_installed`
4. Safe rollout: endpoints coexist with any existing config; non-honcho users never hit honcho code paths
5. Rollback: remove 4 endpoints + honcho module directory + index.yaml entry — no schema changes

## Open Questions

- [ ] Session key generation strategy: What format should `session_key` follow? UUID? Timestamp? User-provided?
- [ ] Should `/honcho/context` GET also support a POST variant with query params (the spec mentions search + context)?
- [ ] `max_tokens` validation: Should we enforce the 800 default/2000 max on the client side or just pass through to Honcho SDK?
- [ ] Should the awareness skill mention the specific endpoint URLs directly, or abstract them as `honcho_search` / `honcho_conclude` tool names? (The spec uses tool names in REQ-007/REQ-008.)
