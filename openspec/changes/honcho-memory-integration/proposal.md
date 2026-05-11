# Proposal: Honcho Persistent Memory Integration

## Intent

Lumen has no cross-session persistent memory. Every conversation starts from zero — facts, preferences, and context are lost when a session ends. This prevents Lumen from remembering users across platforms, learning from past interactions, or providing personalized responses over time.

Honcho (honcho.dev) is a cloud-based AI memory service providing semantic search, peer modelling, dialectic reasoning, and persistent conclusions. We integrate Honcho as a first-class Lumen module so Lumen can remember, recall, and reason from accumulated knowledge across sessions.

## Scope

### In Scope
- New `lumen/catalog/modules/honcho/` module with manifest, lifecycle hooks (`on_install.py`, `on_configure.py`), awareness skill, and SDK runtime
- `honcho-ai` SDK integration: client creation, session management, semantic search, memory sync (read context on session start, write conclusions at session end)
- 4 new REST endpoints in `lumen/channels/web.py`: `POST /honcho/search`, `GET /honcho/context`, `POST /honcho/conclude`, `POST /honcho/memory` — all guarded by bearer token auth + module readiness check
- `on_install.py`: install `honcho-ai` dependency, register Lumen as an AI peer with Honcho
- `on_configure.py`: connect to Honcho API, validate workspace credentials, test connectivity
- `honcho-awareness` skill: gives AI agent instructions to recall, search, and write memories
- Catalog entry in `lumen/catalog/index.yaml` adding honcho module with `provides` and `requires` fields

### Out of Scope
- Honcho server deployment or self-hosted infrastructure (the `base_url` config key is planned for future, not implemented in this PR)
- Honcho SDK development or modifications to the external `honcho-ai` package
- Memory migration from Lumen's existing local memory stores
- Multi-agent memory sharing or distributed peer-to-peer memory between Lumen instances

## Approach

1. **Module scaffold**: Create `lumen/catalog/modules/honcho/` with `module.yaml`, `__init__.py`, `SKILL.md`, `README.md`, `on_install.py`, `on_configure.py`, and `skills/honcho/SKILL.md`.
2. **SDK client**: In `__init__.py`, provision a `HonchoClient` wrapper that reads `HONCHO_WORKSPACE_ID`, `HONCHO_API_KEY`, and `HONCHO_BASE_URL` (if set) from config. The client exposes methods: `search(query, limit)`, `get_context(session_id)`, `conclude(text)`, `add_memory(text, tags)`, `peer_card()`.
3. **Memory sync hooks**: The module's `on_configure.py` validates the connection and creates a peer identity. At runtime, before each `brain.think()` call, the awareness skill instructs the agent to call `/honcho/context` to inject relevant memories into its reasoning. After a session concludes, `/honcho/memory` or `/honcho/conclude` saves new facts.
4. **REST endpoints**: Each Honcho endpoint in `web.py` follows the established pattern: `_validate_bearer_token()` guard → `_check_module_enabled("honcho")` check → auth → body validation → handler → `JSONResponse`. Endpoints delegate to the honcho client layer.
5. **Catalog registration**: Add a honcho entry to `index.yaml` with `provides: [honcho.memory, honcho.search, honcho.remember]`, `requires: {skills: [honcho-awareness]}`, `min_capability: tier-1`, and `tags: [memory, ai, persistence]`.
6. **Config storage**: Workspace ID and strategy in flat keys (`honcho.workspace_id`, `honcho.session_strategy`, `honcho.recall_mode`, `honcho.base_url`). API key stored in `_config["secrets"]["honcho"]["api_key"]`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `lumen/catalog/modules/honcho/` | New | Full module directory: manifest, hooks, skill, runtime client |
| `lumen/channels/web.py` | Modified | 4 new REST endpoints with bearer auth and module guard |
| `lumen/catalog/index.yaml` | Modified | Add honcho catalog entry with provides/requires/tags |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| API rate limits from honcho.dev | Medium | Client respects rate limits; endpoints return 429 with retry-after; graceful degradation (module unavailable → continue without memory) |
| Network dependency on Lumen's core flow | High | Honcho calls are fire-and-forget or timeout-bounded (5s). If Honcho is unreachable, Lumen continues normally — just without cross-session memory. |
| Credential exposure / leakage | Low | API key stored in secrets store (`_config["secrets"]["honcho"]`). Endpoints never echo credentials in responses. Bearer token auth required for all Honcho endpoints. |
| SDK compatibility / `honcho-ai` breaking changes | Low | Pin SDK version in `on_install.py`. Vendor a frozen copy of the SDK interface if changes occur. |

## Rollback Plan

1. Remove `/honcho/*` endpoints from `web.py` (one file, 4 endpoints, ~80 lines).
2. Delete `lumen/catalog/modules/honcho/` directory and entry from `index.yaml`.
3. Uninstall `honcho-ai` dependency (removed from `on_install.py` or via `pip uninstall`).
4. No schema changes, no database migrations, no config key removals needed (module config is namespaced under `honcho.*`).
5. Safe to revert by restoring previous `web.py` and `index.yaml` from git.

## Dependencies

- External: `honcho-ai` Python package (>=1.0.0)
- Workspace credentials: a Honcho workspace with a valid API key
- Lumen capability: `honcho.remember` requires `memory` connector (inherited from `tier-1`)

## Success Criteria

- [ ] `lumen catalog install honcho` installs the module and all dependencies without errors
- [ ] `lumen configure honcho` validates workspace credentials and reports success
- [ ] POST `/honcho/search` returns semantic search results from Honcho workspace
- [ ] GET `/honcho/context` returns relevant context summary for a session
- [ ] POST `/honcho/conclude` persists a new conclusion/fact to Honcho
- [ ] POST `/honcho/memory` stores new memory with optional tags
- [ ] All 4 endpoints reject unauthenticated requests with 401
- [ ] All 4 endpoints reject requests when honcho module is not installed with error
- [ ] Lumen continues to operate normally if Honcho API is unreachable (no crash, no hung requests)
- [ ] Awareness skill instructions are present in `skills/honcho/SKILL.md` and recognized by the registry
