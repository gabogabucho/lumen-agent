# Tasks: Honcho Persistent Memory Integration

## Phase 1: Foundation

- [ ] 1.1 Create honcho module directory structure — Create `lumen/catalog/modules/honcho/` with `__init__.py` (exports `honcho` submodule + `__all__`), `module.yaml` (manifest with name `honcho-memory`, provides: honcho.memory/search/remember, requires: honcho-awareness skill, env specs for workspace_id/secret api_key/base_url/session_strategy/recall_mode), `SKILL.md` (module-level integration docs), `README.md` (setup instructions), and nested `skills/honcho-awareness/SKILL.md` (awareness skill). Pattern: mirror `lumen/catalog/modules/paperclip/` directory structure but adapted for honcho, not paperclip's handler-based split.
  - **Files**: `lumen/catalog/modules/honcho/__init__.py`, `module.yaml`, `SKILL.md`, `README.md`, `skills/honcho-awareness/SKILL.md`
  - **Config**: `module.yaml` must declare — name: honcho-memory, version 1.0.0, min_capability: tier-2, provides: [honcho.memory, honcho.search, honcho.remember], requires skills: [honcho-awareness], x-lumen.runtime.env with 5 env vars (workspace_id plain secret=false pattern: ".+", api_key secret=true, base_url optional plain, session_strategy choices: per-session/per-directory/per-repo/global, recall_mode choices: hybrid/context/tools).
  - **Edge cases**: `skills/honcho-awareness/SKILL.md` is a nested subdirectory inside the module — ensure relative paths use `skills/honcho-awareness/SKILL.md` not just the parent.

- [ ] 1.2 Create `lumen/catalog/modules/honcho/honcho.py` — Implement HonchoConfig dataclass (workspace_id: str, api_key: str|None, base_url: str|None, session_strategy: str, recall_mode: str, host: str|None, environment: str with from_config() static method reading _config["honcho"] flat keys and _config["secrets"]["honcho"]["api_key"] for secret). Implement HonchoClient class (_client: Honcho|None, connect() → Honcho|None, disconnect(), is_connected() → bool, register_peer() → str, search(session_key, query, max_tokens=800) → dict, fetch_context(session_key) → dict, write_conclusion(session_key, content, peer) → dict, store_memory(session, content, memory_type) → dict, build_context_block(session_key) → str with <context-block>... wrapper, inject_context(session_key, brain) → None per REQ-004, save_conclusions(session_key, session_data) → None per REQ-005, _call_with_timeout(method, timeout=30s) → response with 30s default and TimeoutException → 504). Implement module-level singleton state: `_honcho_client: HonchoClient|None = None`, `_honcho_ready: bool = False`. Expose `get_honcho_client()` (lazy init from config on first call, sets _honcho_ready=True if connected) and `reset_honcho_client()` (clears both globals, calls disconnect).
  - **Files**: `lumen/catalog/modules/honcho/honcho.py`
  - **Config pattern**: Use dotkey — `_config["honcho"]["workspace_id"]` for plain config, `_config["secrets"]["honcho"]["api_key"]` for secret. Auto-detect cloud vs self-hosted: if base_url is set/non-empty, route there; otherwise default to honcho.dev cloud.
  - **Edge cases**: honcho-ai SDK not installed → connect() must raise ImportError with clear message. All SDK calls wrapped in _call_with_timeout() → asyncio.TimeoutError maps to "gateway timeout" error. search() result truncates to max_tokens (default 800) if exceeding. register_peer() failure returns None and sets _honcho_ready=False.

- [ ] 1.3 Update `lumen/catalog/index.yaml` — Add honcho-memory module entry at end of modules list: name, display_name, description (in Spanish per catalog convention, e.g. "Integraci\u00f3n con Honcho.dev para memoria persistente cross-session"), version 1.0.0, author, min_capability tier-2, provides [honcho.memory, honcho.search, honcho.remember], requires skills: [honcho-awareness] modules: [], tags [integration, memory, cross-session, honcho].
  - **Files**: `lumen/catalog/index.yaml`
  - **Pattern**: Follow existing module entry format — use `description` field with proper YAML escaping for accented chars (`\u00f3` for ó).

## Phase 2: Module Lifecycle

- [ ] 2.1 Implement `on_install.py` — Create `lumen/catalog/modules/honcho/on_install.py` with `on_install(config, lumen_dir=None)`. Steps: (1) Validate config — check _config["honcho"].workspace_id is non-empty, _config["secrets"]["honcho"]["api_key"] exists, session_strategy and recall_mode are valid choices. (2) Install honcho-ai dependency via `pip install honcho-ai` subprocess call with try/except. (3) Create HonchoClient from config, call connect(). (4) Call register_peer() and store peer_id in _config["module_setup"]["honcho"]["peer_id"] with status "connected". (5) Test connectivity with list_sessions() call. (6) On failure at any step, print error message to stdout (bilingual: EN/ES), mark module_setup status as "error" with reason.
  - **Files**: `lumen/catalog/modules/honcho/on_install.py`
  - **Pattern**: Follow `lumen/catalog/modules/paperclip/on_install.py` — use config.get() pattern, try/except around network calls, print status messages.
  - **Edge cases**: honcho-ai already installed → pip install handles gracefully. Network unreachable → print specific error, don't crash. Missing config → print config instructions, return without error.

- [ ] 2.2 Implement `on_configure.py` — Create `lumen/catalog/modules/honcho/on_configure.py` with `on_configure(config, lumen_dir=None)`. Steps: (1) Call reset_honcho_client() to tear down existing connection. (2) Read updated config values (workspace_id, api_key, base_url, session_strategy, recall_mode). (3) Validate updated credentials. (4) Create new HonchoClient, connect(). (5) If peer_id changed (new workspace), update module_setup. (6) Log reconnection success.
  - **Files**: `lumen/catalog/modules/honcho/on_configure.py`
  - **Pattern**: Same as on_install.py but with reset + reconnect pattern rather than full install flow.
  - **Edge cases**: Config change removes api_key → connect fails → leave _honcho_ready=False. Config change removes workspace_id → validation catches and reports error.

## Phase 3: API Endpoints

- [ ] 3.1 Add `POST /honcho/search` endpoint to `lumen/channels/web.py` — Bearer token auth via `_validate_honcho_bearer_token(request)` + module installed guard (check honcho in installed modules) + brain ready check (`if not _brain: return JSONResponse 503`). Then dynamic import: `from lumen.catalog.modules.honcho.honcho import get_honcho_client`. Validate request body: query (str, required), max_tokens (int, optional, default 800). Call `client.search(session_key, query, max_tokens)`. Return result or error. Timeout: 30s via asyncio.wait_for. Error responses: 503 service_unavailable ("Honcho API unreachable" EN + "Servicio no disponible" ES), 504 gateway_timeout.
  - **Files**: `lumen/channels/web.py` (add ~30 lines)
  - **Pattern**: Exactly mirror paperclip endpoint pattern — auth → module guard → import → call → response.
  - **Code pattern**:
    ```python
    @app.post("/honcho/search")
    async def honcho_search(request: Request):
        # 1. Module guard
        module = _get_installed_module("honcho")
        if not module:
            return JSONResponse(status_code=503, content={"error": "honcho_module_not_installed", "message": "Modulo honcho no instalado. Ejecute: lumen module install honcho"})
        # 2. Bearer auth
        token = await _validate_honcho_bearer_token(request)
        if not token:
            return JSONResponse(status_code=401, content={"error": "bearer_token_required"})
        # 3. Brain ready
        if not _brain:
            return JSONResponse(status_code=503, content={"error": "service_unavailable", "message": "Lumen brain not ready"})
        # 4. Import & call
        from lumen.catalog.modules.honcho.honcho import get_honcho_client
        client = get_honcho_client()
        if not client.is_connected():
            return JSONResponse(status_code=503, content={"error": "service_unavailable", "message": "Honcho not connected"})
        data = await request.json() if request.headers.get("content-type") == "application/json" else {}
        query = data.get("query", "")
        max_tokens = min(data.get("max_tokens", 800), 2000)  # cap at 2000
        try:
            result = await asyncio.wait_for(_run_sync(client.search, session_key, query, max_tokens), timeout=30)
            return JSONResponse(content={"result": result.get("result", ""), "sessions": result.get("sessions", [])})
        except asyncio.TimeoutError:
            return JSONResponse(status_code=504, content={"error": "gateway_timeout", "message": "Timeout — honcho no responde"})
    ```
  - **Edge cases**: Missing query field → 400 Bad Request. max_tokens negative → default to 800. Empty result → return {"result": "", "sessions": []}. Bearer token mismatch → 401.

- [ ] 3.2 Add `GET /honcho/context` endpoint to `lumen/channels/web.py` — Auth guard chain same as search. Query params: `session_id` (required, str), `peer` (optional, str). Call `client.fetch_context(session_key)`. Return JSON: `{context: str, summary: ..., card: ..., representation: ..., recent: [...]}`. Session not found → empty context block.
  - **Files**: `lumen/channels/web.py` (add ~25 lines)
  - **Pattern**: GET endpoint with query params instead of JSON body, same guard chain.
  - **Edge cases**: Missing session_id → error response (not 404 — return empty context). Empty session → minimal context block, don't block session.

- [ ] 3.3 Add `POST /honcho/conclude` endpoint to `lumen/channels/web.py` — Bearer auth + module guard + brain check. Request body: content (str, required), peer (str, optional, default "user"). Truncate content to 25000 chars if longer (spec REQ-008). Call `client.write_conclusion(session_key, content[:25000], peer)`. Return `{"success": true}` or `{"success": false, "error": "..."}` on failure. No 400 on long content — just truncate.
  - **Files**: `lumen/channels/web.py` (add ~25 lines)
  - **Pattern**: POST with JSON body, but returns success/failure instead of error code on write failure (the SDK call may fail but we don't 503 — we report success:false).
  - **Edge cases**: content > 25000 chars → silently truncate, don't return error. Honcho API unreachable → return `{success: false, error: "..."}` instead of 503 (per spec: conclusion writes are fire-and-forget semantics even over HTTP).

- [ ] 3.4 Add `POST /honcho/memory` endpoint to `lumen/channels/web.py` — Bearer auth + module guard + brain check. Request body: content (str, required), session (str, optional), type (str, optional, default "fact"). Call `client.store_memory(session or session_key, content, memory_type)`. If session omitted → store without session association. Return `{"id": "...", "success": true}` or `{"success": false, "error": "..."}`.
  - **Files**: `lumen/channels/web.py` (add ~25 lines)
  - **Pattern**: POST with JSON body, optional session field.
  - **Edge cases**: Missing content → 400 Bad Request. session absent → no session association (not an error). Timeout → 504 (unlike conclude which is fire-and-forget, memory store is a direct request).

- [ ] 3.5 Add helper: `_validate_honcho_bearer_token(request)` — Create in `lumen/channels/web.py` a bearer token validator function for honcho endpoints. Checks `Authorization: Bearer <token>` header against configured honcho bearer token. Returns None if missing/invalid, valid token string if present.
  - **Files**: `lumen/channels/web.py` (add function)
  - **Pattern**: Mirror `lumen/api/paperclip/bearer.py` pattern. Check header existence → extract token → compare with stored token.
  - **Note**: If bearer token validation doesn't exist yet in web.py, create the pattern. If it exists for paperclip, adapt for honcho (same mechanism, different config key).

## Phase 4: Integration

- [ ] 4.1 Implement honcho-awareness skill — Create the awareness skill at `lumen/catalog/modules/honcho/skills/honcho-awareness/SKILL.md`. Content structure: (1) "What is Honcho" — persistent cross-session memory, semantic search, peer modelling. (2) "When to search" — before answering if question relates to past interactions, after user mentions something previously discussed. (3) "How to use honcho_search" — call `POST /honcho/search` with query+max_tokens, interpret results as relevance-ranked excerpts. (4) "How to use honcho_conclude" — after completing tasks, write learned facts with `POST /honcho/conclude`. (5) "Context injection" — access `GET /honcho/context` for past state before thinking, use session_id. (6) "Memory store" — direct memory writes with `POST /honcho/memory`, use session and type fields. (7) "Memory retention best practices" — what to store (facts, preferences, learned patterns), what to skip (temporary noise, redundant info), summary vs detail tradeoffs. (8) "Error awareness" — if Honcho is down (503/504), continue without memory, don't fail the task. Use tool names `honcho_search` / `honcho_conclude` / `honcho_context` / `honcho_memory` per REQ-007/REQ-008.
  - **Files**: `lumen/catalog/modules/honcho/skills/honcho-awareness/SKILL.md`
  - **Pattern**: Follow other Lumen skill format (title, description, when to apply, tool usage details).
  - **Edge cases**: Skill must handle both cloud and self-hosted modes — mention that the AI uses endpoint URLs directly.

- [ ] 4.2 Implement memory sync hooks — Add context injection and conclusion writing integration in `honcho.py` client: (a) `inject_context(session_key, brain)`: fetches context block via `build_context_block(session_key)`, wraps in `<context-block>...</context-block>` XML, passes to `brain.set_context()` or equivalent to inject into session thought. Hook called before session read. (b) `save_conclusions(session_key, session_data)`: extracts conclusions from session_data (summary, interactions, tool usage), calls `write_conclusion(session_key, content, peer)` for each. Hook called after session write. (c) Both methods are fire-and-forget: if the SDK call fails, log warning but don't raise. (d) If _honcho_ready=False, skip both hooks silently.
  - **Files**: `lumen/catalog/modules/honcho/honcho.py` (add methods to HonchoClient) and potentially `lumen/core/brain.py` or wherever sessions are read/written (if brain hooks need integration points).
  - **Pattern**: Non-blocking — use try/except around SDK calls, log to lumen logger. Follow REQ-004 (context injection) and REQ-005 (conclusion writes).
  - **Edge cases**: brain.set_context() might not exist yet — check existing brain API; if not, add a simple brain.set_context(text) method. session_data format varies — handle both dict with "summary" key and raw string formats. Failed conclusion write → log with `logger.warning("honcho: conclusion write failed, continuing")` but don't crash.
