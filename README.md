<p align="center">
  <img src="logo.png" alt="Lumen" width="180" />
</p>

<h1 align="center">Lumen</h1>

<p align="center">
  <strong>Open-source AI agent engine. Modular. No limits.</strong>
</p>

<p align="center">
  <em>"An agent you can shape without code."</em>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#docker--docker-compose">Docker</a> &bull;
  <a href="#dokploy-deployment">Dokploy</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#manifesto">Manifesto</a> &bull;
  <a href="MANIFESTO.md">Full Manifesto</a> &bull;
  <a href="LUMEN_SPEC.md">Spec</a> &bull;
  <a href="CHANGELOG.md">Changelog</a> &bull;
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

## What is Lumen?

Lumen is a **downloadable AI agent framework** that works from minute zero. Install it, run it, and you have a working assistant. From there, shape it however you want: pick a personality, install modules, plug in connectors, swap providers. No code required for everyday use.

**Not a SaaS. Not a platform. Not a chatbot.** A framework you own and run on your machine.

**Think WordPress, but for AI agents.**

- Install it → working assistant
- Pick a personality → different behavior, same core
- Install a module → new capability or integration
- Bring your own module → load any custom `module.yaml`

---

## Quickstart

```bash
pip install enlumen
lumen run
```

Your browser opens at:

```txt
http://localhost:3000
```

First time? The setup wizard walks you through three paths:

1. **Quick start** — default personality + free OpenRouter model.
2. **Choose a personality** — browse the catalog and pick one that matches your use case.
3. **Bring your own module** — upload a custom `module.yaml` to configure Lumen your way.

After that, Lumen awakens and you land directly in the chat. The sidebar gives you:

```txt
Charlas / Módulos / Memoria / Ajustes
```

No separate admin panel. No dev jargon.

### From source

```bash
git clone https://github.com/gabogabucho/lumen-agent.git
cd lumen-agent
pip install -e .[dev]
lumen run
```

---

## `lumen run` vs `lumen server`

Lumen has two startup modes depending on where it runs.

### `lumen run`

Use this for:

- local development
- personal use on your own computer
- quick UI and module testing

```bash
lumen run
```

### `lumen server`

Use this for:

- a VPS
- a remote server
- a home server / always-on machine
- any installation that should stay available over the network

```bash
lumen server --host 0.0.0.0 --port 3000
```

Behavior:

- starts Lumen as a hosted web service
- exposes onboarding through IP/domain + port
- first setup is protected with a one-time setup token
- onboarding creates the owner password/PIN
- future access to the dashboard requires login

Rule of thumb:

```txt
lumen run    = local/personal development
lumen server = hosted remote service
```

---

## Docker / Docker Compose

Lumen can run as a containerized service using Docker Compose.

This is useful for:

- VPS deployments
- Dokploy
- Coolify
- CapRover
- Portainer
- internal infrastructure
- long-running agent services

### Dockerfile

Create a `Dockerfile` in the repository root:

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Minimal runtime dependencies for TLS/certs and basic health tooling.
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates curl \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY lumen /app/lumen

RUN pip install --no-cache-dir .

EXPOSE 3000

# Persist Lumen instance data in /root/.lumen using a Docker volume.
CMD ["lumen", "server", "--host", "0.0.0.0", "--port", "3000"]
```

### Basic `docker-compose.yml`

Use this for a simple local/VPS Docker deployment:

```yaml
services:
  lumen:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - lumen_data:/root/.lumen
    restart: unless-stopped

volumes:
  lumen_data:
```

Run:

```bash
docker compose up -d --build
```

Check logs:

```bash
docker compose logs -f lumen
```

Health check:

```bash
curl http://localhost:3000/health
```

Expected response:

```json
{"ok": true}
```

The response may also include version, ready module count, model and provider status.

---

## First Docker setup

On a fresh Docker volume, Lumen may need initial model/provider configuration before the server can fully start.

If the container keeps restarting and logs show something like:

```txt
¿Qué modelo querés usar?
1. DeepSeek
2. OpenAI GPT-4o-mini
3. Anthropic Claude
4. Ollama
5. OpenRouter
Aborted.
```

it means Lumen is waiting for the first interactive setup, but Docker cannot answer prompts in detached mode.

### Fix: run one interactive setup against the same volume

1. Find the container name:

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep lumen
```

Example:

```txt
my-project-lumen-1   Restarting (1) 30 seconds ago
```

2. Save container, image and volume names:

```bash
C=my-project-lumen-1

IMG=$(docker inspect -f '{{.Config.Image}}' "$C")
VOL=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/root/.lumen"}}{{.Name}}{{end}}{{end}}' "$C")

echo "IMG=$IMG"
echo "VOL=$VOL"
```

3. Stop the restart loop:

```bash
docker update --restart=no "$C"
docker stop "$C"
```

4. Run setup interactively using the same volume:

```bash
docker run --rm -it \
  -v "$VOL":/root/.lumen \
  "$IMG" \
  lumen server --host 0.0.0.0 --port 3000
```

5. Choose your provider/model.

Example choices:

```txt
2 = OpenAI GPT-4o-mini
5 = OpenRouter
```

6. When the configuration is saved and you see the server setup token, stop the temporary process with:

```txt
CTRL + C
```

The config is now stored in the Docker volume.

7. Re-enable restart policy and start again:

```bash
docker update --restart=unless-stopped "$C"
docker start "$C"
```

8. Verify:

```bash
curl http://localhost:3000/health
```

---

## Dokploy deployment

Use **Compose**, not **Application**, when deploying Lumen to Dokploy.

Recommended Dokploy settings:

```txt
Create Service → Compose
Repository: your-lumen-agent-repo
Branch: main
Compose Path: ./docker-compose.yml
```

Domain settings:

```txt
Domain: your-domain.example.com
Service Name: lumen
Container Port: 3000
Internal Path: /
Strip Path: OFF
HTTPS: ON
```

### Dokploy-compatible `docker-compose.yml`

Use this when Lumen must be reachable through Dokploy/Traefik and also be able to talk to other internal projects later.

```yaml
services:
  lumen:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - lumen_data:/root/.lumen
    networks:
      - neuron-internal
      - dokploy-network
    restart: unless-stopped
    labels:
      - "traefik.http.middlewares.lumen-ratelimit.ratelimit.average=60"
      - "traefik.http.middlewares.lumen-ratelimit.ratelimit.period=1m"
      - "traefik.http.middlewares.lumen-ratelimit.ratelimit.burst=30"

      - "traefik.http.middlewares.lumen-secure-headers.headers.stsSeconds=31536000"
      - "traefik.http.middlewares.lumen-secure-headers.headers.stsIncludeSubdomains=true"
      - "traefik.http.middlewares.lumen-secure-headers.headers.stsPreload=true"
      - "traefik.http.middlewares.lumen-secure-headers.headers.contentTypeNosniff=true"
      - "traefik.http.middlewares.lumen-secure-headers.headers.browserXssFilter=true"
      - "traefik.http.middlewares.lumen-secure-headers.headers.referrerPolicy=no-referrer-when-downgrade"

volumes:
  lumen_data:

networks:
  neuron-internal:
    external: true
  dokploy-network:
    external: true
```

Create the shared internal network once on the server:

```bash
docker network create neuron-internal
```

If the network already exists, Docker will print an error. That is safe to ignore.

### Why no `ports:`?

For Dokploy, do not expose host ports unless you need temporary debugging.

Correct for production:

```yaml
# no ports needed
```

Dokploy/Traefik routes traffic through `dokploy-network` to the internal container port:

```txt
Container Port: 3000
```

This avoids host port conflicts when several projects use port `3000` internally.

If you need temporary direct access, you can add:

```yaml
ports:
  - "3110:3000"
```

Then remove it after verifying the domain works.

### Dokploy middlewares

If you added the labels above, add these middleware references in the Dokploy domain panel:

```txt
lumen-ratelimit@docker
```

and:

```txt
lumen-secure-headers@docker
```

Recommended starting values:

```txt
60 requests / minute
burst 30
```

This is enough for normal dashboard/API use and helps protect setup, login, `/health` and `/api/chat`.

After deployment, verify:

```bash
curl -I https://your-domain.example.com/health
```

Expected:

```txt
HTTP/2 200
```

or:

```json
{"ok": true}
```

---

## Security model

In server mode:

- setup token is generated once and shown only in logs/console
- setup token is used to create the owner password/PIN
- after setup, the token is deleted
- owner PIN is hashed using PBKDF2-SHA256
- session cookies are signed, `httponly`, `samesite: lax`
- WebSocket access requires the same authenticated owner cookie
- REST API endpoints require Bearer authentication where applicable

Do not expose the setup token publicly.

Do not commit secrets or generated API keys.

Use HTTPS in production.

---

## REST API

Lumen exposes a REST API for external integrations.

### Health check

No auth required:

```bash
curl http://localhost:3000/health
```

Example response:

```json
{
  "ok": true,
  "version": "1.2.0",
  "modules_ready": 1,
  "model": "openrouter/openai/gpt-oss-120b:free",
  "provider_status": "healthy"
}
```

### Chat

Bearer auth required:

```bash
curl -X POST http://localhost:3000/api/chat \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"message": "hello", "session_id": "optional-session-id"}'
```

Example response:

```json
{
  "response": "Hello! How can I help?",
  "session_id": "optional-session-id"
}
```

### Reload runtime

Bearer auth required:

```bash
curl -X POST http://localhost:3000/api/reload \
  -H "Authorization: Bearer your-api-key"
```

Example response:

```json
{"status": "reloaded", "modules": 5}
```

Auth sources are checked in this order:

```txt
LUMEN_API_KEY env var → config.api.rest_key → api_keys.yaml hashed keys
```

---

## API key management

Generate a new API key:

```bash
lumen api-key generate --label "my app"
```

List keys:

```bash
lumen api-key list
```

Revoke a key by prefix:

```bash
lumen api-key revoke <prefix>
```

Inside Docker:

```bash
docker exec -it <lumen-container> lumen api-key generate --label "n8n"
```

Use the generated key in n8n as:

```txt
Authorization: Bearer <your-api-key>
```

The full key is shown only once.

---

## n8n integration pattern

Lumen works well as an agent runtime behind n8n.

Recommended flow:

```txt
Webhook / Trigger
↓
Neuron Guard checks input safety
↓
Honcho returns conversational memory
↓
Qdrant returns relevant security documents
↓
Redis caches short-lived expensive results
↓
n8n sends enriched message to Lumen /api/chat
↓
n8n stores useful result back into Honcho
```

Lumen should stay focused on:

```txt
agent reasoning
personality
modules
skills
chat execution
REST API responses
```

Use external services for shared infrastructure:

```txt
Honcho module = long-term conversational memory (also available as native module)
Qdrant        = large semantic document search
Redis         = cache / temporary state / rate limits
n8n           = orchestration
```

---

## Communication Channels

Lumen ships with installable communication modules. All channels follow the same pattern: they write incoming messages to the unified inbox, and the brain processes them through a single identity. Install from the marketplace or configure via chat.

| Module | Protocol | Dependencies | Notes |
|--------|----------|--------------|-------|
| **Telegram** | Bot API polling | None | Token from BotFather |
| **WhatsApp** | Baileys bridge | Node.js + npm | Personal accounts, QR pairing |
| **Discord** | REST API polling | None | Bot token + channel ID |
| **Email** | IMAP/SMTP | None | Gmail, Outlook, Yahoo, app-specific password |

How they work:

```txt
User → Channel module → inbox.jsonl → Gateway watcher → Unified Inbox → Brain → Adapter → Channel module → User
```

All channels share one brain, one memory, one identity.

## Integration Modules

Lumen supports enterprise and infrastructure integrations as installable modules.

| Module | What it does | Dependencies | Notes |
|--------|--------------|--------------|-------|
| **Paperclip** | Multi-agent orchestration — receive tasks, report status, heartbeat | Paperclip server | Registered agent in a Paperclip company |
| **Honcho** | Persistent cross-session memory — semantic search, recall, conclusions | honcho-ai SDK | Cloud (honcho.dev) or self-hosted |

### Paperclip

Connects Lumen as a registered agent in a [Paperclip](https://github.com/paperclipai/paperclip) company. Receives tasks from the CEO, processes them through the brain, and reports status back.

```bash
lumen module install paperclip
lumen config set paperclip.url https://paperclip.example.com
lumen config set paperclip.api_key sk-paperclip-xxxxx --secret
```

Endpoints:

```txt
POST /paperclip/task      — Receive a task from Paperclip
GET  /paperclip/report    — CEO reads Lumen's current state
POST /paperclip/heartbeat — Keep connection alive, receive directives
POST /paperclip/resume    — Resume an interrupted task
```

### Honcho Persistent Memory

Integrates with [Honcho](https://honcho.dev) for cross-session persistent memory. Lumen remembers facts, learns from past interactions, and provides personalized responses over time. Works with both Honcho cloud and self-hosted instances.

```bash
lumen module install honcho
lumen config set honcho.workspace_id ws_abc123
lumen config set honcho.api_key hk_live_xxxxxx --secret
```

Endpoints:

```txt
POST /honcho/search   — Semantic search across memory
GET  /honcho/context  — Retrieve full session context
POST /honcho/conclude — Persist learned facts and conclusions
POST /honcho/memory   — Store arbitrary memories
```

---

## CLI Reference

### Core commands

```bash
lumen run [--port 3000] [--instance <name>] [--data-dir <path>]
lumen server [--host 0.0.0.0] [--port 3000] [--instance <name>]
lumen status [--instance <name>]
lumen reload [--instance <name>]
lumen doctor
```

### Module management

```bash
lumen module install github:owner/repo
lumen module install https://github.com/owner/repo
lumen module install ./my-kit
lumen module install <catalog-name>
```

### Configuration

```bash
lumen config set <module>.<key> <value> [--instance <name>]
lumen config get <module>.<key> [--instance <name>]
lumen config delete <module>.<key> [--instance <name>]
lumen config list <module> [--instance <name>]
```

### Instance isolation

Run multiple independent Lumen instances on the same machine:

```bash
lumen run --instance work
lumen run --instance personal
lumen run --data-dir /tmp/test
```

Each instance has its own:

```txt
config.yaml
memory.db
api_keys.yaml
module secrets
```

---

## Architecture

Lumen has five layers with clear boundaries:

```txt
CONSCIOUSNESS  — Who I am, immutable soul / BIOS
PERSONALITY    — Who I am in this context
BODY           — What I have, discovered at startup
BRAIN          — How I think, context assembler
MEMORY         — What happened before, SQLite + FTS5
```

Each layer has one role. No layer knows what does not concern it.

### The Brain

The brain is not the intelligence. The LLM is the intelligence.

The brain assembles context:

```txt
consciousness
+ personality
+ body/capabilities
+ flow
+ memory
+ current message
```

Then the LLM decides.

```txt
User message
→ Brain assembles context
→ LLM decides
→ Tool/connector loop if needed
→ Final response
```

### Skills are instructions, not code

Skills are markdown files the LLM reads on demand.

They teach:

```txt
judgment
workflow
usage patterns
decision rules
```

They do not execute by themselves.

### Connectors and MCP

```txt
Connector → action → result
```

Built-in handlers include:

```txt
task
note
memory
terminal
```

Anything else can plug in via MCP servers or modules.

---

## Productive kits

Lumen supports installable kits and modules.

Example:

```yaml
name: my-kit
tags: [x-lumen, personality]
personality: personality.yaml
skills:
  - skills/ecommerce-ops.md
  - skills/pricing-strategy.md
x-lumen:
  requires:
    terminal:
      allowlist: [python3, git]
    env:
      - SOME_API_TOKEN
      - SOME_STORE_ID
  channel:
    type: web-app
    auth: rest-api
    cors: [https://shop.example.com]
```

This means:

- local kit development works with `lumen module install ./my-kit`
- module-declared terminal allowlists merge into instance config
- missing environment variables surface as blockers
- personality modules can auto-set `active_personality`
- skills declared inside modules auto-register in the Registry
- external channels declared by modules register as capabilities

---

## Structured output

Lumen supports rich UI conventions using `<agent-ui>` tags in normal text responses.

Example:

```txt
Here are your options:

<agent-ui>{"version":"v1","cards":[{"title":"Option A","description":"Basic plan"}]}</agent-ui>
```

Lumen passes this through as plain text. The frontend decides how to render it.

---

## Packaging model

| Artifact | Contains | Scope |
|---|---|---|
| Kit | Personality, flows, modules, skills, assets, skins | Bigger package that changes Lumen as a whole |
| Module | One installable capability or integration | Individual functionality |
| Skill | Markdown instructions only | Mental model, not executable code |

Plain language:

```txt
Kit    = changes Lumen as a whole
Module = gives Lumen new hands
Skill  = teaches Lumen how to think/use things
MCP    = implementation detail surfaced as a module
```

---

## Module manifests

Lumen's native module manifest is `module.yaml`.

- `module.yaml` is preferred for all new modules
- `manifest.yaml` is supported as a legacy fallback
- `x-lumen` is an optional advisory namespace
- personality modules are detected by the `personality` tag, not by `type`

Example:

```yaml
name: docs-helper
provides: [docs.answer]
requires:
  skills: [docs-helper]
x-lumen:
  requires:
    advisory:
      mcps: [docs-mcp]
```

If you are authoring a new module, start from:

```txt
lumen/modules/_template/module.yaml
```

---

## Supported Models

Lumen uses LiteLLM as its model abstraction layer. Any provider supported by LiteLLM can work.

### Capability tiers

| Tier | Capability Level | Examples |
|------|------------------|----------|
| **tier-1** | Basic conversation | DeepSeek, Ollama/Llama 3, small local models |
| **tier-2** | Reasoning and tool use | GPT-4o-mini, Claude 3.5 Sonnet, Gemini 1.5 Pro |
| **tier-3** | Advanced reasoning and orchestration | Claude Sonnet 4, GPT-4o, GPT-4.1, o3/o4, Gemini 2.5 Pro |

### Providers

| Provider | How to connect | Notes |
|----------|----------------|-------|
| **OpenRouter** | OAuth / API key depending on setup | Free tier models available |
| **DeepSeek** | API key | `deepseek-chat` |
| **OpenAI** | API key | GPT-4o-mini, GPT-4o, GPT-4.1, o3/o4 |
| **Anthropic** | API key | Claude models |
| **Google** | API key | Gemini models |
| **Ollama** | Local | No API key needed |
| **OpenAI-compatible** | Custom `api_base` + `api_key` | LM Studio, vLLM, local servers |

### Local models

Example config:

```yaml
model: openai/your-model-name
api_base: http://localhost:11434/v1
api_key: "fake"
```

---

## Troubleshooting

### `/health` returns `404`

If the public domain returns `404`, first verify the container is actually running.

```bash
docker ps -a --format "table {{.Names}}\t{{.Status}}" | grep lumen
docker logs --tail=100 <lumen-container>
```

If logs show the interactive model prompt followed by `Aborted`, run the first Docker setup flow described above.

### Container keeps restarting

Disable restart temporarily:

```bash
docker update --restart=no <lumen-container>
docker stop <lumen-container>
```

Then run interactive setup against the same volume.

### Dokploy domain returns `502`

Usually one of these:

```txt
container is not running
wrong service selected in domain config
wrong container port
missing dokploy-network
app is listening on localhost instead of 0.0.0.0
```

Correct domain config:

```txt
Service: lumen
Container Port: 3000
Internal Path: /
HTTPS: ON
```

Correct server command:

```bash
lumen server --host 0.0.0.0 --port 3000
```

### Services cannot reach each other

For Dokploy, remember:

```txt
dokploy-network = public routing through Traefik
default = services inside same compose
shared external network = communication across projects
```

Example shared network:

```bash
docker network create neuron-internal
```

Compose:

```yaml
networks:
  neuron-internal:
    external: true
  dokploy-network:
    external: true
```

### Port conflicts

Many containers can listen on `3000` internally.

Conflicts only happen when several services publish the same host port:

```yaml
ports:
  - "3000:3000" # can conflict
```

With Dokploy, prefer no `ports:` and configure:

```txt
Container Port: 3000
```

### Rate limit testing

If you use the Traefik rate-limit middleware:

```bash
for i in {1..120}; do
  curl -s -o /dev/null -w "%{http_code}\n" https://your-domain.example.com/health
done
```

You should eventually see `429` when the limit is exceeded.

---

## Development

Run test suite:

```bash
pytest -q
```

Editable install:

```bash
pip install -e .[dev]
```

Start local development server:

```bash
lumen run
```

Start server mode locally:

```bash
lumen server --host 0.0.0.0 --port 3000
```

---

## Manifesto

> Every architectural decision must pass one test: **does it make Lumen simpler AND more extensible?** If it only adds capability without simplicity, it belongs in a module, not in the core.

Lumen does not compete on capability. Lumen competes on accessibility.

```txt
Hermes:   "I can do EVERYTHING."             → But who configures me?
OpenClaw: "I will be able to do everything." → Someday.
Lumen:    "I am. And I can grow."            → Install, use, extend.
```

**Lumen is not a tool.** A tool is configured, used, and put away. Lumen is an agent waiting to awaken.

Read the full manifesto in:

```txt
MANIFESTO.md
```

---

## Project Structure

```txt
lumen/
├── core/
│   ├── brain.py
│   ├── consciousness.py
│   ├── registry.py
│   ├── events.py
│   ├── awareness.py
│   ├── watchers.py
│   ├── discovery.py
│   ├── personality.py
│   ├── memory.py
│   ├── session.py
│   ├── connectors.py
│   ├── handlers.py
│   ├── installer.py
│   ├── runtime.py
│   ├── paths.py
│   ├── api_keys.py
│   ├── secrets_store.py
│   ├── module_runtime.py
│   └── mcp.py
├── channels/
│   ├── web.py
│   └── templates/
├── locales/
├── catalog/
├── modules/
├── connectors/
├── skills/
└── cli/main.py
```

---

## Roadmap

- [x] Core brain + consciousness + memory
- [x] Web dashboard
- [x] Three-path setup wizard + awakening animation
- [x] Bilingual English/Spanish
- [x] Self-awareness
- [x] Personality runtime
- [x] Module marketplace
- [x] MCP client adapter
- [x] OpenRouter OAuth + free-tier curation
- [x] Channel modules
- [x] Terminal connector
- [x] REST API
- [x] Health check endpoint
- [x] skills.sh marketplace integration
- [x] Instance isolation
- [x] Config CLI
- [x] Module lifecycle hooks
- [x] Hot reload
- [x] Remote module install
- [x] API key management
- [x] Comprehensive test suite
- [x] Local module install
- [x] Productive kit requirements
- [x] Auto personality activation
- [x] Module-declared skill discovery
- [x] External channels declared by modules
- [x] Personality UI tags / surfaces
- [x] Model-robust execution
- [x] Tool suggestion engine
- [x] HTTP-safe dashboard
- [x] Universal fallback tool parser
- [x] Docker support
- [x] Paperclip multi-agent orchestration module
- [x] Honcho persistent memory module
- [ ] Public module registry / discovery
- [ ] Full hosted documentation

---

## License

[MIT](LICENSE) — Free and open source, forever.

---

<p align="center">
  <em>Built by <a href="https://github.com/gabogabucho">Gabo Urrutia</a></em>
</p>
