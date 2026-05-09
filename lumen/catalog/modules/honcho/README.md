# Honcho module for Lumen

Persistent cross-session memory powered by Honcho.dev.

## Install

```bash
lumen module install honcho
```

## Configure

```bash
lumen config set honcho.workspace_id YOUR_WORKSPACE_ID
lumen config set honcho.api_key YOUR_HONCHO_API_KEY
lumen config set honcho.session_strategy per-session    # optional: per-session|per-directory|per-repo|global
lumen config set honcho.recall_mode hybrid              # optional: hybrid|context|tools
lumen config set honcho.base_url https://honcho.internal  # optional — self-hosted
```

## What you get

- **POST /honcho/search** — Semantic search across all your Honcho memory
- **GET /honcho/context** — Retrieve full context block for a session
- **POST /honcho/conclude** — Persist learned facts and conclusions
- **POST /honcho/memory** — Direct memory store with optional session association

## Memory sync

The module automatically injects context before session thought and writes
conclusions after session end — configured via the awareness skill.

## Requirements

- A Honcho.dev workspace with a valid API key
- Capability tier-2 or higher
