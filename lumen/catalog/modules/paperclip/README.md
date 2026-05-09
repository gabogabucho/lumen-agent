# Paperclip module for Lumen

Connect this Lumen instance to a Paperclip company.
Once installed, Lumen becomes a registered agent in your org chart —
it receives tasks, reports to the CEO, and operates within company goals.

<!--
Paperclip orchestrates agents but cannot hold a conversation, remember context,
or interact with humans through messaging channels.

Lumen converses, remembers, and acts — but has no org chart, no governance,
and no place in a larger company structure.

This module bridges both systems. Once installed, a Lumen instance becomes
a first-class employee in any Paperclip-managed company: it receives tasks,
reports status, and operates within the company's goals and budget.
-->

## Install

```bash
lumen module install paperclip
```

## Configure

```bash
lumen config set paperclip.url http://your-paperclip-server:3100
lumen config set paperclip.api_key YOUR_PAPERCLIP_API_KEY
lumen config set paperclip.company_id YOUR_COMPANY_ID
lumen config set paperclip.agent_role "Your Agent Role"
```

## What you get

- **POST /paperclip/task** — Paperclip sends tasks, Lumen executes them
- **GET /paperclip/report** — Paperclip CEO reads Lumen's status
- **POST /paperclip/heartbeat** — keeps the connection alive, delivers directives

## Custom metrics

If your active personality implements `paperclip_stats()`, those metrics
appear in the CEO's daily report under the "custom" field.
This lets any Lumen personality expose domain-specific data
(leads, sales, tasks completed, etc.) without changing this module.

## Requirements

- A running Paperclip instance (paperclipai/paperclip)
- A Paperclip API key with agent registration permissions
