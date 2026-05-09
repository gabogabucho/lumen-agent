---
name: paperclip
description: "Integración con Paperclip para orquestación multi-agente. Recibe tareas de Paperclip, procesa con el brain de Lumen y reporta estado al CEO."
min_capability: tier-2
provides:
  - paperclip.agent
  - paperclip.task.receiver
  - paperclip.report.endpoint
---

# Paperclip Integration

Módulo de integración con Paperclip. Conecta esta instancia de Lumen como un agente registrado en una compañía Paperclip.

- **POST /paperclip/task** — Paperclip envía tareas, Lumen las ejecuta
- **GET /paperclip/report** — El CEO de Paperclip lee el estado actual de Lumen
- **POST /paperclip/heartbeat** — mantiene la conexión viva y recibe directivas

## Configuración

Requiere `paperclip.url`, `paperclip.api_key`, `paperclip.company_id`.

## Requisitos

- A running Paperclip instance (paperclipai/paperclip)
- Paperclip API key con permisos de registro de agentes
