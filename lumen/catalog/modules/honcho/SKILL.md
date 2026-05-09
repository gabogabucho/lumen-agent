---
name: honcho-memory
description: "Integración con Honcho.dev para memoria persistente cross-session — recuperar hechos, aprender de interacciones pasadas, modelado de pares y razonamiento dialéctico."
min_capability: tier-2
provides:
  - honcho.memory
  - honcho.search
  - honcho.remember
---

# Honcho Persistent Memory

Módulo de integración con Honcho.dev. Conecta esta instancia de Lumen para almacenar,
recuperar y razonar sobre conocimiento acumulado a través de sesiones.

## Endpoints

- **POST /honcho/search** — Búsqueda semántica en la memoria de Honcho
- **GET /honcho/context** — Recuperación de contexto completo de una sesión
- **POST /honcho/conclude** — Escritura de conclusiones/hechos aprendidos
- **POST /honcho/memory** — Almacenamiento directo de memorias

## Configuración

Requiere `workspace_id` (obligatorio), `api_key` (secreto), y configuración opcional de
`base_url`, `session_strategy` y `recall_mode`.

## Requisitos

- Una cuenta en Honcho.dev (o instancia self-hosted con `base_url`)
- Clave de API válida (`api_key`)
- Capability tier-2 o superior
