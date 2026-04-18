# x-lumen-comunicacion-telegram

Módulo autocontenido de Telegram para Lumen.

| Tipo | Qué incluye | Ejemplo |
|---|---|---|
| Kit | Bundle más grande: puede traer personalidad, flows, módulos, skills y assets como una solución armada | `x-lumen-personal` |
| Módulo | Capability puntual instalable que agrega una función concreta | `x-lumen-comunicacion-telegram` |
| Skill | Instrucciones Markdown para que el cerebro sepa usar una capacidad | `SKILL.md` |

## Setup

1. Crear un bot con `@BotFather`.
2. Exportar `TELEGRAM_BOT_TOKEN`.
3. Opcional: exportar `TELEGRAM_DEFAULT_CHAT_ID`.
4. Instalar el módulo desde el catálogo.

## Runtime

- Estado: `~/.lumen/modules/x-lumen-comunicacion-telegram/runtime.json`
- Config: `~/.lumen/modules/x-lumen-comunicacion-telegram/config.yaml`
- Inbox: `~/.lumen/modules/x-lumen-comunicacion-telegram/inbox.jsonl`

Al desinstalar, el módulo limpia su runtime completo.
