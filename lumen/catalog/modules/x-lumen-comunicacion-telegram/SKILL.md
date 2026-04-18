---
name: x-lumen-comunicacion-telegram
description: "Tenés Telegram disponible como canal de salida y polling de entrada."
min_capability: tier-1
provides:
  - message.send_telegram
---

# Telegram

Tenés Telegram disponible.

- Para mandar mensajes usá `message.send_telegram(chat_id, text)`.
- Si no te dan `chat_id`, pedilo o usá el default configurado.
- Los updates entrantes se capturan por polling y quedan registrados por el módulo.
- Preferí texto simple y claro.
