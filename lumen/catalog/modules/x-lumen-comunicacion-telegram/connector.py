from __future__ import annotations

import asyncio
import json
from pathlib import Path
from time import time
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml


API_ROOT = "https://api.telegram.org"
MODULE_NAME = "x-lumen-comunicacion-telegram"


def install(context):
    context.ensure_runtime_dir()

    config_path = context.runtime_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.dump(
                {
                    "bot_token_env": "TELEGRAM_BOT_TOKEN",
                    "default_chat_id_env": "TELEGRAM_DEFAULT_CHAT_ID",
                    "poll_interval_seconds": 2,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    if not (context.runtime_dir / "runtime.json").exists():
        context.write_runtime_state(
            {
                "module": MODULE_NAME,
                "status": "installed",
                "polling": False,
                "last_update_id": None,
            }
        )


def uninstall(context):
    token = context.resolve_setting("bot_token", "TELEGRAM_BOT_TOKEN")
    if token:
        try:
            _telegram_api(token, "deleteWebhook", {"drop_pending_updates": False})
        except RuntimeError:
            pass


class TelegramRuntime:
    def __init__(self, context):
        self.context = context
        self._poll_task: asyncio.Task | None = None
        self._stopping = False

    async def start(self):
        state = self.context.read_runtime_state()
        token = self._bot_token()
        if not token:
            state.update(
                {
                    "module": MODULE_NAME,
                    "status": "degraded",
                    "polling": False,
                    "error": "Missing TELEGRAM_BOT_TOKEN",
                    "updated_at": time(),
                }
            )
            self.context.write_runtime_state(state)
            return

        await asyncio.to_thread(
            _telegram_api, token, "deleteWebhook", {"drop_pending_updates": False}
        )
        state.update(
            {
                "module": MODULE_NAME,
                "status": "running",
                "polling": True,
                "error": None,
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name=f"{MODULE_NAME}-poll"
        )

    async def stop(self):
        self._stopping = True
        if self._poll_task is not None:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        state = self.context.read_runtime_state()
        state.update(
            {
                "module": MODULE_NAME,
                "status": "stopped",
                "polling": False,
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)

    async def send_message(self, text: str, chat_id: str | None = None):
        token = self._bot_token()
        if not token:
            return {
                "status": "error",
                "error": "Missing TELEGRAM_BOT_TOKEN",
            }

        resolved_chat_id = str(chat_id or self._default_chat_id() or "").strip()
        if not resolved_chat_id:
            return {
                "status": "error",
                "error": "Missing Telegram chat_id",
            }

        result = await asyncio.to_thread(
            _telegram_api,
            token,
            "sendMessage",
            {"chat_id": resolved_chat_id, "text": text},
        )
        return {
            "status": "ok",
            "chat_id": resolved_chat_id,
            "message_id": ((result.get("result") or {}).get("message_id")),
        }

    async def _poll_loop(self):
        while not self._stopping:
            try:
                token = self._bot_token()
                if not token:
                    await asyncio.sleep(2)
                    continue

                state = self.context.read_runtime_state()
                offset = state.get("last_update_id")
                payload = {"timeout": 20}
                if offset is not None:
                    payload["offset"] = int(offset) + 1

                response = await asyncio.to_thread(
                    _telegram_api,
                    token,
                    "getUpdates",
                    payload,
                    25,
                )
                for update in response.get("result") or []:
                    await self._handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                state = self.context.read_runtime_state()
                state.update(
                    {
                        "module": MODULE_NAME,
                        "status": "degraded",
                        "polling": False,
                        "error": str(exc),
                        "updated_at": time(),
                    }
                )
                self.context.write_runtime_state(state)
                await asyncio.sleep(2)

    async def _handle_update(self, update: dict):
        update_id = update.get("update_id")
        message = update.get("message") or update.get("edited_message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = message.get("text") or message.get("caption") or ""

        state = self.context.read_runtime_state()
        state.update(
            {
                "module": MODULE_NAME,
                "status": "running",
                "polling": True,
                "last_update_id": update_id,
                "last_chat_id": chat_id,
                "last_message_preview": text[:120],
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)

        if chat_id is None:
            return

        inbox_path = self.context.runtime_dir / "inbox.jsonl"
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        with inbox_path.open("a", encoding="utf-8") as inbox_file:
            inbox_file.write(
                json.dumps(
                    {
                        "update_id": update_id,
                        "chat_id": chat_id,
                        "text": text,
                        "received_at": time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if (
            self.context.memory is not None
            and getattr(self.context.memory, "_db", None) is not None
            and text
        ):
            await self.context.memory.remember(
                text,
                category="telegram_message",
                metadata={"chat_id": str(chat_id), "module": MODULE_NAME},
            )

    def _bot_token(self) -> str | None:
        return self.context.resolve_setting("bot_token", "TELEGRAM_BOT_TOKEN")

    def _default_chat_id(self) -> str | None:
        return self.context.resolve_setting(
            "default_chat_id", "TELEGRAM_DEFAULT_CHAT_ID"
        )


async def activate(context):
    runtime = TelegramRuntime(context)
    context.register_tool(
        "message.send_telegram",
        "Send a Telegram message using the installed Telegram communication module.",
        {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "Telegram chat ID. Optional if TELEGRAM_DEFAULT_CHAT_ID is configured.",
                },
                "text": {
                    "type": "string",
                    "description": "Plain-text message to send.",
                },
            },
            "required": ["text"],
        },
        runtime.send_message,
        metadata={"kind": "module", "module": MODULE_NAME},
    )
    await runtime.start()
    return runtime


async def deactivate(context, runtime):
    if runtime is not None:
        await runtime.stop()


def _telegram_api(
    token: str,
    method: str,
    payload: dict | None = None,
    timeout: int = 15,
) -> dict:
    url = f"{API_ROOT}/bot{token}/{method}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Telegram API error: {exc.code} {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Telegram API unavailable: {exc.reason}") from exc

    if not body.get("ok"):
        raise RuntimeError(body.get("description", "Telegram API rejected the request"))
    return body
