from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import sleep, time
from urllib import error as urllib_error
from urllib import request as urllib_request

import yaml


MODULE_NAME = "x-lumen-comunicacion-whatsapp"
BRIDGE_SOURCE_DIR = Path(__file__).parent
DEFAULT_BRIDGE_PORT = 3100
POLL_INTERVAL = 2
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT = 30  # seconds to wait for bridge to become ready
DEFAULT_NPM_INSTALL_TIMEOUT = 180
DEFAULT_NPM_INSTALL_RETRIES = 2
BRIDGE_LOG_TAIL_LINES = 40


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install(context):
    context.ensure_runtime_dir()

    # Copy bridge files (bridge.js, allowlist.js, package.json) to runtime dir
    _copy_bridge_files(context.runtime_dir)

    config_path = context.runtime_dir / "config.yaml"
    if not config_path.exists():
        config_path.write_text(
            yaml.dump(
                {
                    "bridge_port_env": "WHATSAPP_BRIDGE_PORT",
                    "mode_env": "WHATSAPP_MODE",
                    "allowed_users_env": "WHATSAPP_ALLOWED_USERS",
                    "poll_interval_seconds": POLL_INTERVAL,
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
                "bridge_pid": None,
            }
        )


def uninstall(context):
    # The deactivate path handles process cleanup.  On uninstall we also
    # remove the copied bridge files — the catalog copy is the source of truth.
    pass


def _copy_bridge_files(runtime_dir: Path):
    """Copy Node.js bridge sources into the runtime directory."""
    for filename in ("bridge.js", "allowlist.js", "package.json", "package-lock.json"):
        src = BRIDGE_SOURCE_DIR / filename
        dst = runtime_dir / filename
        if src.exists():
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# WhatsAppRuntime
# ---------------------------------------------------------------------------

class WhatsAppRuntime:
    def __init__(self, context):
        self.context = context
        self._poll_task: asyncio.Task | None = None
        self._health_task: asyncio.Task | None = None
        self._bridge_proc: subprocess.Popen | None = None
        self._stopping = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self):
        state = self.context.read_runtime_state()

        node_bin = _find_node()
        if node_bin is None:
            state.update(
                {
                    "module": MODULE_NAME,
                    "status": "degraded",
                    "polling": False,
                    "error": "Node.js not found. Install Node.js (v18+) to use the WhatsApp bridge.",
                    "updated_at": time(),
                }
            )
            self.context.write_runtime_state(state)
            return

        # Ensure bridge files are present in runtime dir
        _copy_bridge_files(self.context.runtime_dir)

        # npm install / validation — transactional: never start bridge.js until
        # Baileys is present and importable.
        node_modules = self.context.runtime_dir / "node_modules"
        validation_result = await asyncio.to_thread(
            _validate_bridge_deps, node_bin, self.context.runtime_dir
        )
        if not node_modules.exists() or not validation_result["ok"]:
            npm_bin = _find_npm()
            if npm_bin is None:
                state.update(
                    {
                        "module": MODULE_NAME,
                        "status": "degraded",
                        "polling": False,
                        "error": "npm not found. Install Node.js (v18+) to use the WhatsApp bridge.",
                        "updated_at": time(),
                    }
                )
                self.context.write_runtime_state(state)
                return

            state.update(
                {
                    "module": MODULE_NAME,
                    "status": "installing",
                    "polling": False,
                    "error": None,
                    "updated_at": time(),
                }
            )
            self.context.write_runtime_state(state)

            install_timeout = _resolve_int_setting(
                self.context,
                "npm_install_timeout_seconds",
                "WHATSAPP_NPM_INSTALL_TIMEOUT_SECONDS",
                DEFAULT_NPM_INSTALL_TIMEOUT,
            )
            install_retries = _resolve_int_setting(
                self.context,
                "npm_install_retries",
                "WHATSAPP_NPM_INSTALL_RETRIES",
                DEFAULT_NPM_INSTALL_RETRIES,
            )
            install_result = await asyncio.to_thread(
                _ensure_bridge_deps,
                npm_bin,
                node_bin,
                self.context.runtime_dir,
                install_timeout,
                install_retries,
                validation_result,
            )
            if not install_result["ok"]:
                state.update(
                    {
                        "module": MODULE_NAME,
                        "status": "degraded",
                        "polling": False,
                        "error": install_result["error"],
                        "updated_at": time(),
                    }
                )
                self.context.write_runtime_state(state)
                return

        # Kill orphaned bridge processes on the configured port
        port = self._bridge_port()
        await asyncio.to_thread(_kill_orphans_on_port, port)

        # Start the bridge subprocess
        bridge_log = self.context.runtime_dir / "bridge.log"
        env = _build_bridge_env(self.context, port)

        try:
            log_fh = open(bridge_log, "a", encoding="utf-8")
            self._bridge_proc = subprocess.Popen(
                [node_bin, "bridge.js", "--port", str(port)],
                cwd=str(self.context.runtime_dir),
                stdout=log_fh,
                stderr=log_fh,
                env=env,
            )
        except Exception as exc:
            state.update(
                {
                    "module": MODULE_NAME,
                    "status": "degraded",
                    "polling": False,
                    "error": f"Failed to start bridge: {exc}",
                    "updated_at": time(),
                }
            )
            self.context.write_runtime_state(state)
            return

        # Wait for bridge to become healthy
        ready = await asyncio.to_thread(_wait_for_health, port, HEALTH_TIMEOUT)
        if not ready:
            bridge_error = _summarize_bridge_start_failure(
                bridge_log, self._bridge_proc
            )
            state.update(
                {
                    "module": MODULE_NAME,
                    "status": "degraded",
                    "polling": False,
                    "error": bridge_error,
                    "bridge_pid": self._bridge_proc.pid if self._bridge_proc else None,
                    "updated_at": time(),
                }
            )
            self.context.write_runtime_state(state)
            return

        state.update(
            {
                "module": MODULE_NAME,
                "status": "running",
                "polling": True,
                "error": None,
                "bridge_pid": self._bridge_proc.pid if self._bridge_proc else None,
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)

        self._poll_task = asyncio.create_task(
            self._poll_loop(), name=f"{MODULE_NAME}-poll"
        )
        self._health_task = asyncio.create_task(
            self._poll_health(), name=f"{MODULE_NAME}-health"
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

        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None

        if self._bridge_proc is not None:
            try:
                self._bridge_proc.terminate()
                self._bridge_proc.wait(timeout=5)
            except Exception:
                try:
                    self._bridge_proc.kill()
                except Exception:
                    pass
            self._bridge_proc = None

        state = self.context.read_runtime_state()
        state.update(
            {
                "module": MODULE_NAME,
                "status": "stopped",
                "polling": False,
                "bridge_pid": None,
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)

    # -- send methods --------------------------------------------------------

    async def send(self, recipient_id: str, message: str) -> None:
        """ChannelAdapter protocol -- route inbox response back to WhatsApp."""
        chat_id = str(recipient_id or "").strip()
        if not chat_id:
            return

        # Feature: typing indicator (opt-out via whatsapp_typing_indicator: false)
        typing_setting = self.context.resolve_setting(
            "whatsapp_typing_indicator", "WHATSAPP_TYPING_INDICATOR"
        )
        typing_enabled = str(typing_setting).lower() not in ("false", "0", "no") if typing_setting is not None else True
        if typing_enabled:
            try:
                await asyncio.to_thread(
                    _bridge_post,
                    self._bridge_port(),
                    "/typing",
                    {"chatId": chat_id},
                )
            except Exception:
                pass  # non-blocking — failed indicator never blocks the actual send

        # Feature: TTS — convert response to a WhatsApp voice note
        tts_model = self.context.resolve_setting("tts_model", "TOKAINE_TTS_MODEL") or ""
        if tts_model:
            audio_path = await _synthesize_speech(self.context, message)
            if audio_path:
                try:
                    await asyncio.to_thread(
                        _bridge_post,
                        self._bridge_port(),
                        "/send-media",
                        {"chatId": chat_id, "filePath": audio_path, "mediaType": "audio"},
                    )
                    return  # sent as voice note — skip text fallback
                except Exception:
                    pass  # fall through to text if the media send fails

        # Default: send as text
        try:
            await asyncio.to_thread(
                _bridge_post,
                self._bridge_port(),
                "/send",
                {"chatId": chat_id, "message": message},
            )
        except Exception:
            pass

    async def send_message(self, text: str, chat_id: str | None = None) -> dict:
        """Tool-registered method: send a WhatsApp message."""
        resolved_chat_id = str(chat_id or "").strip()
        if not resolved_chat_id:
            return {
                "status": "error",
                "error": "Missing WhatsApp chat_id (phone number or JID).",
            }

        try:
            result = await asyncio.to_thread(
                _bridge_post,
                self._bridge_port(),
                "/send",
                {"chatId": resolved_chat_id, "message": text},
            )
            return {
                "status": "ok",
                "chat_id": resolved_chat_id,
                "message_id": result.get("messageId"),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
            }

    # -- poll loop -----------------------------------------------------------

    async def _poll_loop(self):
        while not self._stopping:
            try:
                messages = await asyncio.to_thread(
                    _bridge_get, self._bridge_port(), "/messages"
                )
                for msg in messages:
                    await self._handle_message(msg)
                await asyncio.sleep(POLL_INTERVAL)
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
                await asyncio.sleep(POLL_INTERVAL)

    async def _poll_health(self):
        while not self._stopping:
            try:
                health = await asyncio.to_thread(
                    _bridge_get, self._bridge_port(), "/health"
                )
                state = self.context.read_runtime_state()
                state["whatsapp_health"] = {
                    "status": health.get("status", "unknown"),
                    "connected": health.get("connected", False),
                    "number": health.get("number"),
                    "session_status": health.get("session_status", "unknown"),
                    "updated_at": time(),
                }
                self.context.write_runtime_state(state)
            except Exception:
                state = self.context.read_runtime_state()
                state["whatsapp_health"] = {
                    "status": "unknown",
                    "connected": False,
                    "number": None,
                    "session_status": "unknown",
                    "updated_at": time(),
                }
                self.context.write_runtime_state(state)
            await asyncio.sleep(HEALTH_POLL_INTERVAL)

    async def _handle_message(self, msg: dict):
        chat_id = msg.get("chatId", "")
        body = msg.get("body", "")
        sender = msg.get("senderId", "")
        timestamp = msg.get("timestamp")
        media_type = msg.get("mediaType", "")
        media_urls = msg.get("mediaUrls") or []

        # Feature: STT — transcribe incoming voice/PTT messages before routing
        if media_type in ("ptt", "audio") and media_urls:
            transcript = await _transcribe_audio(self.context, media_urls[0])
            if transcript:
                body = transcript
            else:
                # No STT configured or transcription failed — drop the message.
                # (Without a transcript, Lumen would only see a placeholder like
                # "[ptt received]" which produces a confused/unhelpful reply.)
                return

        state = self.context.read_runtime_state()
        state.update(
            {
                "module": MODULE_NAME,
                "status": "running",
                "polling": True,
                "last_chat_id": chat_id,
                "last_sender": sender,
                "last_message_preview": body[:120],
                "updated_at": time(),
            }
        )
        self.context.write_runtime_state(state)

        if not chat_id or not body:
            return

        # Write to inbox.jsonl -- the framework watches this file and
        # bridges entries to the unified Inbox automatically.
        inbox_path = self.context.runtime_dir / "inbox.jsonl"
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        with inbox_path.open("a", encoding="utf-8") as inbox_file:
            inbox_file.write(
                json.dumps(
                    {
                        "chat_id": chat_id,
                        "text": body,
                        "received_at": time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        if (
            self.context.memory is not None
            and getattr(self.context.memory, "_db", None) is not None
        ):
            await self.context.memory.remember(
                body,
                category="whatsapp_message",
                metadata={"chat_id": str(chat_id), "module": MODULE_NAME},
            )

    # -- helpers -------------------------------------------------------------

    def _bridge_port(self) -> int:
        raw = self.context.resolve_setting(
            "bridge_port", "WHATSAPP_BRIDGE_PORT"
        )
        if raw:
            try:
                return int(raw)
            except (ValueError, TypeError):
                pass
        return DEFAULT_BRIDGE_PORT


# ---------------------------------------------------------------------------
# activate / deactivate
# ---------------------------------------------------------------------------

async def activate(context):
    runtime = WhatsAppRuntime(context)
    context.register_tool(
        "message.send_whatsapp",
        "Send a WhatsApp message using the installed WhatsApp communication module.",
        {
            "type": "object",
            "properties": {
                "chat_id": {
                    "type": "string",
                    "description": "WhatsApp chat ID (phone number with country code or JID like 5491112345678@s.whatsapp.net).",
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


# ---------------------------------------------------------------------------
# HTTP helpers (urllib, no external deps)
# ---------------------------------------------------------------------------

def _bridge_url(port: int, path: str) -> str:
    return f"http://127.0.0.1:{port}{path}"


def _bridge_get(port: int, path: str, timeout: int = 10) -> list:
    """GET from the bridge and return parsed JSON (expects a list)."""
    url = _bridge_url(port, path)
    req = urllib_request.Request(url, method="GET")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        raise RuntimeError(f"Bridge HTTP error: {exc.code}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Bridge unreachable: {exc.reason}") from exc


def _bridge_post(port: int, path: str, payload: dict, timeout: int = 15) -> dict:
    """POST to the bridge and return parsed JSON."""
    url = _bridge_url(port, path)
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib_request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib_request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Bridge HTTP error: {exc.code} {body}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Bridge unreachable: {exc.reason}") from exc


def _resolve_int_setting(context, key: str, env_key: str, default: int) -> int:
    raw = context.resolve_setting(key, env_key)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------------------
# STT helpers
# ---------------------------------------------------------------------------

async def _transcribe_audio(context, audio_path: str) -> str | None:
    """Transcribe an audio file using the configured STT model.

    Returns the transcript string, or None if STT is not configured or fails.
    """
    stt_model = context.resolve_setting("stt_model", "TOKAINE_STT_MODEL") or ""
    if not stt_model:
        return None
    api_base = (
        context.resolve_setting("api_base", "LUMEN_API_BASE") or "https://api.openai.com/v1"
    ).rstrip("/")
    api_key = context.resolve_setting("api_key", "LUMEN_API_KEY") or ""
    try:
        return await asyncio.to_thread(
            _transcribe_audio_sync, stt_model, api_base, api_key, audio_path
        )
    except Exception:
        return None


def _transcribe_audio_sync(model: str, api_base: str, api_key: str, audio_path: str) -> str | None:
    """Synchronous call to an OpenAI-compatible /audio/transcriptions endpoint."""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        return None

    audio_bytes = audio_file.read_bytes()
    filename = audio_file.name
    ext = audio_file.suffix.lower()
    _AUDIO_MIME = {
        ".ogg": "audio/ogg",
        ".opus": "audio/opus",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".flac": "audio/flac",
        ".webm": "audio/webm",
    }
    mime_type = _AUDIO_MIME.get(ext, "audio/ogg")

    # Build multipart/form-data manually (no external deps)
    boundary = "----LumenSTTBoundary" + os.urandom(8).hex()

    def _field(name: str, value: str) -> bytes:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode("utf-8")

    body = (
        _field("model", model)
        + f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8")
        + audio_bytes
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )

    url = f"{api_base}/audio/transcriptions"
    headers: dict = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib_request.Request(url, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result.get("text") or None


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

async def _synthesize_speech(context, text: str) -> str | None:
    """Convert text to speech using the configured TTS model.

    Returns the path to the generated audio file, or None if TTS is not
    configured or synthesis fails.
    """
    tts_model = context.resolve_setting("tts_model", "TOKAINE_TTS_MODEL") or ""
    if not tts_model:
        return None
    api_base = (
        context.resolve_setting("api_base", "LUMEN_API_BASE") or "https://api.openai.com/v1"
    ).rstrip("/")
    api_key = context.resolve_setting("api_key", "LUMEN_API_KEY") or ""
    tts_voice = context.resolve_setting("tts_voice", "TOKAINE_TTS_VOICE") or "alloy"
    try:
        return await asyncio.to_thread(
            _synthesize_speech_sync, tts_model, api_base, api_key, tts_voice, text
        )
    except Exception:
        return None


def _synthesize_speech_sync(
    model: str, api_base: str, api_key: str, voice: str, text: str
) -> str | None:
    """Synchronous call to an OpenAI-compatible /audio/speech endpoint.

    Saves the returned audio to the WhatsApp audio_cache directory and
    returns the file path so bridge.js /send-media can serve it.
    """
    url = f"{api_base}/audio/speech"
    payload = json.dumps({"model": model, "input": text, "voice": voice}).encode("utf-8")
    headers: dict = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib_request.Request(url, data=payload, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=60) as response:
        audio_bytes = response.read()
        content_type = response.headers.get("Content-Type", "")

    # Determine extension from Content-Type; default to .ogg (WhatsApp PTT friendly)
    if "ogg" in content_type:
        ext = ".ogg"
    elif "mpeg" in content_type or "mp3" in content_type:
        ext = ".mp3"
    else:
        ext = ".ogg"

    audio_dir = Path(os.path.expanduser("~")) / ".lumen" / "whatsapp" / "audio_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"tts_{os.urandom(6).hex()}{ext}"
    audio_path.write_bytes(audio_bytes)
    return str(audio_path)


# ---------------------------------------------------------------------------
# Process management helpers
# ---------------------------------------------------------------------------

def _find_node() -> str | None:
    """Find the Node.js binary."""
    for candidate in ("node", "node.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _find_npm() -> str | None:
    """Find the npm binary."""
    for candidate in ("npm", "npm.cmd"):
        found = shutil.which(candidate)
        if found:
            return found
    return None


def _run_npm_install(npm_bin: str, cwd: Path, timeout: int) -> dict:
    """Run npm install synchronously and capture stdout/stderr."""
    try:
        result = subprocess.run(
            [npm_bin, "install", "--production"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "error": None,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "error": f"npm install timed out after {timeout}s",
        }
    except Exception as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "error": f"npm install failed to start: {exc}",
        }


def _ensure_bridge_deps(
    npm_bin: str,
    node_bin: str,
    runtime_dir: Path,
    timeout: int,
    retries: int,
    initial_validation: dict | None = None,
) -> dict:
    """Install + validate bridge dependencies before starting bridge.js."""
    attempts = max(retries, 1)
    last_error = "Unknown dependency installation failure"
    previous_validation = initial_validation or {"ok": True}
    node_modules = runtime_dir / "node_modules"

    for attempt in range(1, attempts + 1):
        if attempt > 1 or not previous_validation.get("ok", True):
            shutil.rmtree(node_modules, ignore_errors=True)

        install_result = _run_npm_install(npm_bin, runtime_dir, timeout)
        install_summary = _format_install_result(install_result)
        if not install_result["ok"]:
            last_error = (
                f"WhatsApp bridge dependency install failed on attempt {attempt}/{attempts}. "
                f"{install_result.get('error') or 'npm install returned non-zero exit status.'}\n"
                f"{install_summary}"
            )
            continue

        validation = _validate_bridge_deps(node_bin, runtime_dir)
        if validation["ok"]:
            return {"ok": True}

        last_error = (
            f"Baileys install incomplete / import failed on attempt {attempt}/{attempts}.\n"
            f"Validation: {validation['error']}\n"
            f"{install_summary}"
        )

    return {"ok": False, "error": last_error}


def _validate_bridge_deps(node_bin: str, runtime_dir: Path) -> dict:
    """Validate that Baileys exists and is importable before starting bridge.js."""
    package_dir = runtime_dir / "node_modules" / "@whiskeysockets" / "baileys"
    package_json = package_dir / "package.json"
    if not package_json.exists():
        return {
            "ok": False,
            "error": f"Missing package metadata: {package_json}",
        }

    try:
        result = subprocess.run(
            [
                node_bin,
                "--input-type=module",
                "-e",
                "await import('@whiskeysockets/baileys'); console.log('ok');",
            ],
            cwd=str(runtime_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "Timed out validating import('@whiskeysockets/baileys')",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Failed to validate Baileys import: {exc}",
        }

    if result.returncode != 0:
        detail = _combine_output(result.stdout or "", result.stderr or "")
        return {
            "ok": False,
            "error": f"import('@whiskeysockets/baileys') failed. {_truncate_text(detail)}",
        }

    return {"ok": True, "error": None}


def _format_install_result(result: dict) -> str:
    parts = []
    if result.get("returncode") is not None:
        parts.append(f"npm exit code: {result['returncode']}")
    if result.get("stdout"):
        parts.append(f"npm stdout:\n{_truncate_text(result['stdout'])}")
    if result.get("stderr"):
        parts.append(f"npm stderr:\n{_truncate_text(result['stderr'])}")
    return "\n".join(parts).strip()


def _summarize_bridge_start_failure(
    bridge_log: Path, bridge_proc: subprocess.Popen | None
) -> str:
    """Return a clearer bridge startup error than generic health timeout."""
    parts = ["Bridge did not become healthy in time."]
    if bridge_proc is not None:
        returncode = bridge_proc.poll()
        if returncode is not None:
            parts.append(f"bridge.js exited early with code {returncode}.")

    log_tail = _read_log_tail(bridge_log, BRIDGE_LOG_TAIL_LINES)
    if log_tail:
        parts.append(f"Last bridge.log lines:\n{log_tail}")
    else:
        parts.append("No bridge.log output captured.")
    return " ".join(parts)


def _read_log_tail(log_path: Path, max_lines: int) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return ""
    return _truncate_text("\n".join(lines[-max_lines:]))


def _combine_output(stdout: str, stderr: str) -> str:
    parts = []
    if stdout.strip():
        parts.append(stdout.strip())
    if stderr.strip():
        parts.append(stderr.strip())
    return "\n".join(parts)


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_bridge_env(context, port: int) -> dict:
    """Build the environment dict for the bridge subprocess."""
    env = os.environ.copy()
    env["WHATSAPP_BRIDGE_PORT"] = str(port)

    mode = context.resolve_setting("mode", "WHATSAPP_MODE")
    if mode:
        env["WHATSAPP_MODE"] = str(mode)

    allowed = context.resolve_setting("allowed_users", "WHATSAPP_ALLOWED_USERS")
    if allowed:
        env["WHATSAPP_ALLOWED_USERS"] = str(allowed)

    pairing_phone = context.resolve_setting("pairing_phone", "LUMEN_PAIRING_PHONE")
    if pairing_phone:
        env["LUMEN_PAIRING_PHONE"] = str(pairing_phone)

    pairing_device = context.resolve_setting("pairing_device_name", "LUMEN_PAIRING_DEVICE_NAME")
    if pairing_device:
        env["LUMEN_PAIRING_DEVICE_NAME"] = str(pairing_device)

    return env


def _wait_for_health(port: int, timeout: int = 30) -> bool:
    """Block until the bridge /health endpoint responds, or timeout."""
    deadline = time() + timeout
    while time() < deadline:
        try:
            url = _bridge_url(port, "/health")
            req = urllib_request.Request(url, method="GET")
            with urllib_request.urlopen(req, timeout=3) as response:
                data = json.loads(response.read().decode("utf-8"))
                if data.get("status") in ("connected", "disconnected"):
                    # "disconnected" is fine -- it means the bridge is up,
                    # waiting for QR scan or reconnection.
                    return True
        except Exception:
            pass
        sleep(1)
    return False


def _kill_orphans_on_port(port: int) -> None:
    """Kill any process already listening on the given port."""
    if sys.platform == "win32":
        _kill_orphans_win32(port)
    else:
        _kill_orphans_unix(port)


def _kill_orphans_win32(port: int) -> None:
    """Windows: use netstat + taskkill."""
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.splitlines():
            # Lines look like:  TCP    127.0.0.1:3100    0.0.0.0:0    LISTENING    12345
            parts = line.split()
            if len(parts) >= 5 and parts[1].endswith(f":{port}"):
                try:
                    pid = int(parts[-1])
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                except (ValueError, subprocess.TimeoutExpired):
                    pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _kill_orphans_unix(port: int) -> None:
    """Unix: use lsof or fuser."""
    try:
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
