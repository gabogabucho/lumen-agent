"""Web channel — FastAPI dashboard + WebSocket chat. UI-FIRST."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from neo.core.brain import Brain
from neo.core.session import SessionManager


# Set at startup by CLI, before uvicorn starts
_brain: Brain | None = None
_locale: dict = {}
_config: dict = {}
_neo_dir: Path = Path.home() / ".neo"


def configure(brain: Brain, locale: dict, config: dict):
    """Configure the web channel with brain, locale, and config."""
    global _brain, _locale, _config
    _brain = brain
    _locale = locale
    _config = config


def _has_awakened() -> bool:
    """Check if Neo has completed its first awakening."""
    return (_neo_dir / ".awakened").exists()


def _mark_awakened():
    """Mark that Neo has completed its first awakening."""
    _neo_dir.mkdir(parents=True, exist_ok=True)
    (_neo_dir / ".awakened").write_text("1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize async resources (memory DB) on startup, clean up on shutdown."""
    if _brain:
        await _brain.memory.init()
    yield
    if _brain:
        await _brain.memory.close()


app = FastAPI(title="Neo", version="0.1.0", lifespan=lifespan)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))
session_manager = SessionManager()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """First visit: awakening. After that: dashboard."""
    if not _has_awakened():
        return templates.TemplateResponse(
            "awakening.html",
            {
                "request": request,
                "language": _config.get("language", "en"),
            },
        )
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """The main dashboard — Neo's UI-FIRST experience."""
    ui = _locale.get("dashboard", {})
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "ui": ui,
            "model": _config.get("model", "not configured"),
            "language": _config.get("language", "en"),
            "version": "0.1.0",
            "connectors_count": len(_brain.connectors.list()) if _brain else 0,
            "flows_count": len(_brain.flows) if _brain else 0,
        },
    )


@app.post("/api/awakened")
async def mark_awakened():
    """Called by the awakening animation when it completes."""
    _mark_awakened()
    return {"status": "ok"}


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """Real-time chat via WebSocket."""
    await websocket.accept()
    session = session_manager.get_or_create(session_id)

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            user_text = payload.get("content", "").strip()

            if not user_text or not _brain:
                continue

            # Typing indicator
            await websocket.send_text(
                json.dumps({"type": "typing", "status": True})
            )

            # Brain thinks
            result = await _brain.think(user_text, session)

            # Send response
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": result["message"],
                    }
                )
            )

            # Stop typing
            await websocket.send_text(
                json.dumps({"type": "typing", "status": False})
            )

    except WebSocketDisconnect:
        session_manager.remove(session_id)


@app.get("/api/status")
async def api_status():
    """API endpoint for Neo's current status — from the Body (registry)."""
    registry = _brain.registry if _brain else None

    flows_info = []
    if _brain:
        for flow in _brain.flows:
            flows_info.append(
                {
                    "intent": flow.get("intent", "unknown"),
                    "triggers": flow.get("triggers", []),
                    "slots": list(flow.get("slots", {}).keys()),
                }
            )

    capabilities = []
    if registry:
        for cap in registry.all():
            capabilities.append(cap.to_dict())

    return {
        "status": "active",
        "version": "0.1.0",
        "model": _config.get("model", "not configured"),
        "language": _config.get("language", "en"),
        "capabilities": capabilities,
        "summary": registry.summary() if registry else {},
        "flows": flows_info,
        "ready": len(registry.ready()) if registry else 0,
        "gaps": len(registry.gaps()) if registry else 0,
    }
