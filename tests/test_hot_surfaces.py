"""Tests del Bloque 10: watchers y refresco en caliente.

Cubre tasks.md 10.1 a 10.4:
- los directorios vigilados incluyen surfaces/agents/machines (10.1)
- instalar un surface.yaml en caliente deja la superficie en no_renderer
  sin reiniciar (10.2)
- instalar un agent.yaml en caliente deja el especialista READY (10.3)
- 200 manifiestos se escanean en menos de 1 s (10.4)
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import time
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from lumen.channels import web as web_channel
from lumen.core import discovery
from lumen.core.awareness import CapabilityAwareness
from lumen.core.connectors import ConnectorRegistry
from lumen.core.registry import CapabilityKind, CapabilityStatus
from lumen.core.watchers import FilePoller


def _write_surface(root: Path, name: str) -> None:
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "schema.json").write_text(
        jsonlib.dumps({"type": "object", "properties": {"items": {"type": "array"}}}),
        encoding="utf-8",
    )
    (base / "surface.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "name": name,
                "description": f"Superficie {name}.",
                "schema": "./schema.json",
            }
        ),
        encoding="utf-8",
    )


def _write_agent(root: Path, name: str) -> None:
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "personality.md").write_text("Voz directa y cálida.\n", encoding="utf-8")
    (base / "agent.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "name": name,
                "description": f"Especialista {name}.",
                "personality": "./personality.md",
                "skills": [],
                "tools": [],
            }
        ),
        encoding="utf-8",
    )


class _BrainStub:
    """Stub liviano: refresh_runtime_registry solo toca estos atributos."""

    def __init__(self, lumen_dir: Path, pkg_dir: Path):
        self.registry = _discovered_registry(lumen_dir, pkg_dir)
        self.connectors = ConnectorRegistry()
        self.flows = []
        self.memory = None
        self.config = {}
        self.module_manager = MagicMock()
        self.module_manager.config = None
        self.mcp_manager = MagicMock()
        self.mcp_manager.discovery_payload.return_value = None
        self.capability_awareness = None
        self.marketplace = None
        self.model = "test-model"


def _discovered_registry(lumen_dir: Path, pkg_dir: Path):
    from lumen.core.registry import Registry

    registry = Registry()
    discovery.discover_all(
        registry=registry,
        pkg_dir=pkg_dir,
        connectors=ConnectorRegistry(),
        active_channels=["web"],
        lumen_dir=lumen_dir,
    )
    return registry


# --- 10.1: directorios vigilados ---


def test_file_poller_watches_new_roots(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    lumen_dir.mkdir()
    pkg_dir.mkdir()

    (lumen_dir / "modules").mkdir()
    (lumen_dir / "surfaces").mkdir()
    (pkg_dir / "surfaces").mkdir()
    (lumen_dir / "agents").mkdir()
    (pkg_dir / "agents").mkdir()
    (lumen_dir / "machines").mkdir()
    (pkg_dir / "machines").mkdir()

    watched = web_channel.capability_watch_dirs(lumen_dir, pkg_dir)

    names = [d.name for d in watched]
    assert names == ["modules", "surfaces", "surfaces", "agents", "agents", "machines", "machines"]
    assert watched[0] == lumen_dir / "modules"  # modules sigue primero
    assert watched[2] == pkg_dir / "surfaces"
    assert lumen_dir / "skills" not in watched  # solo cambia lo especificado

    # Directorios inexistentes no entran a la lista.
    empty_dir = tmp_path / "vacio"
    empty_dir.mkdir()
    assert web_channel.capability_watch_dirs(empty_dir, empty_dir) == []


# --- 10.2 / 10.3: instalación en caliente ---


async def test_new_surface_appears_without_restart(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    lumen_dir.mkdir()
    pkg_dir.mkdir()
    # Los roots existen desde el arranque, como en el lifespan del canal web.
    (lumen_dir / "surfaces").mkdir()
    (pkg_dir / "surfaces").mkdir()

    brain = _BrainStub(lumen_dir, pkg_dir)
    awareness = CapabilityAwareness(brain.registry)
    brain.capability_awareness = awareness

    watched = web_channel.capability_watch_dirs(lumen_dir, pkg_dir)

    async def _on_change():
        _refresh(brain, lumen_dir, pkg_dir)

    poller = FilePoller(brain.registry, watched, on_change=_on_change)
    await poller.start(interval=0.05)
    try:
        # Instalar una superficie con el motor corriendo.
        _write_surface(lumen_dir / "surfaces", "timeline")
        await _wait_for(lambda: brain.registry.get(CapabilityKind.SURFACE, "timeline"))
    finally:
        await poller.stop()

    cap = brain.registry.get(CapabilityKind.SURFACE, "timeline")
    assert cap is not None
    # Sin reiniciar, y sin renderer confirmado todavía.
    assert cap.status == CapabilityStatus.NO_RENDERER

    # El prompt del turno siguiente ya la menciona como no lista.
    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert "timeline" in prompt


async def test_new_agent_ready_without_restart(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    lumen_dir.mkdir()
    pkg_dir.mkdir()
    # Los roots existen desde el arranque, como en el lifespan del canal web.
    (lumen_dir / "agents").mkdir()
    (pkg_dir / "agents").mkdir()

    brain = _BrainStub(lumen_dir, pkg_dir)
    awareness = CapabilityAwareness(brain.registry)
    brain.capability_awareness = awareness

    watched = web_channel.capability_watch_dirs(lumen_dir, pkg_dir)

    async def _on_change():
        _refresh(brain, lumen_dir, pkg_dir)

    poller = FilePoller(brain.registry, watched, on_change=_on_change)
    await poller.start(interval=0.05)
    try:
        _write_agent(lumen_dir / "agents", "crecimiento")
        await _wait_for(lambda: brain.registry.get(CapabilityKind.AGENT, "crecimiento"))
    finally:
        await poller.stop()

    cap = brain.registry.get(CapabilityKind.AGENT, "crecimiento")
    assert cap is not None
    assert cap.status == CapabilityStatus.READY

    proactive = awareness.format_for_proactive()
    assert proactive is not None
    assert "crecimiento" in proactive


def _refresh(brain, lumen_dir, pkg_dir) -> None:
    from lumen.core.runtime import refresh_runtime_registry

    refresh_runtime_registry(brain, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"])


async def _wait_for(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not met in time")


# --- 10.4: costo acotado del discovery ---


def test_discovery_200_surfaces_under_one_second(tmp_path):
    root = tmp_path / "surfaces"
    root.mkdir()
    schema_body = jsonlib.dumps({"type": "object"})
    for i in range(200):
        base = root / f"surface_{i:03d}"
        base.mkdir()
        (base / "schema.json").write_text(schema_body, encoding="utf-8")
        (base / "surface.yaml").write_text(
            yaml.safe_dump(
                {
                    "manifest_version": 1,
                    "name": f"surface_{i:03d}",
                    "description": "x",
                    "schema": "./schema.json",
                }
            ),
            encoding="utf-8",
        )

    registry = growth_registry()
    start = time.perf_counter()
    discovery._discover_surfaces(registry, [root])
    elapsed = time.perf_counter() - start

    assert len(registry.list_by_kind(CapabilityKind.SURFACE)) == 200
    assert elapsed < 1.0, f"discovery tardó {elapsed:.3f}s"


def growth_registry():
    from lumen.core.registry import Registry

    return Registry()
