"""Tests del Bloque A — el refresh en caliente conserva superficies confirmadas.

Cubre .refs/onboarding-catalogo/tasks.md (Bloque A):
- `test_refresh_keeps_confirmed_surface`: confirmar `hello`, correr
  `refresh_runtime_registry`, seguir en `READY` y poder `render_surface`
- `test_refresh_does_not_promote_stale_confirmation`: una confirmación de una
  superficie que ya no existe no se cuela
- `test_refresh_does_not_promote_unconfirmed_surface`: sin confirmación no hay
  promoción automática
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from lumen.core import discovery
from lumen.core.connectors import ConnectorRegistry
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry
from lumen.core.runtime import refresh_runtime_registry
from lumen.core.surfaces import (
    RendererConfirmation,
    apply_hello,
    build_render_surface_tool,
    execute_render_surface,
)

CONTRACT_V3 = "x-lumen-desktop/3"


def _write_surface(pkg_dir: Path, name: str) -> None:
    base = pkg_dir / "surfaces" / name
    base.mkdir(parents=True, exist_ok=True)
    (base / "schema.json").write_text(
        json.dumps({"type": "object", "properties": {"items": {"type": "array"}}}),
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


class _BrainStub:
    """Stub liviano: refresh_runtime_registry solo toca estos atributos."""

    def __init__(self, lumen_dir: Path, pkg_dir: Path):
        self.registry = self._discovered(lumen_dir, pkg_dir)
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

    @staticmethod
    def _discovered(lumen_dir: Path, pkg_dir: Path) -> Registry:
        registry = Registry()
        discovery.discover_all(
            registry=registry,
            pkg_dir=pkg_dir,
            connectors=ConnectorRegistry(),
            active_channels=["web"],
            lumen_dir=lumen_dir,
        )
        return registry


def _refresh(brain, lumen_dir: Path, pkg_dir: Path) -> Registry:
    return refresh_runtime_registry(
        brain, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"]
    )


def test_refresh_keeps_confirmed_surface(tmp_path):
    pkg_dir = tmp_path / "pkg"
    lumen_dir = tmp_path / "lumen"
    (lumen_dir / "surfaces").mkdir(parents=True)
    _write_surface(pkg_dir, "hello")

    brain = _BrainStub(lumen_dir, pkg_dir)
    apply_hello(
        brain.registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "hello", "version": "1.0.0"}],
        host={"kind": "electron"},
    )
    cap = brain.registry.get(CapabilityKind.SURFACE, "hello")
    assert cap.status == CapabilityStatus.READY

    _refresh(brain, lumen_dir, pkg_dir)

    cap = brain.registry.get(CapabilityKind.SURFACE, "hello")
    assert cap is not None
    assert cap.status == CapabilityStatus.READY

    # La tool de render sigue ofreciendo la superficie sin otro hello.
    tool = build_render_surface_tool(brain.registry)
    assert tool is not None
    assert (
        "hello"
        in tool["function"]["parameters"]["properties"]["surface"]["enum"]
    )


async def test_render_surface_works_after_refresh(tmp_path):
    pkg_dir = tmp_path / "pkg"
    lumen_dir = tmp_path / "lumen"
    (lumen_dir / "surfaces").mkdir(parents=True)
    _write_surface(pkg_dir, "hello")

    brain = _BrainStub(lumen_dir, pkg_dir)
    apply_hello(
        brain.registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "hello", "version": "1.0.0"}],
    )
    _refresh(brain, lumen_dir, pkg_dir)

    result = await execute_render_surface(
        brain.registry,
        surface="hello",
        data={"items": []},
        render=lambda surface, data, render_id: {"ok": True, "surface": surface},
    )

    assert result["ok"] is True
    assert result["surface"] == "hello"


def test_refresh_does_not_promote_stale_confirmation(tmp_path):
    pkg_dir = tmp_path / "pkg"
    lumen_dir = tmp_path / "lumen"
    (lumen_dir / "surfaces").mkdir(parents=True)
    _write_surface(pkg_dir, "hello")

    brain = _BrainStub(lumen_dir, pkg_dir)
    apply_hello(
        brain.registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "hello", "version": "1.0.0"}],
    )
    # Confirmación vieja de una superficie que ya no existe (renombrada).
    store = brain.registry._surface_renderers  # noqa: SLF001
    store["shell-1"]["ghost"] = RendererConfirmation(
        name="ghost", version="1.0.0", host_kind="electron", connection_id="shell-1"
    )

    _refresh(brain, lumen_dir, pkg_dir)

    assert brain.registry.get(CapabilityKind.SURFACE, "ghost") is None
    tool = build_render_surface_tool(brain.registry)
    enum = tool["function"]["parameters"]["properties"]["surface"]["enum"]
    assert "ghost" not in enum
    assert "hello" in enum


def test_refresh_does_not_promote_unconfirmed_surface(tmp_path):
    pkg_dir = tmp_path / "pkg"
    lumen_dir = tmp_path / "lumen"
    (lumen_dir / "surfaces").mkdir(parents=True)
    _write_surface(pkg_dir, "timeline")
    _write_surface(pkg_dir, "hello")

    brain = _BrainStub(lumen_dir, pkg_dir)
    apply_hello(
        brain.registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "hello", "version": "1.0.0"}],
    )

    _refresh(brain, lumen_dir, pkg_dir)

    hello = brain.registry.get(CapabilityKind.SURFACE, "hello")
    timeline = brain.registry.get(CapabilityKind.SURFACE, "timeline")
    assert hello.status == CapabilityStatus.READY
    assert timeline.status == CapabilityStatus.NO_RENDERER
