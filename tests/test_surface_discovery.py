"""Tests del Bloque 4: discovery de superficies.

Cubre tasks.md 4.1 a 4.6:
- SurfaceManifest + load_surface_manifest con validación del schema (4.1)
- _discover_surfaces con la forma de _discover_modules_multi (4.2)
- el discovery nunca deja una superficie READY (4.3)
- forma de metadata (4.4)
- orden de precedencia de roots (4.5)
- enganche desde discover_all (4.6)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lumen.core import discovery
from lumen.core.connectors import ConnectorRegistry
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry
from lumen.core.surfaces import SurfaceManifest, load_surface_manifest


VALID_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["items"],
    "properties": {"items": {"type": "array"}},
}


def _write_surface(
    root: Path,
    name: str,
    *,
    schema: dict | None = None,
    schema_body: str | None = None,
    when_to_use: str | None = None,
    version: str = "1.0.0",
    contract: str | None = "x-lumen-desktop/3",
    renderer: str | None = "./renderer.js",
    interactions: list[str] | None = None,
    persistent: bool | None = True,
    manifest_version=1,
    with_name_field: bool = True,
) -> Path:
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    if schema_body is None:
        if schema is None:
            schema = VALID_SCHEMA
        schema_body = json.dumps(schema, ensure_ascii=False)
    (base / "schema.json").write_text(schema_body, encoding="utf-8")

    manifesto: dict = {
        "version": version,
        "description": f"Superficie {name}.",
        "schema": "./schema.json",
    }
    if manifest_version is not None:
        manifesto["manifest_version"] = manifest_version
    if with_name_field:
        manifesto["name"] = name
    if contract is not None:
        manifesto["contract"] = contract
    if when_to_use is not None:
        manifesto["when_to_use"] = when_to_use
    if renderer is not None:
        manifesto["renderer"] = renderer
    if interactions is not None:
        manifesto["interactions"] = interactions
    if persistent is not None:
        manifesto["persistent"] = persistent
    (base / "surface.yaml").write_text(yaml.safe_dump(manifesto), encoding="utf-8")
    return base


# --- 4.1: manifiesto y esquema ---


def test_load_surface_manifest_and_schema(tmp_path):
    base = _write_surface(
        tmp_path,
        "email_list",
        when_to_use="Cuando el usuario quiera ver o elegir entre varios correos.",
        interactions=["select", "open", "close"],
    )

    manifest = load_surface_manifest(base / "surface.yaml")

    assert isinstance(manifest, SurfaceManifest)
    assert manifest.name == "email_list"
    assert manifest.version == "1.0.0"
    assert manifest.contract == "x-lumen-desktop/3"
    assert manifest.description == "Superficie email_list."
    assert manifest.when_to_use == "Cuando el usuario quiera ver o elegir entre varios correos."
    assert manifest.schema is not None
    assert manifest.schema["type"] == "object"
    assert manifest.schema_path == str((base / "schema.json").resolve())
    assert manifest.renderer == "./renderer.js"
    assert manifest.interactions == ["select", "open", "close"]
    assert manifest.persistent is True
    assert manifest.manifest_version == 1
    assert manifest.schema_error is None


def test_load_surface_manifest_missing_schema(tmp_path):
    base = tmp_path / "sin_schema"
    base.mkdir()
    (base / "surface.yaml").write_text(
        yaml.safe_dump({"manifest_version": 1, "name": "sin_schema", "description": "x"}),
        encoding="utf-8",
    )

    manifest = load_surface_manifest(base / "surface.yaml")
    assert manifest.schema is None
    assert manifest.schema_error is not None
    assert "schema" in manifest.schema_error.lower()


def test_load_surface_manifest_schema_path_missing(tmp_path):
    base = _write_surface(tmp_path, "huérfana")
    (base / "schema.json").unlink()

    manifest = load_surface_manifest(base / "surface.yaml")
    assert manifest.schema is None
    assert manifest.schema_error is not None


@pytest.mark.parametrize(
    "body",
    [
        "{ esto no es json",
        json.dumps({"type": "no-es-un-tipo-valido", "properties": {"a": {"type": "string"}}}),
    ],
)
def test_load_surface_manifest_invalid_schema(tmp_path, body):
    base = _write_surface(tmp_path, "rota")
    (base / "schema.json").write_text(body, encoding="utf-8")

    manifest = load_surface_manifest(base / "surface.yaml")
    assert manifest.schema is None
    assert manifest.schema_error is not None


def test_load_surface_manifest_name_falls_back_to_directory(tmp_path):
    base = _write_surface(tmp_path, "titulo_raro", with_name_field=False)
    (base / "surface.yaml").write_text(
        (base / "surface.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    manifest = load_surface_manifest(base / "surface.yaml")
    assert manifest.name == "titulo_raro"


# --- 4.2: _discover_surfaces ---


def test_discover_surfaces_roots_and_dedup(tmp_path, caplog):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    _write_surface(lumen_dir / "surfaces", "email_list")
    _write_surface(pkg_dir / "surfaces", "timeline")
    _write_surface(pkg_dir / "surfaces", "rota", schema_body="{ no es json")

    # Un directorio arrancado con _ se ignora; un manifiesto sin schema no aborta.
    _write_surface(pkg_dir / "surfaces", "_legacy")

    registry = Registry()
    discovery._discover_surfaces(registry, [lumen_dir / "surfaces", pkg_dir / "surfaces"])

    names = {c.name for c in registry.list_by_kind(CapabilityKind.SURFACE)}
    assert names == {"email_list", "timeline", "rota"}

    email = registry.get(CapabilityKind.SURFACE, "email_list")
    timeline = registry.get(CapabilityKind.SURFACE, "timeline")
    rota = registry.get(CapabilityKind.SURFACE, "rota")
    assert email.status == CapabilityStatus.NO_RENDERER
    assert timeline.status == CapabilityStatus.NO_RENDERER
    assert rota.status == CapabilityStatus.MISSING_DEPS
    assert rota.metadata["error"]


def test_discover_surfaces_missing_root_is_silent(tmp_path):
    registry = Registry()
    # No lanza nada si el root no existe.
    discovery._discover_surfaces(registry, [tmp_path / "no_existe"])
    assert registry.all() == []


# --- 4.3: el filesystem por sí solo nunca deja una superficie READY ---


def test_surface_never_ready_from_filesystem(tmp_path):
    root = tmp_path / "surfaces"
    _write_surface(root, "email_list")
    _write_surface(root, "sin_schema", schema=None, schema_body="{}")
    (tmp_path / "surfaces" / "sin_schema" / "surface.yaml").write_text(
        yaml.safe_dump({"manifest_version": 1, "name": "sin_schema", "description": "x"}),
        encoding="utf-8",
    )

    registry = Registry()
    discovery._discover_surfaces(registry, [root])

    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.NO_RENDERER
    assert registry.get(CapabilityKind.SURFACE, "sin_schema").status == CapabilityStatus.MISSING_DEPS
    assert registry.ready() == []


# --- 4.4: metadata ---


def test_surface_metadata_shape(tmp_path):
    lumen_dir = tmp_path / "lumen"
    module_src = tmp_path / "modules" / "mi-modulo"
    _write_surface(
        module_src / "surfaces",
        "timeline",
        when_to_use="Cuando el usuario quiera seguir el hilo del tiempo.",
    )

    registry = Registry()
    discovery._discover_surfaces(
        registry,
        [lumen_dir / "surfaces", module_src / "surfaces"],
        module_name_by_root={str(module_src / "surfaces"): "mi-modulo"},
    )

    cap = registry.get(CapabilityKind.SURFACE, "timeline")
    meta = cap.metadata
    assert meta["version"] == "1.0.0"
    assert meta["contract"] == "x-lumen-desktop/3"
    assert meta["when_to_use"] == "Cuando el usuario quiera seguir el hilo del tiempo."
    assert meta["schema_path"] == str((module_src / "surfaces" / "timeline" / "schema.json").resolve())
    assert meta["schema"] == VALID_SCHEMA
    assert meta["renderer"] == "./renderer.js"
    assert meta["interactions"] == []
    assert meta["persistent"] is True
    assert meta["path"] == str(module_src / "surfaces" / "timeline")
    assert meta["module_name"] == "mi-modulo"


# --- 4.5: precedencia de roots ---


def test_surface_roots_precedence(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"

    _write_surface(pkg_dir / "surfaces", "email_list", version="0.1.0")
    _write_surface(lumen_dir / "surfaces", "email_list", version="9.9.9")

    registry = Registry()
    discovery._discover_surfaces(registry, [lumen_dir / "surfaces", pkg_dir / "surfaces"])

    # El primero en la lista gana si el nombre se repite.
    assert registry.get(CapabilityKind.SURFACE, "email_list").metadata["version"] == "9.9.9"


def test_surfaces_of_modules_lose_against_global_roots(tmp_path):
    pkg_dir = tmp_path / "pkg"
    module_dir = pkg_dir / "modules" / "kit"
    lumen_dir = tmp_path / "lumen"

    _write_surface(lumen_dir / "surfaces", "post_preview", version="global")
    _write_surface(module_dir / "surfaces", "post_preview", version="del-kit")

    registry = Registry()
    discovery.discover_all(registry, pkg_dir, ConnectorRegistry(), lumen_dir=lumen_dir)

    assert registry.get(CapabilityKind.SURFACE, "post_preview").metadata["version"] == "global"


# --- 4.6: enganche en discover_all ---


def test_discover_all_registers_surfaces(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    lumen_dir = tmp_path / "lumen"
    lumen_dir.mkdir()
    _write_surface(lumen_dir / "surfaces", "email_list")

    registry = Registry()
    result = discovery.discover_all(
        registry,
        pkg_dir,
        ConnectorRegistry(),
        lumen_dir=lumen_dir,
    )

    assert result is registry
    cap = registry.get(CapabilityKind.SURFACE, "email_list")
    assert cap is not None
    assert cap.status == CapabilityStatus.NO_RENDERER


def test_discover_all_registers_module_surfaces(tmp_path):
    pkg_dir = tmp_path / "pkg"
    module_dir = pkg_dir / "modules" / "mi-modulo"
    _write_surface(module_dir / "surfaces", "timeline")
    (module_dir / "module.yaml").write_text(
        yaml.safe_dump({"name": "mi-modulo", "description": "kit de prueba"}),
        encoding="utf-8",
    )

    registry = Registry()
    discovery.discover_all(registry, pkg_dir, ConnectorRegistry())

    cap = registry.get(CapabilityKind.SURFACE, "timeline")
    assert cap is not None
    assert cap.metadata["module_name"] == "mi-modulo"


# --- contrato de versión en superficies ---


def test_surface_with_unknown_version_registers_error(tmp_path):
    root = tmp_path / "surfaces"
    _write_surface(root, "futura", manifest_version=9)

    registry = Registry()
    discovery._discover_surfaces(registry, [root])

    cap = registry.get(CapabilityKind.SURFACE, "futura")
    assert cap.status == CapabilityStatus.ERROR
    assert cap.metadata["error"] == "unsupported manifest_version 9; supported: 1"
    # Un contrato incompatible no aporta superficies: no hay enum de render.
    from lumen.core.surfaces import build_render_surface_tool

    assert build_render_surface_tool(registry) is None
