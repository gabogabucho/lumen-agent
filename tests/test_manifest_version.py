"""Tests del Bloque 3: versionado del contrato de módulo (manifest_version).

Cubre tasks.md 3.1 a 3.5:
- parse_manifest_version / validate_manifest_version con las formas aceptadas (3.1)
- lumen_api como alias y su conflicto (3.2)
- manifiesto sin campo: v1 por defecto con un solo warning por path (3.3)
- major desconocido: capacidad en ERROR con metadata.error y aislamiento (3.4, 3.5)
"""

from __future__ import annotations

import logging

import pytest

from lumen.core import discovery
from lumen.core.module_manifest import (
    SUPPORTED_MANIFEST_VERSION,
    ManifestVersionError,
    parse_manifest_version,
    validate_manifest_version,
)
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry


# --- 3.1: parseo y validación ---


def test_parse_and_validate():
    assert SUPPORTED_MANIFEST_VERSION == 1
    assert parse_manifest_version({"manifest_version": 1}) == (1, False)
    assert parse_manifest_version({"manifest_version": "1"}) == (1, False)
    assert parse_manifest_version({"manifest_version": "1.2.3"}) == (1, False)
    assert validate_manifest_version({"manifest_version": 1}, path="/tmp/uno/module.yaml") == 1


def test_parse_rejects_garbage_version():
    with pytest.raises(ManifestVersionError):
        parse_manifest_version({"manifest_version": "uno"})


# --- 3.2: alias lumen_api ---


def test_lumen_api_alias_and_conflict():
    assert parse_manifest_version({"lumen_api": 1}) == (1, False)
    assert parse_manifest_version({"lumen_api": "1.0.0"}) == (1, False)
    assert parse_manifest_version({"manifest_version": 1, "lumen_api": 1}) == (1, False)

    with pytest.raises(ManifestVersionError):
        parse_manifest_version({"manifest_version": 1, "lumen_api": 2})


# --- 3.3: sin campo → v1 con un solo warning por path ---


def test_missing_defaults_to_v1_with_single_warning(tmp_path, caplog):
    manifesto = {"name": "core"}  # el catálogo de siempre no declara versión
    assert parse_manifest_version(manifesto) == (1, True)

    ruta = str(tmp_path / "module.yaml")
    with caplog.at_level(logging.WARNING):
        assert validate_manifest_version(manifesto, path=ruta) == 1
        assert validate_manifest_version(manifesto, path=ruta) == 1

    avisos = [r for r in caplog.records if "manifest_version" in r.getMessage()]
    assert len(avisos) == 1

    # Otro path distinto sí avisa: el warning es una vez por path.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        validate_manifest_version(manifesto, path=str(tmp_path / "otro" / "module.yaml"))
    otros = [r for r in caplog.records if "manifest_version" in r.getMessage()]
    assert len(otros) == 1


# --- 3.4: major desconocido ---


def test_unknown_major_message():
    with pytest.raises(ManifestVersionError) as excinfo:
        validate_manifest_version({"manifest_version": 2}, path="/tmp/malo/module.yaml")
    assert str(excinfo.value) == "unsupported manifest_version 2; supported: 1"


def test_unknown_major_registers_error(tmp_path):
    module_dir = tmp_path / "malo"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: malo\nmanifest_version: 2\ndescription: x\n", encoding="utf-8")

    registry = Registry()
    discovery._discover_modules_multi(registry, [tmp_path])

    cap = registry.get(CapabilityKind.MODULE, "malo")
    assert cap is not None
    assert cap.status == CapabilityStatus.ERROR
    assert cap.metadata["error"] == "unsupported manifest_version 2; supported: 1"


def test_unknown_major_via_alias_registers_error(tmp_path):
    module_dir = tmp_path / "malo_alias"
    module_dir.mkdir()
    (module_dir / "module.yaml").write_text("name: malo_alias\nlumen_api: 2\n", encoding="utf-8")

    registry = Registry()
    discovery._discover_modules_multi(registry, [tmp_path])
    assert registry.get(CapabilityKind.MODULE, "malo_alias").status == CapabilityStatus.ERROR


# --- 3.5: aislamiento entre módulos del mismo root ---


def test_invalid_manifest_does_not_abort_other_modules(tmp_path):
    malo = tmp_path / "malo"
    malo.mkdir()
    (malo / "module.yaml").write_text("name: malo\nmanifest_version: 9\n", encoding="utf-8")

    bueno = tmp_path / "bueno"
    bueno.mkdir()
    (bueno / "module.yaml").write_text("name: bueno\ndescription: anda bien\n", encoding="utf-8")
    (bueno / "SKILL.md").write_text("---\nname: bueno\n---\nCuerpo\n", encoding="utf-8")

    registry = Registry()
    discovery._discover_modules_multi(registry, [tmp_path])

    assert registry.get(CapabilityKind.MODULE, "malo").status == CapabilityStatus.ERROR
    assert registry.get(CapabilityKind.MODULE, "bueno") is not None
