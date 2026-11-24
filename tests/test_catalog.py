"""Tests del Bloque A — enchufe E5: fuentes de catálogo y use_cases.

Cubre .refs/onboarding-catalogo/tasks.md (Bloque A):
- fusión de un index extra con el del paquete (overlay sin `path`)
- `catalog_paths_from_config`: `config.catalog.sources` + `LUMEN_CATALOG_PATHS`
- YAML roto falla nombrando el archivo; ruta inexistente solo avisa
- `use_cases` en `search` y `as_context` (incluido el filtro de instalados)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest
import yaml

from lumen.core.catalog import Catalog, catalog_paths_from_config


def _write_index(path: Path, modules: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"modules": modules}), encoding="utf-8")
    return path


# ── Fusión por nombre ─────────────────────────────────────────────────


def test_extra_sources_merge_by_name(tmp_path):
    pkg_index = _write_index(
        tmp_path / "pkg-catalog" / "index.yaml",
        [
            {
                "name": "x-lumen-comunicacion-whatsapp",
                "display_name": "WhatsApp",
                "description": "Enviar mensajes de WhatsApp",
                "price": "free",
                "path": "modules/x-lumen-comunicacion-whatsapp",
                "tags": ["x-lumen", "whatsapp"],
            }
        ],
    )
    extra_index = _write_index(
        tmp_path / "extra" / "index.yaml",
        [
            {
                "name": "x-lumen-comunicacion-whatsapp",
                "use_cases": ["publicar en whatsapp"],
                "recomendado": True,
            }
        ],
    )

    catalog = Catalog(pkg_index, extra_paths=[extra_index])
    mod = catalog.get("x-lumen-comunicacion-whatsapp")

    assert mod is not None
    # El overlay aporta sus campos...
    assert mod["use_cases"] == ["publicar en whatsapp"]
    assert mod["recomendado"] is True
    # ...pero no redirige la instalación: path y source_root siguen en el paquete.
    assert mod["path"] == "modules/x-lumen-comunicacion-whatsapp"
    assert catalog.source_root("x-lumen-comunicacion-whatsapp") == pkg_index.parent.resolve()
    assert catalog.source_root("x-lumen-comunicacion-whatsapp") != extra_index.parent.resolve()


def test_extra_only_module_resolves_to_its_own_source(tmp_path):
    pkg_index = _write_index(tmp_path / "pkg-catalog" / "index.yaml", [])
    extra_index = _write_index(
        tmp_path / "extra" / "index.yaml",
        [
            {
                "name": "x-pegaseum-demo",
                "description": "Módulo vendoreado en la fuente extra",
                "path": "fuentes/x-pegaseum-demo",
            }
        ],
    )

    catalog = Catalog(pkg_index, extra_paths=[extra_index])

    assert catalog.get("x-pegaseum-demo") is not None
    assert catalog.source_root("x-pegaseum-demo") == extra_index.parent.resolve()


def test_source_root_is_none_for_unknown_module(tmp_path):
    pkg_index = _write_index(tmp_path / "pkg-catalog" / "index.yaml", [])
    catalog = Catalog(pkg_index, extra_paths=[])
    assert catalog.source_root("no-existe") is None


# ── Errores y warnings de las fuentes ─────────────────────────────────


def test_invalid_extra_yaml_fails_naming_the_file(tmp_path):
    pkg_index = _write_index(tmp_path / "pkg-catalog" / "index.yaml", [])
    bad = tmp_path / "extra" / "index.yaml"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("modules: [ {name: 'x'\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        Catalog(pkg_index, extra_paths=[bad])

    assert str(bad) in str(excinfo.value)


def test_missing_extra_path_only_warns(tmp_path, caplog):
    pkg_index = _write_index(
        tmp_path / "pkg-catalog" / "index.yaml",
        [{"name": "x-lumen-solo-paquete", "description": "ok"}],
    )
    missing = tmp_path / "no-existe" / "index.yaml"

    with caplog.at_level(logging.WARNING):
        catalog = Catalog(pkg_index, extra_paths=[missing])

    assert catalog.get("x-lumen-solo-paquete") is not None
    assert any("no-existe" in record.getMessage() for record in caplog.records)


# ── catalog_paths_from_config ─────────────────────────────────────────


def test_config_sources_resolve_relative_to_lumen_dir(tmp_path):
    lumen_dir = tmp_path / "lumen"
    config = {"catalog": {"sources": ["catalogo-local/index.yaml"]}}

    paths = catalog_paths_from_config(config, lumen_dir=lumen_dir, env={})

    assert paths == [lumen_dir / "catalogo-local" / "index.yaml"]


def test_env_splits_on_pathsep_and_comma(tmp_path):
    lumen_dir = tmp_path / "lumen"
    a = tmp_path / "a" / "index.yaml"
    b = tmp_path / "b" / "index.yaml"
    c = tmp_path / "c" / "index.yaml"
    config = {"catalog": {"sources": [str(a)]}}

    paths = catalog_paths_from_config(
        config, lumen_dir=lumen_dir, env={"LUMEN_CATALOG_PATHS": f"{b}{os.pathsep}{c}"}
    )
    assert paths == [a, b, c]

    paths = catalog_paths_from_config(
        config, lumen_dir=lumen_dir, env={"LUMEN_CATALOG_PATHS": f"{b},{c}"}
    )
    assert paths == [a, b, c]


def test_env_paths_come_after_config_paths(tmp_path):
    lumen_dir = tmp_path / "lumen"
    config = {"catalog": {"sources": ["cfg/index.yaml"]}}
    env_index = tmp_path / "env" / "index.yaml"

    paths = catalog_paths_from_config(
        config, lumen_dir=lumen_dir, env={"LUMEN_CATALOG_PATHS": str(env_index)}
    )

    # Orden: config primero, env después → el env gana en la fusión por nombre.
    assert paths == [lumen_dir / "cfg" / "index.yaml", env_index]


def test_env_wins_over_config_on_name_conflict(tmp_path):
    pkg_index = _write_index(tmp_path / "pkg-catalog" / "index.yaml", [])
    cfg_index = _write_index(
        tmp_path / "cfg" / "index.yaml",
        [{"name": "x-demo", "description": "desde config"}],
    )
    env_index = _write_index(
        tmp_path / "env" / "index.yaml",
        [{"name": "x-demo", "description": "desde env"}],
    )

    paths = catalog_paths_from_config(
        {"catalog": {"sources": [str(cfg_index)]}},
        env={"LUMEN_CATALOG_PATHS": str(env_index)},
    )
    catalog = Catalog(pkg_index, extra_paths=paths)

    assert catalog.get("x-demo")["description"] == "desde env"


def test_empty_config_returns_no_paths():
    assert catalog_paths_from_config({}, env={}) == []
    assert catalog_paths_from_config({"catalog": {}}, env={}) == []


# ── use_cases ─────────────────────────────────────────────────────────


def test_use_cases_in_search_and_context(tmp_path):
    index = _write_index(
        tmp_path / "catalog" / "index.yaml",
        [
            {
                "name": "x-lumen-redes",
                "display_name": "Redes",
                "description": "Publicar contenido en redes",
                "use_cases": ["publicar en instagram"],
                "fills_gaps": [],
            },
            {
                "name": "x-lumen-sin-casos",
                "display_name": "Otro",
                "description": "No tiene casos de uso",
                "use_cases": [],
            },
        ],
    )
    catalog = Catalog(index)

    results = catalog.search("publicar en instagram")
    assert [m["name"] for m in results] == ["x-lumen-redes"]

    context = catalog.as_context()
    assert "[casos: publicar en instagram]" in context


def test_as_context_limits_use_cases_to_three(tmp_path):
    index = _write_index(
        tmp_path / "catalog" / "index.yaml",
        [
            {
                "name": "x-lumen-redes",
                "description": "Redes",
                "use_cases": ["caso uno", "caso dos", "caso tres", "caso cuatro"],
            }
        ],
    )
    context = Catalog(index).as_context()

    assert "caso uno" in context
    assert "caso tres" in context
    assert "caso cuatro" not in context


def test_as_context_hides_installed_use_case_modules(tmp_path):
    index = _write_index(
        tmp_path / "catalog" / "index.yaml",
        [
            {
                "name": "x-lumen-redes",
                "description": "Redes",
                "use_cases": ["publicar en instagram"],
            }
        ],
    )
    catalog = Catalog(index)

    context = catalog.as_context(installed_names={"x-lumen-redes"})

    assert "x-lumen-redes" not in context
    assert "publicar en instagram" not in context
