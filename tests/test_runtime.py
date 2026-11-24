"""Tests del Bloque A — bootstrap con fuentes de catálogo adicionales.

Cubre .refs/onboarding-catalogo/tasks.md (Bloque A):
- `bootstrap_runtime` pasa `extra_paths=catalog_paths_from_config(...)`
- `config.catalog.sources` se resuelve contra `lumen_dir`
- `LUMEN_CATALOG_PATHS` (env) gana sobre la config sin editar archivos
- demo punta a punta: fuente extra → instalación en caliente → sync + refresh
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from lumen.core.installer import Installer
from lumen.core.registry import CapabilityKind, CapabilityStatus
from lumen.core.runtime import (
    bootstrap_runtime,
    refresh_runtime_registry,
    sync_runtime_modules,
)
from lumen.core.surfaces import apply_hello

CONTRACT_V3 = "x-lumen-desktop/3"


def _package_index_names() -> set[str]:
    """Nombres definidos en el index del paquete (lumen/catalog/index.yaml)."""
    index = Path(__file__).resolve().parents[1] / "lumen" / "catalog" / "index.yaml"
    data = yaml.safe_load(index.read_text(encoding="utf-8")) or {}
    return {m["name"] for m in data.get("modules", [])}


def _write_index(path: Path, modules: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"modules": modules}), encoding="utf-8")
    return path


def _make_runtime_pkg(temp_root: Path) -> Path:
    pkg_dir = temp_root / "pkg"
    (pkg_dir / "locales" / "en" / "flows").mkdir(parents=True)
    (pkg_dir / "catalog").mkdir(parents=True)
    (pkg_dir / "locales" / "en" / "personality.yaml").write_text(
        yaml.safe_dump({"identity": {"name": "Locale Lumen", "role": "Assistant"}}),
        encoding="utf-8",
    )
    (pkg_dir / "locales" / "en" / "flows" / "default.yaml").write_text(
        yaml.safe_dump({"intent": "locale-default", "triggers": ["hello"]}),
        encoding="utf-8",
    )
    _write_index(pkg_dir / "catalog" / "index.yaml", [])
    return pkg_dir


async def test_bootstrap_loads_config_catalog_sources(tmp_path, monkeypatch):
    monkeypatch.delenv("LUMEN_CATALOG_PATHS", raising=False)
    pkg_dir = _make_runtime_pkg(tmp_path)
    lumen_dir = tmp_path / "lumen"
    _write_index(
        lumen_dir / "catalogo-local" / "index.yaml",
        [{"name": "x-local-demo", "description": "Fuente local del cliente"}],
    )

    config = {
        "language": "en",
        "model": "deepseek/deepseek-chat",
        "catalog": {"sources": ["catalogo-local/index.yaml"]},
    }
    runtime = await bootstrap_runtime(
        config, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"]
    )

    try:
        assert runtime.brain.catalog.get("x-local-demo") is not None
        # El index del paquete sigue cargado primero.
        package_names = _package_index_names()
        assert package_names <= {m["name"] for m in runtime.brain.catalog.modules}
    finally:
        await runtime.brain.memory.close()


async def test_bootstrap_env_catalog_paths_win_over_config(tmp_path, monkeypatch):
    pkg_dir = _make_runtime_pkg(tmp_path)
    lumen_dir = tmp_path / "lumen"
    lumen_dir.mkdir(parents=True, exist_ok=True)

    env_index = _write_index(
        tmp_path / "env-source" / "index.yaml",
        [{"name": "x-desde-env", "description": "Cargada por LUMEN_CATALOG_PATHS"}],
    )
    _write_index(
        lumen_dir / "config-source" / "index.yaml",
        [{"name": "x-desde-config", "description": "Debería seguir disponible"}],
    )
    monkeypatch.setenv("LUMEN_CATALOG_PATHS", str(env_index))

    config = {
        "language": "en",
        "model": "deepseek/deepseek-chat",
        "catalog": {"sources": ["config-source/index.yaml"]},
    }
    runtime = await bootstrap_runtime(
        config, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"]
    )

    try:
        names = {m["name"] for m in runtime.brain.catalog.modules}
        # Env y config conviven; el env va después y gana en conflictos de nombre.
        assert "x-desde-env" in names
        assert "x-desde-config" in names
    finally:
        await runtime.brain.memory.close()


async def test_bootstrap_without_catalog_sources_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("LUMEN_CATALOG_PATHS", raising=False)
    pkg_dir = _make_runtime_pkg(tmp_path)
    lumen_dir = tmp_path / "lumen"
    lumen_dir.mkdir(parents=True, exist_ok=True)

    runtime = await bootstrap_runtime(
        {"language": "en", "model": "deepseek/deepseek-chat"},
        pkg_dir=pkg_dir,
        lumen_dir=lumen_dir,
        active_channels=["web"],
    )

    try:
        # Sin sources ni env: solo el index del paquete, idéntico al archivo.
        assert {m["name"] for m in runtime.brain.catalog.modules} == _package_index_names()
    finally:
        await runtime.brain.memory.close()


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


async def test_install_from_extra_source_end_to_end(tmp_path, monkeypatch):
    """Demo punta a punta: fuente extra → instalación en caliente → refresh.

    - el módulo vive en una fuente de catálogo adicional, no en `lumen/catalog`
    - `install_from_catalog` lo copia desde esa fuente
    - `sync_runtime_modules` + `refresh_runtime_registry` lo registran sin reiniciar
    - la superficie confirmada antes del refresh sigue `READY`
    """
    monkeypatch.delenv("LUMEN_CATALOG_PATHS", raising=False)
    pkg_dir = _make_runtime_pkg(tmp_path)
    _write_surface(pkg_dir, "hello")

    lumen_dir = tmp_path / "lumen"
    extra_root = lumen_dir / "modules-pegaseum"
    module_dir = extra_root / "fuentes" / "x-pegaseum-demo"
    module_dir.mkdir(parents=True)
    (module_dir / "module.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "x-pegaseum-demo",
                "display_name": "Demo Pegaseum",
                "description": "Módulo vendoreado en una fuente de catálogo adicional",
                "version": "1.0.0",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (module_dir / "SKILL.md").write_text(
        "---\n"
        "name: x-pegaseum-demo\n"
        'description: "Demo vendoreada"\n'
        "---\n"
        "# Demo\n",
        encoding="utf-8",
    )
    _write_index(
        extra_root / "index.yaml",
        [
            {
                "name": "x-pegaseum-demo",
                "display_name": "Demo Pegaseum",
                "description": "Módulo vendoreado en una fuente de catálogo adicional",
                "path": "fuentes/x-pegaseum-demo",
            }
        ],
    )

    config = {
        "language": "en",
        "model": "deepseek/deepseek-chat",
        "catalog": {"sources": ["modules-pegaseum/index.yaml"]},
    }
    runtime = await bootstrap_runtime(
        config, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"]
    )
    brain = runtime.brain

    try:
        assert brain.catalog.get("x-pegaseum-demo") is not None

        # La shell confirma la superficie, como en producción.
        apply_hello(
            brain.registry,
            connection_id="shell-1",
            contract=CONTRACT_V3,
            renderers=[{"name": "hello", "version": "1.0.0"}],
        )
        hello = brain.registry.get(CapabilityKind.SURFACE, "hello")
        assert hello.status == CapabilityStatus.READY

        installer = Installer(
            pkg_dir,
            brain.connectors,
            brain.memory,
            catalog=brain.catalog,
            lumen_dir=lumen_dir,
            config=config,
        )
        result = installer.install_from_catalog("x-pegaseum-demo")
        assert result["status"] == "installed"
        assert (lumen_dir / "modules" / "x-pegaseum-demo" / "SKILL.md").exists()

        # En caliente, sin reiniciar: sync + refresh.
        await sync_runtime_modules(
            brain, config=config, pkg_dir=pkg_dir, lumen_dir=lumen_dir
        )
        refresh_runtime_registry(
            brain, pkg_dir=pkg_dir, lumen_dir=lumen_dir, active_channels=["web"]
        )

        module_names = {
            cap.name for cap in brain.registry.list_by_kind(CapabilityKind.MODULE)
        }
        skill_names = {
            cap.name for cap in brain.registry.list_by_kind(CapabilityKind.SKILL)
        }
        assert "x-pegaseum-demo" in module_names
        assert "x-pegaseum-demo" in skill_names

        # El refresh no bajó la superficie confirmada.
        hello = brain.registry.get(CapabilityKind.SURFACE, "hello")
        assert hello.status == CapabilityStatus.READY
    finally:
        await brain.memory.close()
