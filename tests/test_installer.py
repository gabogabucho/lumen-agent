"""Tests del Bloque A — el Installer resuelve desde la fuente correcta.

Cubre .refs/onboarding-catalogo/tasks.md (Bloque A):
- `test_install_from_extra_source_uses_own_root`: un módulo vendoreado en una
  fuente extra se instala desde esa fuente, no desde `lumen/catalog/`
- `test_path_traversal_rejected`: `path: ../secreto` no copia nada
- el fallback de manifest generado sigue funcionando sin `path`
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.installer import Installer


def _write_index(path: Path, modules: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"modules": modules}), encoding="utf-8")
    return path


def _write_vendored_module(source_root: Path, relative: str, name: str) -> Path:
    module_dir = source_root / relative
    module_dir.mkdir(parents=True, exist_ok=True)
    (module_dir / "module.yaml").write_text(
        yaml.safe_dump(
            {
                "name": name,
                "display_name": "Módulo vendoreado",
                "description": "Vive en una fuente de catálogo adicional",
                "version": "1.0.0",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (module_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
    return module_dir


def _make_installer(pkg_dir: Path, lumen_dir: Path, catalog: Catalog) -> Installer:
    lumen_dir.mkdir(parents=True, exist_ok=True)
    return Installer(
        pkg_dir,
        ConnectorRegistry(),
        memory=None,
        catalog=catalog,
        lumen_dir=lumen_dir,
    )


def test_install_from_extra_source_uses_own_root(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_index = _write_index(pkg_dir / "catalog" / "index.yaml", [])

    extra_root = tmp_path / "modules-pegaseum"
    _write_vendored_module(extra_root, "fuentes/x-pegaseum-demo", "x-pegaseum-demo")
    extra_index = _write_index(
        extra_root / "index.yaml",
        [
            {
                "name": "x-pegaseum-demo",
                "display_name": "Módulo vendoreado",
                "description": "Vive en una fuente de catálogo adicional",
                "path": "fuentes/x-pegaseum-demo",
            }
        ],
    )

    catalog = Catalog(pkg_index, extra_paths=[extra_index])
    installer = _make_installer(pkg_dir, tmp_path / "lumen", catalog)

    result = installer.install_from_catalog("x-pegaseum-demo")

    assert result["status"] == "installed"
    installed = installer.installed_dir / "x-pegaseum-demo"
    assert (installed / "module.yaml").exists()
    assert (installed / "SKILL.md").read_text(encoding="utf-8") == "# x-pegaseum-demo\n"
    # El catálogo del paquete está vacío: la copia vino de la fuente extra.
    assert not (pkg_dir / "catalog" / "fuentes" / "x-pegaseum-demo").exists()


def test_path_traversal_rejected(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_index = _write_index(pkg_dir / "catalog" / "index.yaml", [])

    extra_root = tmp_path / "extra"
    secreto = tmp_path / "secreto"
    secreto.mkdir()
    (secreto / "clave.txt").write_text("no-copiar", encoding="utf-8")

    extra_index = _write_index(
        extra_root / "index.yaml",
        [{"name": "x-evil", "description": "traversal", "path": "../secreto"}],
    )

    catalog = Catalog(pkg_index, extra_paths=[extra_index])
    installer = _make_installer(pkg_dir, tmp_path / "lumen", catalog)

    resolved = installer._resolve_catalog_module_dir(catalog.get("x-evil"))
    assert resolved.name == "__invalid__"

    result = installer.install_from_catalog("x-evil")

    assert result["status"] in ("error", "not_found")
    assert not (installer.installed_dir / "x-evil").exists()


def test_entry_without_path_generates_manifest(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_index = _write_index(
        pkg_dir / "catalog" / "index.yaml",
        [
            {
                "name": "x-solo-metadata",
                "display_name": "Solo metadata",
                "description": "No trae archivos",
                "fills_gaps": ["llenar un hueco"],
            }
        ],
    )

    catalog = Catalog(pkg_index, extra_paths=[])
    installer = _make_installer(pkg_dir, tmp_path / "lumen", catalog)

    result = installer.install_from_catalog("x-solo-metadata")

    assert result["status"] == "installed"
    manifest_path = installer.installed_dir / "x-solo-metadata" / "module.yaml"
    assert manifest_path.exists()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "x-solo-metadata"
    assert (installer.installed_dir / "x-solo-metadata" / "SKILL.md").exists()


def test_reinstall_is_idempotent(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_index = _write_index(
        pkg_dir / "catalog" / "index.yaml",
        [{"name": "x-unico", "description": "d"}],
    )
    catalog = Catalog(pkg_index, extra_paths=[])
    installer = _make_installer(pkg_dir, tmp_path / "lumen", catalog)

    first = installer.install_from_catalog("x-unico")
    second = installer.install_from_catalog("x-unico")

    assert first["status"] == "installed"
    assert second["status"] == "already_installed"
