"""Shared module manifest resolution helpers.

Preferred manifest order:
1. module.yaml
2. manifest.yaml

Contrato de versión: todo manifiesto declara ``manifest_version``. El alias
``lumen_api`` sigue aceptándose por compatibilidad con el catálogo anterior.
"""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

logger = logging.getLogger(__name__)

MANIFEST_FILENAMES = ("module.yaml", "manifest.yaml")

SUPPORTED_MANIFEST_VERSION = 1

_paths_warned_missing_version: set[str] = set()


class ManifestVersionError(ValueError):
    """El manifiesto declara una versión de contrato que no se puede cargar."""


def _manifest_major(value: Any, field_name: str) -> int:
    """Acepta entero o string con semver ("1", "1.2.3") y devuelve el major."""
    if isinstance(value, bool):
        raise ManifestVersionError(f"invalid {field_name}: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        major = value.strip().split(".", 1)[0].strip()
        if major.isdigit():
            return int(major)
    raise ManifestVersionError(f"invalid {field_name}: {value!r}")


def parse_manifest_version(manifest: dict[str, Any]) -> tuple[int, bool]:
    """Devuelve (major, defaulted) del manifiesto.

    - ``manifest_version`` es el campo canónico: entero o semver en string.
    - ``lumen_api`` es alias de compatibilidad; si ambos están y los majors
      difieren, es un conflicto de contrato.
    - Si no hay ninguno, se asume v1 (``defaulted=True``).
    """
    manifest_version = manifest.get("manifest_version")
    lumen_api = manifest.get("lumen_api")

    if manifest_version is None and lumen_api is None:
        return (SUPPORTED_MANIFEST_VERSION, True)

    if manifest_version is not None and lumen_api is not None:
        major_version = _manifest_major(manifest_version, "manifest_version")
        major_alias = _manifest_major(lumen_api, "lumen_api")
        if major_version != major_alias:
            raise ManifestVersionError(
                f"manifest_version {major_version} conflicts with lumen_api {major_alias}; supported: 1"
            )
        return (major_version, False)

    if manifest_version is not None:
        return (_manifest_major(manifest_version, "manifest_version"), False)
    return (_manifest_major(lumen_api, "lumen_api"), False)


def validate_manifest_version(manifest: dict[str, Any], *, path: str) -> int:
    """Valida la versión del manifiesto y devuelve su major.

    Lanza ManifestVersionError si el major no está soportado. En manifiestos
    sin versión avisa una sola vez por path y sigue con v1.
    """
    major, defaulted = parse_manifest_version(manifest)

    if defaulted and path and path not in _paths_warned_missing_version:
        _paths_warned_missing_version.add(path)
        logger.warning("manifest missing manifest_version; assuming v1: %s", path)

    if major != SUPPORTED_MANIFEST_VERSION:
        raise ManifestVersionError(f"unsupported manifest_version {major}; supported: 1")

    return major

_RESERVED_FILENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
}


def _validate_capability_name(name: str) -> bool:
    """Validate that a capability name is safe to use as a directory name.

    Rejects path separators, null bytes, reserved filenames, and empty strings.
    """
    if not name or not isinstance(name, str):
        return False
    if "\x00" in name:
        return False
    if any(sep in name for sep in ("/", "\\", "..")):
        return False
    if name.upper() in _RESERVED_FILENAMES:
        return False
    return True


def parse_capabilities(manifest: dict[str, Any]) -> list[str]:
    """Parse and validate the ``capabilities`` list from a manifest dict.

    Returns a list of non-empty, trimmed capability name strings.
    Logs warnings for invalid entries but does not raise.
    """
    raw = manifest.get("capabilities")
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "capabilities must be a list, got %s", type(raw).__name__
        )
        return []

    result: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            logger.warning("capabilities items must be non-empty strings")
            continue
        result.append(item.strip())
    return result


def resolve_capability_paths(
    capability_names: list[str],
    *,
    lumen_dir: Path | None = None,
    pkg_dir: Path | None = None,
) -> list[Path]:
    """Resolve capability names to absolute paths.

    Looks in ``~/.lumen/capabilities/`` first, then in
    ``pkg_dir/capabilities/``.  Logs warnings for missing or invalid
    names but does not raise.
    """
    paths: list[Path] = []
    seen: set[str] = set()

    for name in capability_names:
        if not _validate_capability_name(name):
            logger.warning("Invalid capability name: %r", name)
            continue

        if name in seen:
            continue
        seen.add(name)

        resolved: Path | None = None
        if lumen_dir is not None:
            global_path = (lumen_dir / "capabilities" / name).resolve()
            if global_path.exists():
                resolved = global_path

        if resolved is None and pkg_dir is not None:
            pkg_path = (pkg_dir / "capabilities" / name).resolve()
            if pkg_path.exists():
                resolved = pkg_path

        if resolved is not None:
            paths.append(resolved)
        else:
            logger.warning("Capability not found: %s", name)

    return paths


def resolve_module_manifest_path(module_dir: Path) -> Path | None:
    """Return the preferred manifest path for a module directory."""
    for filename in MANIFEST_FILENAMES:
        manifest_path = module_dir / filename
        if manifest_path.exists():
            return manifest_path
    return None


def load_module_manifest(module_dir: Path) -> tuple[Path | None, dict[str, Any]]:
    """Load a module manifest from disk using the preferred resolution order."""
    manifest_path = resolve_module_manifest_path(module_dir)
    if manifest_path is None:
        return None, {}

    with open(manifest_path, encoding="utf-8") as f:
        return manifest_path, yaml.safe_load(f) or {}


def find_module_manifest_in_zip(names: list[str]) -> str | None:
    """Return the preferred manifest entry from a ZIP file."""
    for filename in MANIFEST_FILENAMES:
        for name in names:
            if PurePosixPath(name).name == filename:
                return name
    return None


def zip_manifest_root_prefix(manifest_path: str) -> str:
    """Return the directory prefix that contains the manifest entry."""
    parent = PurePosixPath(manifest_path).parent
    return "" if str(parent) == "." else f"{parent.as_posix()}/"
