"""Shared module manifest resolution helpers.

Preferred manifest order:
1. module.yaml
2. manifest.yaml
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

import yaml


MANIFEST_FILENAMES = ("module.yaml", "manifest.yaml")


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
