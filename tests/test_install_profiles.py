"""Packaging contracts for the lightweight Core and full Lumen profiles."""

from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as project_file:
        return tomllib.load(project_file)["project"]


def test_default_install_is_the_litellm_free_core_profile():
    """`pip install enlumen` must not install the heavyweight provider stack."""
    dependencies = _project_metadata()["dependencies"]

    assert not any(dependency.lower().startswith("litellm") for dependency in dependencies)


def test_full_extra_keeps_litellm_provider_compatibility():
    """Full Lumen deliberately opts in to the legacy multi-provider adapter."""
    extras = _project_metadata()["optional-dependencies"]

    assert "full" in extras
    assert any(dependency.lower().startswith("litellm") for dependency in extras["full"])


def test_core_dockerfile_installs_the_default_litellm_free_profile():
    dockerfile = (ROOT / "Dockerfile.core").read_text(encoding="utf-8")

    assert "pip install --no-cache-dir ." in dockerfile
    assert ".[full]" not in dockerfile
    assert "nodesource.com" not in dockerfile


def test_full_dockerfile_explicitly_installs_the_full_extra():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert 'pip install --no-cache-dir ".[full]"' in dockerfile
