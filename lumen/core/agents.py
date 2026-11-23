"""Especialistas (kits de agent): manifiesto ``agent.yaml`` y discovery.

Un especialista declara personalidad, skills, tools y superficies propias.
El discovery los lee y registra como ``CapabilityKind.AGENT``. "Descubierto"
nunca significa "usable": si una skill o tool que declara no resuelve, el
especialista queda en ``missing_deps`` con la lista de faltantes en
``metadata.missing``, hasta que el poller refresque.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lumen.core.cerebellum import (
    COMPAT_READY,
    build_runtime_surface,
    match_declared_tools,
)
from lumen.core.connectors import ConnectorRegistry
from lumen.core.module_manifest import validate_manifest_version
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry

logger = logging.getLogger(__name__)


@dataclass
class AgentManifest:
    """Lo que declara un ``agents/<name>/agent.yaml``."""

    name: str
    display_name: str
    description: str
    personality_path: str | None
    personality_error: str | None
    skills: list[str]
    tools: list[str]
    surfaces: list[str]
    min_capability: str
    model: str | None
    max_context_tokens: int | None
    manifest_version: int
    path: str


def _load_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and item.strip()]


def load_agent_manifest(path: Path) -> AgentManifest:
    """Carga y valida un ``agent.yaml``.

    Lanza si el YAML no parsea o si el ``manifest_version`` no está soportado;
    la personalidad problema no lanza: llega como ``personality_error`` para
    que el discovery registre el especialista en ``ERROR``.
    """
    path = Path(path)
    agent_dir = path.parent

    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"agent manifest must be a mapping: {path}")

    manifest_version = validate_manifest_version(manifest, path=str(path))

    name = str(manifest.get("name") or "").strip()
    if not name:
        name = agent_dir.name
        logger.warning("agent manifest without name; using directory name: %s", name)

    personality_error: str | None = None
    personality_path: str | None = None
    personality_ref = str(manifest.get("personality") or "").strip()
    if not personality_ref:
        personality_error = "missing personality reference"
    else:
        resolved = (agent_dir / personality_ref).resolve()
        if not resolved.is_file():
            personality_error = f"personality file not found: {personality_ref}"
        else:
            try:
                body = resolved.read_text(encoding="utf-8")
            except OSError as exc:
                body = ""
                personality_error = f"personality file could not be read: {exc}"
            if not body.strip():
                personality_error = f"personality file is empty: {personality_ref}"
            else:
                personality_path = str(resolved)

    return AgentManifest(
        name=name,
        display_name=str(manifest.get("display_name") or name),
        description=str(manifest.get("description") or ""),
        personality_path=personality_path,
        personality_error=personality_error,
        skills=_load_string_list(manifest.get("skills")),
        tools=_load_string_list(manifest.get("tools")),
        surfaces=_load_string_list(manifest.get("surfaces")),
        min_capability=str(manifest.get("min_capability") or "tier-1"),
        model=str(manifest["model"]) if manifest.get("model") else None,
        max_context_tokens=(
            int(manifest["max_context_tokens"]) if manifest.get("max_context_tokens") else None
        ),
        manifest_version=manifest_version,
        path=str(path),
    )


def register_agent_manifest(
    registry: Registry,
    manifest: AgentManifest,
    *,
    connectors: ConnectorRegistry | None = None,
    module_name: str | None = None,
) -> Capability:
    """Resuelve las dependencias del especialista y lo registra."""
    agent_dir = str(Path(manifest.path).parent)
    base_metadata: dict[str, Any] = {
        "display_name": manifest.display_name,
        "path": agent_dir,
        "module_name": module_name,
        "manifest_version": manifest.manifest_version,
    }

    if manifest.personality_error:
        capability = Capability(
            kind=CapabilityKind.AGENT,
            name=manifest.name,
            description=manifest.description,
            status=CapabilityStatus.ERROR,
            metadata={**base_metadata, "error": manifest.personality_error},
        )
        registry.register(capability)
        return capability

    missing: list[str] = []
    surfaces_pending: list[str] = []
    resolved_skills: list[str] = []
    resolved_surfaces: list[str] = []
    resolved_tools: list[str] = []

    for skill_name in manifest.skills:
        skill = registry.get(CapabilityKind.SKILL, skill_name)
        if skill is None or not skill.is_ready():
            # La segunda pasada del discovery puede marcar skills
            # missing_deps: el especialista no puede declararlas resueltas.
            if skill_name not in missing:
                missing.append(skill_name)
        else:
            resolved_skills.append(skill_name)

    if connectors is None:
        connectors = ConnectorRegistry()
    runtime_surface = build_runtime_surface(connectors, registry)
    for match in match_declared_tools(manifest.tools, runtime_surface):
        declared = str(match.get("declared") or "").strip()
        if not declared:
            continue
        if match.get("status") == COMPAT_READY:
            resolved_tools.append(declared)
        elif declared not in missing:
            missing.append(declared)

    for surface_name in manifest.surfaces:
        surface = registry.get(CapabilityKind.SURFACE, surface_name)
        if surface is None:
            if surface_name not in missing:
                missing.append(surface_name)
            continue
        resolved_surfaces.append(surface_name)
        if not surface.is_ready() and surface_name not in surfaces_pending:
            # No es obligatorio que esté READY: el especialista produce y cae
            # a artifact; se anota como pendiente para no prometer de más.
            surfaces_pending.append(surface_name)

    metadata: dict[str, Any] = {
        **base_metadata,
        "personality_path": manifest.personality_path,
        "skills": resolved_skills,
        "tools": resolved_tools,
        "surfaces": resolved_surfaces,
        "min_capability": manifest.min_capability,
        "model": manifest.model,
        "max_context_tokens": manifest.max_context_tokens,
    }
    if missing:
        metadata["missing"] = missing
        status = CapabilityStatus.MISSING_DEPS
    else:
        status = CapabilityStatus.READY
    if surfaces_pending:
        metadata["surfaces_pending"] = surfaces_pending

    capability = Capability(
        kind=CapabilityKind.AGENT,
        name=manifest.name,
        description=manifest.description,
        status=status,
        metadata=metadata,
    )
    registry.register(capability)
    return capability
