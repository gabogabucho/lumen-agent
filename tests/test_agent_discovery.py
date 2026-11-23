"""Tests del Bloque 5: discovery de especialistas (kits de agent).

Cubre tasks.md 5.1 a 5.6:
- AgentManifest + load_agent_manifest con personalidad validada (5.1)
- roots globales y dentro de módulos (5.2)
- skills y tools resueltas contra el Registry / runtime (5.3)
- superficies propias no listas no bloquean, quedan anotadas (5.4)
- metadata del especialista READY (5.5)
- personalidad ausente → ERROR (5.6)
"""

from __future__ import annotations

from pathlib import Path

import yaml

from lumen.core import discovery
from lumen.core.cerebellum import COMPAT_READY
from lumen.core.connectors import Connector, ConnectorRegistry
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry
from lumen.core.agents import AgentManifest, load_agent_manifest


def _write_agent(
    root: Path,
    name: str,
    *,
    with_personality: bool = True,
    personality_body: str = "Voz de la marca: directa, cálida.\n",
    skills: list[str] | None = None,
    tools: list[str] | None = None,
    surfaces: list[str] | None = None,
    min_capability: str | None = None,
    model: str | None = None,
    max_context_tokens: int | None = None,
    manifest_version=1,
) -> Path:
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    if with_personality:
        (base / "personality.md").write_text(personality_body, encoding="utf-8")

    manifesto: dict = {
        "name": name,
        "display_name": name.title(),
        "description": f"Especialista {name} de prueba.",
        "personality": "./personality.md",
    }
    if manifest_version is not None:
        manifesto["manifest_version"] = manifest_version
    if skills is not None:
        manifesto["skills"] = skills
    if tools is not None:
        manifesto["tools"] = tools
    if surfaces is not None:
        manifesto["surfaces"] = surfaces
    if min_capability is not None:
        manifesto["min_capability"] = min_capability
    if model is not None:
        manifesto["model"] = model
    if max_context_tokens is not None:
        manifesto["max_context_tokens"] = max_context_tokens
    (base / "agent.yaml").write_text(yaml.safe_dump(manifesto), encoding="utf-8")
    return base


def _register_ready_skill(registry: Registry, name: str) -> None:
    registry.register(
        Capability(
            kind=CapabilityKind.SKILL,
            name=name,
            description=f"skill {name}",
            status=CapabilityStatus.READY,
        )
    )


def _connectors_with_ready_search() -> ConnectorRegistry:
    reg = ConnectorRegistry()
    conn = Connector("web", "Web connector", ["search"])
    conn.register_handler("search", lambda **kwargs: None)
    reg.register(conn)
    return reg


# --- 5.1: manifiesto ---


def test_load_agent_manifest(tmp_path):
    base = _write_agent(
        tmp_path,
        "contenido",
        skills=["brand-voice", "copy-instagram"],
        tools=["web__search"],
        surfaces=["post_preview"],
        min_capability="tier-2",
        model="openai/gpt-4.1-mini",
        max_context_tokens=24000,
    )

    manifest = load_agent_manifest(base / "agent.yaml")

    assert isinstance(manifest, AgentManifest)
    assert manifest.name == "contenido"
    assert manifest.display_name == "Contenido"
    assert manifest.description == "Especialista contenido de prueba."
    assert manifest.personality_path == str((base / "personality.md").resolve())
    assert manifest.personality_error is None
    assert manifest.skills == ["brand-voice", "copy-instagram"]
    assert manifest.tools == ["web__search"]
    assert manifest.surfaces == ["post_preview"]
    assert manifest.min_capability == "tier-2"
    assert manifest.model == "openai/gpt-4.1-mini"
    assert manifest.max_context_tokens == 24000
    assert manifest.manifest_version == 1


def test_load_agent_manifest_empty_personality(tmp_path):
    base = _write_agent(tmp_path, "vacío", personality_body="")
    manifest = load_agent_manifest(base / "agent.yaml")
    assert manifest.personality_error is not None


def test_load_agent_manifest_personality_with_only_whitespace(tmp_path):
    base = _write_agent(tmp_path, "blanco", personality_body="   \n\n  ")
    manifest = load_agent_manifest(base / "agent.yaml")
    assert manifest.personality_error is not None


def test_compat_ready_constant_is_what_match_declared_tools_returns():
    # Ancordamos el valor que usa la resolución de tools para "lista".
    assert COMPAT_READY == "ready"


# --- 5.2: roots globales y de módulos ---


def test_discover_agents_from_global_and_module_roots(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    module_dir = pkg_dir / "modules" / "kit-crecimiento"

    _write_agent(lumen_dir / "agents", "contenido")
    _write_agent(module_dir / "agents", "crecimiento")

    registry = Registry()
    discovery._discover_agents(
        registry,
        [lumen_dir / "agents", module_dir / "agents"],
        module_name_by_root={str(module_dir / "agents"): "kit-crecimiento"},
    )

    contenido = registry.get(CapabilityKind.AGENT, "contenido")
    crecimiento = registry.get(CapabilityKind.AGENT, "crecimiento")
    assert contenido is not None
    assert crecimiento is not None
    assert contenido.metadata["module_name"] is None
    assert crecimiento.metadata["module_name"] == "kit-crecimiento"


def test_discover_agents_same_name_first_root_wins(tmp_path):
    lumen_dir = tmp_path / "lumen"
    pkg_dir = tmp_path / "pkg"
    _write_agent(lumen_dir / "agents", "contenido", model="uno")
    _write_agent(pkg_dir / "agents", "contenido", model="dos")

    registry = Registry()
    discovery._discover_agents(
        registry,
        [lumen_dir / "agents", pkg_dir / "agents"],
        connectors=_connectors_with_ready_search(),
    )

    assert registry.get(CapabilityKind.AGENT, "contenido").metadata["model"] == "uno"


def test_discover_agents_missing_root_is_silent(tmp_path):
    registry = Registry()
    discovery._discover_agents(registry, [tmp_path / "no_existe"])
    assert registry.all() == []


# --- 5.3: dependencias ---


def test_agent_missing_deps_missing_skill(tmp_path):
    root = tmp_path / "agents"
    _write_agent(
        root,
        "contenido",
        skills=["brand-voice", "copy-instagram"],
        tools=["web__search"],
        surfaces=["post_preview"],
    )
    registry = Registry()
    _register_ready_skill(registry, "brand-voice")
    # copy-instagram no existe en el Registry.

    discovery._discover_agents(
        registry, [root], connectors=_connectors_with_ready_search()
    )

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.MISSING_DEPS
    assert "copy-instagram" in cap.metadata["missing"]


def test_agent_missing_deps_blocked_tool(tmp_path):
    root = tmp_path / "agents"
    _write_agent(root, "contenido", skills=[], tools=["ghost__action"])

    registry = Registry()
    discovery._discover_agents(registry, [root], connectors=ConnectorRegistry())

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.MISSING_DEPS
    assert "ghost__action" in cap.metadata["missing"]


def test_agent_ready_with_resolved_deps(tmp_path):
    root = tmp_path / "agents"
    _write_agent(
        root,
        "contenido",
        skills=["brand-voice"],
        tools=["web__search"],
        surfaces=[],
    )
    registry = Registry()
    _register_ready_skill(registry, "brand-voice")

    discovery._discover_agents(
        registry, [root], connectors=_connectors_with_ready_search()
    )

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.READY
    assert cap in registry.ready()


def test_agent_missing_from_no_connectors_context_blocks_tools(tmp_path):
    # Sin connectors en el entorno, ninguna tool declarada se puede resolver.
    root = tmp_path / "agents"
    _write_agent(root, "contenido", skills=[], tools=["web__search"])

    registry = Registry()
    discovery._discover_agents(registry, [root])

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.MISSING_DEPS
    assert "web__search" in cap.metadata["missing"]


# --- 5.4: superficies pendientes ---


def test_agent_pending_surfaces(tmp_path):
    root = tmp_path / "agents"
    _write_agent(root, "contenido", skills=[], tools=[], surfaces=["post_preview"])

    registry = Registry()
    registry.register(
        Capability(
            kind=CapabilityKind.SURFACE,
            name="post_preview",
            description="vista previa",
            status=CapabilityStatus.NO_RENDERER,
            metadata={"schema": {"type": "object"}},
        )
    )

    discovery._discover_agents(registry, [root], connectors=ConnectorRegistry())

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    # La superficie existe pero sin renderer: el especialista puede quedar
    # READY (produce y cae a artifact), con la pendiente anotada.
    assert cap.status == CapabilityStatus.READY
    assert "post_preview" in cap.metadata["surfaces_pending"]


def test_agent_missing_surface_is_missing_dep(tmp_path):
    root = tmp_path / "agents"
    _write_agent(root, "contenido", skills=[], tools=[], surfaces=["no_existe"])

    registry = Registry()
    discovery._discover_agents(registry, [root])

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.MISSING_DEPS
    assert "no_existe" in cap.metadata["missing"]


# --- 5.5: metadata del READY ---


def test_agent_ready_metadata(tmp_path):
    root = tmp_path / "agents"
    base = _write_agent(
        root,
        "contenido",
        skills=["brand-voice"],
        tools=["web__search"],
        surfaces=["post_preview"],
        min_capability="tier-2",
        model="openai/gpt-4.1-mini",
        max_context_tokens=24000,
    )
    registry = Registry()
    _register_ready_skill(registry, "brand-voice")
    registry.register(
        Capability(
            kind=CapabilityKind.SURFACE,
            name="post_preview",
            description="vista previa",
            status=CapabilityStatus.READY,
            metadata={"schema": {"type": "object"}},
        )
    )

    discovery._discover_agents(
        registry, [root], connectors=_connectors_with_ready_search()
    )

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    meta = cap.metadata
    assert meta["personality_path"] == str((base / "personality.md").resolve())
    assert meta["skills"] == ["brand-voice"]
    assert meta["tools"] == ["web__search"]
    assert meta["surfaces"] == ["post_preview"]
    assert meta["min_capability"] == "tier-2"
    assert meta["model"] == "openai/gpt-4.1-mini"
    assert meta["max_context_tokens"] == 24000


# --- 5.6: personalidad ausente ---


def test_agent_missing_personality_is_error(tmp_path):
    root = tmp_path / "agents"
    _write_agent(root, "contenido", with_personality=False)

    registry = Registry()
    discovery._discover_agents(registry, [root])

    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap.status == CapabilityStatus.ERROR
    assert "personality" in cap.metadata["error"].lower()


def test_agent_broken_manifest_is_isolated(tmp_path):
    root = tmp_path / "agents"
    roto = root / "roto"
    roto.mkdir(parents=True)
    (roto / "agent.yaml").write_text("{ yaml: [roto", encoding="utf-8")
    _write_agent(root, "bueno", skills=[], tools=[])

    registry = Registry()
    discovery._discover_agents(registry, [root])

    assert registry.get(CapabilityKind.AGENT, "roto").status == CapabilityStatus.ERROR
    assert registry.get(CapabilityKind.AGENT, "bueno").status == CapabilityStatus.READY


# --- contrato de versión ---


def test_agent_with_unknown_version_registers_error(tmp_path):
    root = tmp_path / "agents"
    _write_agent(root, "futuro", manifest_version=5)

    registry = Registry()
    discovery._discover_agents(registry, [root])

    cap = registry.get(CapabilityKind.AGENT, "futuro")
    assert cap.status == CapabilityStatus.ERROR
    assert cap.metadata["error"] == "unsupported manifest_version 5; supported: 1"


# --- enganche en discover_all ---


def test_discover_all_registers_agents_after_skills(tmp_path):
    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    lumen_dir = tmp_path / "lumen"
    _write_agent(lumen_dir / "agents", "contenido", skills=[], tools=[])

    registry = Registry()
    result = discovery.discover_all(registry, pkg_dir, ConnectorRegistry(), lumen_dir=lumen_dir)

    assert result is registry
    cap = registry.get(CapabilityKind.AGENT, "contenido")
    assert cap is not None
    assert cap.status == CapabilityStatus.READY
