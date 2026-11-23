"""Superficies declarativas: manifiestos, handshake y tool de render.

El motor aporta la mitad "schema"; la shell aporta la mitad "renderer".
Una superficie queda READY solo cuando ambas mitades existen:

- el discovery registra ``SURFACE`` en ``no_renderer`` leyendo ``surface.yaml``;
- un ``hello`` (o ``renderer_added``) de la shell confirma el renderer y
  ``apply_hello`` promueve la superficie a ``READY``;
- ``build_render_surface_tool`` arma el enum de ``render_surface`` en cada
  turno desde el Registry, sin cacheos.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from inspect import isawaitable
from pathlib import Path
from typing import Any, Callable

import yaml
from jsonschema import Draft7Validator

from lumen.core.module_manifest import validate_manifest_version
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry

PROTOCOL_CONTRACT_V3 = "x-lumen-desktop/3"
PROTOCOL_MAJOR = 3

_WHEN_TO_USE_MAX_CHARS = 200
_DESCRIPTION_MAX_CHARS = 4000
_MAX_SURFACES_IN_TOOL = 100


# ---------------------------------------------------------------------------
# Manifiesto surface.yaml
# ---------------------------------------------------------------------------


@dataclass
class SurfaceManifest:
    """Lo que declara un ``surfaces/<name>/surface.yaml``."""

    name: str
    version: str
    contract: str
    description: str
    when_to_use: str
    schema: dict[str, Any] | None
    schema_path: str | None
    renderer: str | None
    interactions: list[str]
    persistent: bool
    manifest_version: int
    schema_error: str | None = None


def _load_schema_file(manifest_dir: Path, schema_ref: str) -> tuple[dict | None, str | None, str | None]:
    """Lee el archivo de schema y lo valida como JSON Schema draft-07.

    Devuelve (schema, ruta_absoluta, error). Si hay error, schema es None.
    """
    schema_path = (manifest_dir / schema_ref).resolve()
    if not schema_path.exists():
        return None, None, f"schema file not found: {schema_ref}"

    try:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return None, None, f"schema is not valid JSON: {exc}"

    if not isinstance(schema, dict):
        return None, None, "schema must be a JSON object"

    try:
        Draft7Validator.check_schema(schema)
    except Exception as exc:  # jsonschema.SchemaError y derivados
        return None, None, f"schema failed draft-07 validation: {exc}"

    return schema, str(schema_path), None


def load_surface_manifest(path: Path) -> SurfaceManifest:
    """Carga y valida un ``surface.yaml``.

    Lanza si el YAML no parsea o si el ``manifest_version`` no está soportado;
    los problemas de schema no lanzan: llegan como ``schema_error`` para que
    el discovery registre la superficie en ``missing_deps``.
    """
    path = Path(path)
    manifest_dir = path.parent

    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}
    if not isinstance(manifest, dict):
        raise ValueError(f"surface manifest must be a mapping: {path}")

    manifest_version = validate_manifest_version(manifest, path=str(path))

    name = str(manifest.get("name") or "").strip()
    if not name:
        name = manifest_dir.name
        import logging

        logging.getLogger(__name__).warning(
            "surface manifest without name; using directory name: %s", name
        )

    schema: dict | None = None
    schema_path: str | None = None
    schema_error: str | None = None
    schema_ref = manifest.get("schema")
    if not schema_ref or not str(schema_ref).strip():
        schema_error = "missing schema reference"
    else:
        schema, schema_path, error = _load_schema_file(manifest_dir, str(schema_ref).strip())
        schema_error = error

    interactions_raw = manifest.get("interactions") or []
    interactions = [str(i) for i in interactions_raw if isinstance(i, str)] if isinstance(
        interactions_raw, list
    ) else []
    renderer = manifest.get("renderer")
    # renderer es declarativo: no se abre ni se ejecuta acá. Si la shell lo
    # tiene, lo confirma vía hello.

    return SurfaceManifest(
        name=name,
        version=str(manifest.get("version") or ""),
        contract=str(manifest.get("contract") or PROTOCOL_CONTRACT_V3),
        description=str(manifest.get("description") or ""),
        when_to_use=str(manifest.get("when_to_use") or ""),
        schema=schema,
        schema_path=schema_path,
        renderer=str(renderer) if renderer else None,
        interactions=interactions,
        persistent=bool(manifest.get("persistent", False)),
        manifest_version=manifest_version,
        schema_error=schema_error,
    )


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def register_surface_manifest(
    registry: Registry,
    manifest: SurfaceManifest,
    *,
    path: Path,
    module_name: str | None = None,
) -> Capability:
    """Registra una superficie en el Registry según lo que se pudo leer."""
    error = manifest.schema_error
    if error:
        status = CapabilityStatus.MISSING_DEPS
    else:
        # Con schema válido pero sin confirmación de la shell: no_renderer.
        status = CapabilityStatus.NO_RENDERER

    metadata: dict[str, Any] = {
        "version": manifest.version,
        "contract": manifest.contract,
        "when_to_use": manifest.when_to_use,
        "schema_path": manifest.schema_path,
        "schema": manifest.schema,
        "renderer": manifest.renderer,
        "interactions": list(manifest.interactions),
        "persistent": manifest.persistent,
        "path": str(path),
        "module_name": module_name,
    }
    if error:
        metadata["error"] = error

    capability = Capability(
        kind=CapabilityKind.SURFACE,
        name=manifest.name,
        description=manifest.description,
        status=status,
        metadata=metadata,
    )
    registry.register(capability)
    return capability


# ---------------------------------------------------------------------------
# Handshake: hello / renderer_added / desconexión
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RendererConfirmation:
    """Un renderer confirmado por una conexión de shell."""

    name: str
    version: str
    host_kind: str
    connection_id: str


def _confirmations_store(registry: Registry) -> dict[str, dict[str, RendererConfirmation]]:
    """Confirmaciones por connection_id → {nombre superficie: confirmación}.

    Vive en memoria, colgado del Registry: tras reiniciar, todas las
    superficies vuelven a ``no_renderer`` hasta el próximo ``hello``.
    """
    store = getattr(registry, "_surface_renderers", None)
    if store is None:
        store = {}
        registry._surface_renderers = store  # noqa: SLF001 — registro interno del Registry
    return store


def _contract_major_3(contract: Any) -> bool:
    match = re.search(r"/(\d+)", str(contract or ""))
    return bool(match) and int(match.group(1)) == PROTOCOL_MAJOR


def _surface_has_valid_schema(cap: Capability) -> bool:
    """El motor puede mostrar la superficie solo si tiene schema y sin errores."""
    return isinstance(cap.metadata.get("schema"), dict) and not cap.metadata.get("error")


def _collect_registry_events(registry: Registry):
    """Contexto helper: recolecta los eventos que el Registry emite."""
    collected: list = []

    def _on_event(event):
        collected.append(event)

    registry.subscribe(_on_event)
    return collected, _on_event


def _promote_to_ready(registry: Registry, cap: Capability, confirmation: RendererConfirmation) -> None:
    metadata = dict(cap.metadata)
    metadata["renderer_confirmed"] = True
    if confirmation.version:
        metadata["renderer_version"] = confirmation.version
    if confirmation.host_kind:
        metadata["host_kind"] = confirmation.host_kind
    metadata["confirmed_at"] = time.time()
    cap.metadata = metadata
    registry.update_status(CapabilityKind.SURFACE, cap.name, CapabilityStatus.READY)


def _demote_to_no_renderer(registry: Registry, cap: Capability) -> None:
    metadata = dict(cap.metadata)
    metadata["renderer_confirmed"] = False
    metadata.pop("renderer_version", None)
    metadata.pop("host_kind", None)
    metadata.pop("confirmed_at", None)
    cap.metadata = metadata
    registry.update_status(CapabilityKind.SURFACE, cap.name, CapabilityStatus.NO_RENDERER)


def _recompute_surface_status(registry: Registry, cap: Capability) -> None:
    """READY si alguna conexión sigue confirmando el renderer; si no, no_renderer."""
    store = _confirmations_store(registry)
    still_confirmed = any(cap.name in conns for conns in store.values())
    if still_confirmed:
        # Sigue con sus metadatos de renderer: otra conexión lo respalda.
        return
    if cap.status == CapabilityStatus.READY:
        _demote_to_no_renderer(registry, cap)


def apply_hello(
    registry: Registry,
    *,
    connection_id: str,
    contract: str,
    renderers: list[dict],
    host: dict | None = None,
) -> list:
    """Confirma los renderers de una conexión y recalcula el estado.

    El hello es un snapshot completo de ESA conexión: su lista reemplaza las
    confirmaciones previas de ``connection_id``. Devuelve los eventos que el
    Registry emitió (nada se inventa afuera).
    """
    if not _contract_major_3(contract):
        import logging

        logging.getLogger(__name__).warning(
            "hello ignored: unsupported contract %s (major must be %d)", contract, PROTOCOL_MAJOR
        )
        return []

    store = _confirmations_store(registry)
    previous = set(store.get(connection_id, {}))

    collected, _on_event = _collect_registry_events(registry)
    try:
        confirmed_now: set[str] = set()
        connection_renderers = store.setdefault(connection_id, {})
        for renderer in renderers or []:
            if not isinstance(renderer, dict):
                continue
            name = str(renderer.get("name") or "").strip()
            if not name:
                continue
            cap = registry.get(CapabilityKind.SURFACE, name)
            if cap is None:
                # El hello nunca crea superficies: solo confirma las existentes.
                continue
            confirmation = RendererConfirmation(
                name=name,
                version=str(renderer.get("version") or ""),
                host_kind=str((host or {}).get("kind") or ""),
                connection_id=str(connection_id),
            )
            connection_renderers[name] = confirmation
            confirmed_now.add(name)
            if _surface_has_valid_schema(cap):
                _promote_to_ready(registry, cap, confirmation)

        # Lo que esta conexión confirmaba y ya no declare: fuera del snapshot.
        for name in sorted(previous - confirmed_now):
            connection_renderers.pop(name, None)
            cap = registry.get(CapabilityKind.SURFACE, name)
            if cap is not None:
                _recompute_surface_status(registry, cap)
    finally:
        registry.unsubscribe(_on_event)

    return collected


def apply_renderer_added(
    registry: Registry,
    *,
    connection_id: str,
    renderer: dict,
) -> Any:
    """Confirma un solo renderer en caliente sin tocar los demás.

    A diferencia del hello (snapshot), esto suma a lo que la conexión ya
    confirmaba: el hot-load de la shell no debe bajar superficies vivas.
    """
    if not isinstance(renderer, dict):
        return None
    name = str(renderer.get("name") or "").strip()
    if not name:
        return None
    cap = registry.get(CapabilityKind.SURFACE, name)
    if cap is None:
        return None

    store = _confirmations_store(registry)
    connection_renderers = store.setdefault(str(connection_id), {})

    collected, _on_event = _collect_registry_events(registry)
    try:
        confirmation = RendererConfirmation(
            name=name,
            version=str(renderer.get("version") or ""),
            host_kind=str(connection_renderers.get(name).host_kind) if connection_renderers.get(name) else "",
            connection_id=str(connection_id),
        )
        connection_renderers[name] = confirmation
        if _surface_has_valid_schema(cap):
            _promote_to_ready(registry, cap, confirmation)
    finally:
        registry.unsubscribe(_on_event)

    return collected[0] if collected else None


def drop_connection(registry: Registry, *, connection_id: str) -> list:
    """Recalcula las superficies que solo ``connection_id`` confirmaba."""
    store = _confirmations_store(registry)
    confirmed = store.pop(str(connection_id), None) or {}

    collected, _on_event = _collect_registry_events(registry)
    try:
        for name in sorted(confirmed):
            cap = registry.get(CapabilityKind.SURFACE, name)
            if cap is not None:
                _recompute_surface_status(registry, cap)
    finally:
        registry.unsubscribe(_on_event)

    return collected


# ---------------------------------------------------------------------------
# Tool render_surface
# ---------------------------------------------------------------------------


def build_render_surface_tool(registry: Registry) -> dict | None:
    """Arma el function-schema de ``render_surface`` desde el Registry.

    Solo entran superficies READY, ordenadas por nombre. Sin candidatas
    devuelve None y el Brain no agrega la tool. Se reconstruye en cada
    turno: nada queda cacheado entre llamadas.
    """
    ready = [
        cap
        for cap in registry.list_by_kind(CapabilityKind.SURFACE)
        if cap.is_ready()
    ]
    ready.sort(key=lambda cap: cap.name)

    if not ready:
        return None

    shown = ready[:_MAX_SURFACES_IN_TOOL]
    omitted = len(ready) - len(shown)

    caps_shown: list[Capability] = list(shown)
    lines: list[str] = []
    for cap in caps_shown:
        hint = str(cap.metadata.get("when_to_use") or cap.description or "")
        hint = " ".join(hint.split())[:_WHEN_TO_USE_MAX_CHARS]
        lines.append(f"- {cap.name}: {hint}")

    header = "Dibuja una superficie en la interfaz. Superficies disponibles:"
    while True:
        description = header + "\n" + "\n".join(lines)
        if omitted:
            description += f"\n(+{omitted} superficies más)"
        if len(description) <= _DESCRIPTION_MAX_CHARS or not lines:
            break
        # No entra todo: una línea menos y el aviso queda siempre visible.
        lines.pop()
        caps_shown.pop()
        omitted += 1

    if len(description) > _DESCRIPTION_MAX_CHARS:
        description = description[: _DESCRIPTION_MAX_CHARS - 3].rstrip() + "..."

    enum = [cap.name for cap in caps_shown]

    return {
        "type": "function",
        "function": {
            "name": "render_surface",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "surface": {"type": "string", "enum": enum},
                    "data": {"type": "object"},
                    "id": {"type": "string"},
                },
                "required": ["surface", "data"],
            },
        },
    }


async def execute_render_surface(
    registry: Registry,
    *,
    surface: str,
    data: dict[str, Any],
    render: Callable[[str, dict, str | None], Any],
    id: str | None = None,
) -> dict[str, Any]:
    """Ejecuta ``render_surface`` validando el estado en el momento.

    Si la superficie dejó de estar READY (o nunca existió) devuelve un error
    estructurado y no emite ningún render. La validación de ``data`` y el ack
    son del Surface Service (F4): acá solo se garantiza que la tool nunca
    ofrece una superficie que la interfaz no puede dibujar.
    """
    cap = registry.get(CapabilityKind.SURFACE, surface)
    if cap is None:
        return {"ok": False, "error": "surface_unknown", "surface": surface}
    if not cap.is_ready():
        return {
            "ok": False,
            "error": "surface_not_ready",
            "surface": surface,
            "status": cap.status.value,
        }
    result = render(surface, data, id)
    if isawaitable(result):
        result = await result
    return result
