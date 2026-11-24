"""Module Catalog — what Lumen can recommend when it detects a gap.

When Lumen can't do something, it doesn't just say "I can't."
It checks the catalog, finds modules that fill the gap, and suggests
installing them. Like WordPress: "You need this plugin. Install it?"

The catalog can come from more than one local source: the package
``index.yaml`` always loads first, then any extra index (config
``catalog.sources`` / ``LUMEN_CATALOG_PATHS``) merges by module name.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Sequence

import yaml

from lumen.core.cerebellum import compatibility_for_catalog_entry

logger = logging.getLogger(__name__)

# Campos de lista que se fusionan como unión deduplicada (orden estable).
_UNION_KEYS = ("fills_gaps", "tags", "use_cases", "provides")


def _union_list(existing, incoming) -> list:
    out = list(existing or [])
    for item in incoming or []:
        if item not in out:
            out.append(item)
    return out


def _merge_entry(existing: dict, incoming: dict) -> None:
    """Fusiona *incoming* sobre *existing* (ver tabla de fusión del design).

    - ``path``: gana la primera fuente que lo definió (un overlay no puede
      redirigir una instalación).
    - listas conocidas y ``requires.*``: unión deduplicada.
    - el resto (escalares, ``x-*``, ...): gana la última fuente.
    """
    for key, value in incoming.items():
        if key == "path":
            if value and not existing.get("path"):
                existing["path"] = value
            continue
        if key == "requires" and isinstance(value, dict):
            current = existing.get("requires")
            if not isinstance(current, dict):
                current = {}
            for req_key, req_val in value.items():
                if isinstance(req_val, list) and isinstance(current.get(req_key), list):
                    current[req_key] = _union_list(current[req_key], req_val)
                else:
                    current[req_key] = req_val
            existing["requires"] = current
            continue
        if key in _UNION_KEYS:
            if value:
                existing[key] = _union_list(existing.get(key), value)
            continue
        existing[key] = value


def _resolve_catalog_path(raw, lumen_dir: Path | None) -> Path:
    path = Path(os.path.expanduser(str(raw)))
    if not path.is_absolute() and lumen_dir is not None:
        path = Path(lumen_dir) / path
    return path


def catalog_paths_from_config(
    config: dict | None,
    *,
    lumen_dir: Path | None = None,
    env: dict | None = None,
) -> list[Path]:
    """Rutas de index extra: ``config.catalog.sources`` + ``LUMEN_CATALOG_PATHS``.

    Las rutas relativas de la config se resuelven contra ``lumen_dir``. El env
    se separa por ``os.pathsep`` o coma y va **al final** de la lista: como las
    fuentes se cargan en orden y la última gana en la fusión, el env pisa la
    config sin editar archivos.
    """
    env = os.environ if env is None else env
    config = config or {}

    catalog_cfg = config.get("catalog") or {}
    raw_sources = catalog_cfg.get("sources") or []
    if isinstance(raw_sources, (str, Path)):
        raw_sources = [raw_sources]

    paths: list[Path] = [
        _resolve_catalog_path(raw, lumen_dir) for raw in raw_sources
    ]

    env_value = str(env.get("LUMEN_CATALOG_PATHS") or "").strip()
    if env_value:
        pattern = "[" + re.escape(os.pathsep) + ",]"
        for part in re.split(pattern, env_value):
            part = part.strip()
            if part:
                paths.append(_resolve_catalog_path(part, lumen_dir))

    return paths


class Catalog:
    """Available modules that can be installed to extend Lumen."""

    def __init__(
        self,
        catalog_path: Path | None = None,
        *,
        extra_paths: Sequence[Path] = (),
    ):
        if catalog_path is None:
            catalog_path = Path(__file__).parent.parent / "catalog" / "index.yaml"
        self._modules: list[dict] = []
        self._capabilities: list[dict] = []
        self._modules_by_name: dict[str, dict] = {}
        self._capabilities_by_name: dict[str, dict] = {}
        # name → directorio del index que definió `path` para ese módulo.
        self._source_roots: dict[str, Path] = {}
        self._sources: list[Path] = []

        # 1) El index del paquete siempre primero.
        self._load_source(Path(catalog_path), required=False)
        # 2) Las fuentes extra después, en el orden dado.
        for raw in extra_paths or ():
            self._load_source(Path(raw), required=True)

    def _load_source(self, path: Path, *, required: bool) -> None:
        if not path.exists():
            if required:
                logger.warning("Catalog source not found, ignoring: %s", path)
            return

        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid catalog YAML in {path}: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid catalog YAML in {path}: root must be a mapping"
            )

        source_root = path.resolve().parent
        for entry in data.get("modules") or []:
            self._merge_module(entry, source_root, path)
        for entry in data.get("capabilities") or []:
            self._merge_capability(entry, source_root, path)
        self._sources.append(path.resolve())

    def _merge_module(self, entry, source_root: Path, source_path: Path) -> None:
        if not isinstance(entry, dict) or not entry.get("name"):
            logger.warning("Ignoring catalog entry without name in %s", source_path)
            return
        name = str(entry["name"])
        existing = self._modules_by_name.get(name)
        if existing is None:
            merged = dict(entry)
            self._modules.append(merged)
            self._modules_by_name[name] = merged
            if merged.get("path"):
                self._source_roots[name] = source_root
            return
        had_path = bool(existing.get("path"))
        incoming_path = bool(entry.get("path"))
        _merge_entry(existing, entry)
        if incoming_path and not had_path:
            self._source_roots[name] = source_root

    def _merge_capability(self, entry, source_root: Path, source_path: Path) -> None:
        if not isinstance(entry, dict) or not entry.get("name"):
            logger.warning("Ignoring catalog capability without name in %s", source_path)
            return
        name = str(entry["name"])
        existing = self._capabilities_by_name.get(name)
        if existing is None:
            merged = dict(entry)
            self._capabilities.append(merged)
            self._capabilities_by_name[name] = merged
            return
        _merge_entry(existing, entry)

    @property
    def modules(self) -> list[dict]:
        return self._modules

    @property
    def sources(self) -> list[Path]:
        """Índices cargados, en orden (paquete primero)."""
        return list(self._sources)

    def source_root(self, name: str) -> Path | None:
        """Directorio del index que originalmente definió ``path`` para *name*.

        Es la raíz contra la que el Installer debe resolver ese módulo. Si
        ninguna fuente definió ``path``, devuelve ``None`` (fallback al
        catálogo del paquete).
        """
        return self._source_roots.get(name)

    def list_capabilities(self) -> list[dict]:
        """Return all capability entries from the catalog."""
        return list(self._capabilities)

    def get_capability(self, name: str) -> dict | None:
        """Get a capability by name from the catalog."""
        for cap in self._capabilities:
            if cap.get("name") == name:
                return dict(cap)
        return None

    def search(
        self, query: str, *, registry=None, connectors=None, model=None
    ) -> list[dict]:
        """Search catalog by name, description, tags, fills_gaps or use_cases."""
        query_lower = query.lower()
        results = []
        for mod in self._modules:
            score = 0
            # Check fills_gaps (highest priority)
            for gap in mod.get("fills_gaps", []):
                if gap in query_lower:
                    score += 10
            # Check use_cases (natural-language phrases, both directions)
            for use_case in mod.get("use_cases", []):
                case = str(use_case).lower()
                if not case:
                    continue
                if case in query_lower or (query_lower and query_lower in case):
                    score += 8
            # Check name and description
            if query_lower in mod.get("name", "").lower():
                score += 5
            if query_lower in mod.get("description", "").lower():
                score += 3
            # Check tags
            for tag in mod.get("tags", []):
                if tag in query_lower or query_lower in tag:
                    score += 2
            if score > 0:
                results.append(
                    (
                        score,
                        self._decorate(
                            mod,
                            registry=registry,
                            connectors=connectors,
                            model=model,
                        ),
                    )
                )
        results.sort(key=lambda x: x[0], reverse=True)
        return [mod for _, mod in results]

    def find_for_gap(
        self, gap_description: str, *, registry=None, connectors=None, model=None
    ) -> list[dict]:
        """Find modules that could fill a specific capability gap.

        This is the key method: Lumen detects it can't do something,
        describes the gap, and this method finds relevant modules.
        """
        return self.search(
            gap_description,
            registry=registry,
            connectors=connectors,
            model=model,
        )

    def get(
        self, name: str, *, registry=None, connectors=None, model=None
    ) -> dict | None:
        """Get a module by name."""
        for mod in self._modules:
            if mod["name"] == name:
                return self._decorate(
                    mod,
                    registry=registry,
                    connectors=connectors,
                    model=model,
                )
        return None

    def list_all(self, *, registry=None, connectors=None, model=None) -> list[dict]:
        """List all available modules."""
        return [
            self._decorate(
                {
                    "name": m["name"],
                    "display_name": m.get("display_name", m["name"]),
                    "description": m.get("description", ""),
                    "version": m.get("version", "0.0.0"),
                    "path": m.get("path"),
                    "price": m.get("price", "free"),
                    "min_capability": m.get("min_capability", "tier-1"),
                    "tags": m.get("tags", []),
                    "provides": m.get("provides", []),
                    "requires": m.get("requires", {}),
                    "fills_gaps": m.get("fills_gaps", []),
                    "use_cases": m.get("use_cases", []),
                },
                registry=registry,
                connectors=connectors,
                model=model,
            )
            for m in self._modules
        ]

    def as_context(
        self,
        installed_names: set[str] | None = None,
        *,
        registry=None,
        connectors=None,
        model=None,
    ) -> str:
        """Format catalog for the LLM prompt.

        Filters out already-installed modules so the LLM doesn't
        recommend installing something Lumen already has.
        """
        installed = installed_names or set()
        available = [
            self._decorate(m, registry=registry, connectors=connectors, model=model)
            for m in self._modules
            if m["name"] not in installed
        ]

        if not available:
            return ""

        lines = [
            "## Module Catalog (what I can recommend to install)",
            "",
            "When I cannot fulfill a request, I should check if a module "
            "in this catalog could help, and suggest installing it.",
            "",
        ]
        for mod in available:
            gaps = ", ".join(mod.get("fills_gaps", [])[:4])
            compat = mod.get("compatibility") or {}
            compat_hint = compat.get("status", "unknown")
            line = (
                f"- **{mod.get('display_name', mod['name'])}** "
                f"({mod['name']}): {mod.get('description', '')} "
                f"[fills: {gaps}; compatibility: {compat_hint}]"
            )
            use_cases = [
                str(u) for u in mod.get("use_cases", [])[:3] if u
            ]
            if use_cases:
                line += f" [casos: {', '.join(use_cases)}]"
            lines.append(line)
        return "\n".join(lines)

    def _decorate(
        self, module: dict, *, registry=None, connectors=None, model=None
    ) -> dict:
        item = dict(module)
        if registry is not None and connectors is not None:
            item["compatibility"] = compatibility_for_catalog_entry(
                item,
                registry,
                connectors,
                model=model,
            )
        return item
