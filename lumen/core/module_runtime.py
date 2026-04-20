"""Runtime hooks for installed x-lumen modules."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from lumen.core.connectors import ConnectorRegistry
from lumen.core.memory import Memory
from lumen.core.module_manifest import load_module_manifest


@dataclass
class ModuleRuntimeContext:
    name: str
    module_dir: Path
    runtime_dir: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    connectors: ConnectorRegistry | None = None
    memory: Memory | None = None
    lumen_dir: Path | None = None
    brain: Any = None
    registered_tools: list[str] = field(default_factory=list)

    def ensure_runtime_dir(self) -> Path:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        return self.runtime_dir

    def read_runtime_state(self) -> dict[str, Any]:
        state_path = self.runtime_dir / "runtime.json"
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def write_runtime_state(self, payload: dict[str, Any]) -> None:
        self.ensure_runtime_dir()
        state_path = self.runtime_dir / "runtime.json"
        state_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    def resolve_setting(self, key: str, env_name: str | None = None) -> str | None:
        module_settings = ((self.config or {}).get("modules") or {}).get(self.name, {})
        if key in module_settings and module_settings[key] not in {None, ""}:
            return str(module_settings[key])
        module_secrets = ((self.config or {}).get("secrets") or {}).get(self.name, {})
        if env_name and env_name in module_secrets and module_secrets[env_name] not in {None, ""}:
            return str(module_secrets[env_name])
        if key in module_secrets and module_secrets[key] not in {None, ""}:
            return str(module_secrets[key])
        if env_name:
            env_value = os.environ.get(env_name)
            if env_value:
                return env_value
        return None

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.connectors is None:
            raise RuntimeError("Connectors registry not available")
        self.connectors.register_tool(name, description, parameters, handler, metadata)
        if name not in self.registered_tools:
            self.registered_tools.append(name)

    def unregister_registered_tools(self) -> None:
        if self.connectors is None:
            return
        for tool_name in list(self.registered_tools):
            self.connectors.unregister_tool(tool_name)
        self.registered_tools.clear()


def _load_runtime_module(module_dir: Path, name: str) -> ModuleType | None:
    connector_path = module_dir / "connector.py"
    if not connector_path.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        f"lumen_module_{name.replace('-', '_')}", connector_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_context(
    *,
    name: str,
    module_dir: Path,
    runtime_root: Path,
    config: dict[str, Any] | None,
    connectors: ConnectorRegistry | None = None,
    memory: Memory | None = None,
    lumen_dir: Path | None = None,
    brain: Any = None,
) -> ModuleRuntimeContext:
    _, manifest = load_module_manifest(module_dir)
    return ModuleRuntimeContext(
        name=name,
        module_dir=module_dir,
        runtime_dir=runtime_root / name,
        manifest=manifest,
        config=config or {},
        connectors=connectors,
        memory=memory,
        lumen_dir=lumen_dir,
        brain=brain,
    )


def run_module_install_hook(
    *,
    name: str,
    module_dir: Path,
    runtime_root: Path,
    config: dict[str, Any] | None = None,
    lumen_dir: Path | None = None,
) -> None:
    module = _load_runtime_module(module_dir, name)
    if module is None or not hasattr(module, "install"):
        return

    context = _build_context(
        name=name,
        module_dir=module_dir,
        runtime_root=runtime_root,
        config=config,
        lumen_dir=lumen_dir,
    )
    context.ensure_runtime_dir()
    module.install(context)


def run_module_uninstall_hook(
    *,
    name: str,
    module_dir: Path,
    runtime_root: Path,
    config: dict[str, Any] | None = None,
    lumen_dir: Path | None = None,
) -> None:
    module = _load_runtime_module(module_dir, name)
    context = _build_context(
        name=name,
        module_dir=module_dir,
        runtime_root=runtime_root,
        config=config,
        lumen_dir=lumen_dir,
    )
    if module is not None and hasattr(module, "uninstall"):
        module.uninstall(context)
    shutil.rmtree(context.runtime_dir, ignore_errors=True)


@dataclass
class LoadedModuleRuntime:
    module: ModuleType
    context: ModuleRuntimeContext
    state: Any = None


class ModuleRuntimeManager:
    def __init__(
        self,
        *,
        pkg_dir: Path,
        lumen_dir: Path,
        config: dict[str, Any],
        connectors: ConnectorRegistry,
        memory: Memory,
        brain: Any = None,
    ):
        self.pkg_dir = pkg_dir
        self.lumen_dir = lumen_dir
        self.runtime_root = lumen_dir / "modules"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.connectors = connectors
        self.memory = memory
        self.brain = brain
        self._loaded: dict[str, LoadedModuleRuntime] = {}

    async def sync(self) -> None:
        modules_dir = self.pkg_dir / "modules"
        installed_names: set[str] = set()

        if modules_dir.exists():
            for module_dir in modules_dir.iterdir():
                if not module_dir.is_dir() or module_dir.name.startswith("_"):
                    continue
                manifest_path, manifest = load_module_manifest(module_dir)
                if manifest_path is None:
                    continue
                name = str(manifest.get("name") or module_dir.name)
                installed_names.add(name)
                if name not in self._loaded:
                    await self._activate(name, module_dir)

        for name in list(self._loaded):
            if name not in installed_names:
                await self.unload(name)

    async def unload(self, name: str) -> None:
        loaded = self._loaded.pop(name, None)
        if loaded is None:
            return

        module = loaded.module
        try:
            if hasattr(module, "deactivate"):
                result = module.deactivate(loaded.context, loaded.state)
                if inspect.isawaitable(result):
                    await result
        finally:
            loaded.context.unregister_registered_tools()

    async def close(self) -> None:
        for name in list(self._loaded):
            await self.unload(name)

    async def _activate(self, name: str, module_dir: Path) -> None:
        module = _load_runtime_module(module_dir, name)
        if module is None or not hasattr(module, "activate"):
            return

        context = _build_context(
            name=name,
            module_dir=module_dir,
            runtime_root=self.runtime_root,
            config=self.config,
            connectors=self.connectors,
            memory=self.memory,
            lumen_dir=self.lumen_dir,
            brain=self.brain,
        )
        context.ensure_runtime_dir()

        try:
            result = module.activate(context)
            if inspect.isawaitable(result):
                result = await result
        except Exception:
            context.unregister_registered_tools()
            return

        self._loaded[name] = LoadedModuleRuntime(
            module=module, context=context, state=result
        )
