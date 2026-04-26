"""Tests for shared capabilities — manifest parsing, runtime injection,
terminal PYTHONPATH, and isolation.
"""

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.discovery import _discover_capabilities, discover_all
from lumen.core.handlers import _build_terminal_env, terminal_execute
from lumen.core.installer import Installer
from lumen.core.module_manifest import (
    _validate_capability_name,
    parse_capabilities,
    resolve_capability_paths,
)
from lumen.core.module_runtime import (
    CapabilityPathInjector,
    ModuleRuntimeManager,
    _load_runtime_module,
)
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry


class CapabilityNameValidationTests(unittest.TestCase):
    def test_valid_names(self):
        self.assertTrue(_validate_capability_name("shared-logger"))
        self.assertTrue(_validate_capability_name("http_client"))
        self.assertTrue(_validate_capability_name("mylib2"))

    def test_invalid_names(self):
        self.assertFalse(_validate_capability_name(""))
        self.assertFalse(_validate_capability_name("../escape"))
        self.assertFalse(_validate_capability_name("with/slash"))
        self.assertFalse(_validate_capability_name("with\\backslash"))
        self.assertFalse(_validate_capability_name("name\x00null"))
        self.assertFalse(_validate_capability_name(".."))
        self.assertFalse(_validate_capability_name("CON"))
        self.assertFalse(_validate_capability_name("NUL"))


class ManifestCapabilityParsingTests(unittest.TestCase):
    def test_parse_capabilities_from_manifest(self):
        manifest = {"name": "test", "capabilities": ["shared-logger", "http-client"]}
        self.assertEqual(parse_capabilities(manifest), ["shared-logger", "http-client"])

    def test_parse_capabilities_missing_field(self):
        self.assertEqual(parse_capabilities({}), [])

    def test_parse_capabilities_invalid_items(self):
        manifest = {"capabilities": ["valid", "", 123, None]}
        self.assertEqual(parse_capabilities(manifest), ["valid"])

    def test_parse_capabilities_not_a_list(self):
        self.assertEqual(parse_capabilities({"capabilities": "string"}), [])


class CapabilityPathResolutionTests(unittest.TestCase):
    def test_resolve_existing_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            lumen_dir = Path(tmp)
            cap_dir = lumen_dir / "capabilities" / "test-cap"
            cap_dir.mkdir(parents=True)
            (cap_dir / "__init__.py").write_text("", encoding="utf-8")

            paths = resolve_capability_paths(["test-cap"], lumen_dir=lumen_dir)
            self.assertEqual(len(paths), 1)
            self.assertEqual(paths[0].name, "test-cap")

    def test_resolve_missing_capability_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            lumen_dir = Path(tmp)
            paths = resolve_capability_paths(["missing"], lumen_dir=lumen_dir)
            self.assertEqual(paths, [])

    def test_resolve_invalid_name_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            lumen_dir = Path(tmp)
            paths = resolve_capability_paths(["../bad"], lumen_dir=lumen_dir)
            self.assertEqual(paths, [])


class CapabilityPathInjectorTests(unittest.TestCase):
    def test_injects_parent_paths_temporarily(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap_dir = Path(tmp) / "mylib"
            cap_dir.mkdir()

            original_len = len(sys.path)
            injector = CapabilityPathInjector([cap_dir])
            with injector:
                # The parent directory is injected so "mylib" can be imported
                self.assertIn(str(cap_dir.parent), sys.path)
            self.assertNotIn(str(cap_dir.parent), sys.path)
            self.assertEqual(len(sys.path), original_len)

    def test_restores_sys_path_after_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            cap_dir = Path(tmp) / "mylib"
            cap_dir.mkdir()

            original = sys.path[:]
            injector = CapabilityPathInjector([cap_dir])
            try:
                with injector:
                    self.assertIn(str(cap_dir.parent), sys.path)
                    raise RuntimeError("test")
            except RuntimeError:
                pass
            self.assertEqual(sys.path, original)


class RuntimeModuleLoadTests(unittest.TestCase):
    def test_module_can_import_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lumen_dir = tmp / "lumen"
            cap_dir = lumen_dir / "capabilities" / "shared_lib"
            cap_dir.mkdir(parents=True)
            (cap_dir / "__init__.py").write_text("value = 42\n", encoding="utf-8")

            module_dir = tmp / "modules" / "test-mod"
            module_dir.mkdir(parents=True)
            (module_dir / "module.yaml").write_text(
                "name: test-mod\ncapabilities:\n  - shared_lib\n",
                encoding="utf-8",
            )
            (module_dir / "connector.py").write_text(
                "import shared_lib\nresult = shared_lib.value\n",
                encoding="utf-8",
            )

            from lumen.core.module_manifest import load_module_manifest

            _, manifest = load_module_manifest(module_dir)
            cap_names = parse_capabilities(manifest)
            caps = resolve_capability_paths(cap_names, lumen_dir=lumen_dir)

            mod = _load_runtime_module(module_dir, "test-mod", capability_paths=caps)
            self.assertIsNotNone(mod)
            self.assertEqual(mod.result, 42)
            self.assertNotIn(str(cap_dir), sys.path)

    def test_module_without_capabilities_works_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            module_dir = Path(tmp) / "modules" / "plain-mod"
            module_dir.mkdir(parents=True)
            (module_dir / "module.yaml").write_text(
                "name: plain-mod\n", encoding="utf-8"
            )
            (module_dir / "connector.py").write_text(
                "value = 1\n", encoding="utf-8"
            )

            original_path = sys.path[:]
            mod = _load_runtime_module(module_dir, "plain-mod")
            self.assertIsNotNone(mod)
            self.assertEqual(mod.value, 1)
            self.assertEqual(sys.path, original_path)


class InstallerCapabilityTests(unittest.TestCase):
    def test_install_copies_bundled_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            source = pkg_dir / "source-mod"
            source.mkdir(parents=True)
            (source / "module.yaml").write_text(
                "name: my-mod\ncapabilities:\n  - mylib\n",
                encoding="utf-8",
            )
            cap_src = source / "capabilities" / "mylib"
            cap_src.mkdir(parents=True)
            (cap_src / "__init__.py").write_text("x = 1\n", encoding="utf-8")

            installer = Installer(
                pkg_dir=pkg_dir,
                connectors=ConnectorRegistry(),
                memory=None,
                lumen_dir=pkg_dir / "lumen",
            )
            result = installer.install_from_local_path(source)
            self.assertEqual(result["status"], "installed")

            cap_dest = pkg_dir / "lumen" / "capabilities" / "mylib"
            self.assertTrue(cap_dest.exists())
            self.assertTrue((cap_dest / "__init__.py").exists())

    def test_missing_capability_warns_but_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            source = pkg_dir / "source-mod"
            source.mkdir(parents=True)
            (source / "module.yaml").write_text(
                "name: my-mod\ncapabilities:\n  - missing-lib\n",
                encoding="utf-8",
            )

            installer = Installer(
                pkg_dir=pkg_dir,
                connectors=ConnectorRegistry(),
                memory=None,
                lumen_dir=pkg_dir / "lumen",
            )
            result = installer.install_from_local_path(source)
            self.assertEqual(result["status"], "installed")

    def test_zip_install_copies_capabilities(self):
        import zipfile
        from io import BytesIO

        import yaml

        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            installer = Installer(
                pkg_dir=pkg_dir,
                connectors=ConnectorRegistry(),
                memory=None,
                lumen_dir=pkg_dir / "lumen",
            )

            buf = BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr(
                    "my-mod/module.yaml",
                    yaml.dump({"name": "my-mod", "capabilities": ["lib-a"]}),
                )
                zf.writestr("my-mod/SKILL.md", "# my-mod\n")
                zf.writestr("my-mod/capabilities/lib-a/__init__.py", "y = 2\n")

            result = installer.install_from_zip(buf.getvalue())
            self.assertEqual(result["status"], "installed")

            cap_dest = pkg_dir / "lumen" / "capabilities" / "lib-a"
            self.assertTrue(cap_dest.exists())

    def test_installed_capabilities_are_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            source = pkg_dir / "source-mod"
            source.mkdir(parents=True)
            (source / "module.yaml").write_text(
                "name: my-mod\ncapabilities:\n  - mylib\n",
                encoding="utf-8",
            )
            cap_src = source / "capabilities" / "mylib"
            cap_src.mkdir(parents=True)
            (cap_src / "__init__.py").write_text("x = 1\n", encoding="utf-8")

            installer = Installer(
                pkg_dir=pkg_dir,
                connectors=ConnectorRegistry(),
                memory=None,
                lumen_dir=pkg_dir / "lumen",
            )
            result = installer.install_from_local_path(source)
            self.assertEqual(result["status"], "installed")

            cap_dest = pkg_dir / "lumen" / "capabilities" / "mylib"
            self.assertTrue(cap_dest.exists())

            # Directory should not be user-writable
            self.assertEqual(
                cap_dest.stat().st_mode & stat.S_IWUSR,
                0,
                "Capability directory should be read-only",
            )

            # File should not be user-writable
            init_file = cap_dest / "__init__.py"
            self.assertEqual(
                init_file.stat().st_mode & stat.S_IWUSR,
                0,
                "Capability file should be read-only",
            )

    def test_capability_reinstall_overwrites_and_restores_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg_dir = Path(tmp)
            source = pkg_dir / "source-mod"
            source.mkdir(parents=True)
            (source / "module.yaml").write_text(
                "name: my-mod\ncapabilities:\n  - mylib\n",
                encoding="utf-8",
            )
            cap_src = source / "capabilities" / "mylib"
            cap_src.mkdir(parents=True)
            (cap_src / "__init__.py").write_text("x = 1\n", encoding="utf-8")

            installer = Installer(
                pkg_dir=pkg_dir,
                connectors=ConnectorRegistry(),
                memory=None,
                lumen_dir=pkg_dir / "lumen",
            )
            result = installer.install_from_local_path(source)
            self.assertEqual(result["status"], "installed")

            cap_dest = pkg_dir / "lumen" / "capabilities" / "mylib"
            self.assertTrue(cap_dest.exists())

            # Verify initial install is read-only
            self.assertEqual(cap_dest.stat().st_mode & stat.S_IWUSR, 0)

            # Update source content and reinstall via zip (different path)
            import zipfile
            from io import BytesIO

            import yaml

            buf = BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr(
                    "my-mod/module.yaml",
                    yaml.dump({"name": "my-mod", "capabilities": ["mylib"]}),
                )
                zf.writestr("my-mod/SKILL.md", "# my-mod\n")
                zf.writestr("my-mod/capabilities/mylib/__init__.py", "x = 2\n")

            # Need to uninstall first so zip install doesn't get "already_installed"
            installer.uninstall("my-mod")
            result = installer.install_from_zip(buf.getvalue())
            self.assertEqual(result["status"], "installed")

            # Content should be updated
            self.assertIn("x = 2", (cap_dest / "__init__.py").read_text(encoding="utf-8"))

            # And still read-only
            self.assertEqual(
                cap_dest.stat().st_mode & stat.S_IWUSR,
                0,
                "Capability directory should remain read-only after reinstall",
            )
            self.assertEqual(
                (cap_dest / "__init__.py").stat().st_mode & stat.S_IWUSR,
                0,
                "Capability file should remain read-only after reinstall",
            )


class TerminalCapabilityPathTests(unittest.TestCase):
    def test_terminal_env_includes_capability_paths(self):
        config = {
            "_capability_paths": {
                "mod-a": ["/home/user/.lumen/capabilities/lib-a"],
                "mod-b": ["/home/user/.lumen/capabilities/lib-b"],
            }
        }
        env = _build_terminal_env(config)
        self.assertIn("PYTHONPATH", env)
        paths = env["PYTHONPATH"].split(os.pathsep)
        self.assertIn("/home/user/.lumen/capabilities/lib-a", paths)
        self.assertIn("/home/user/.lumen/capabilities/lib-b", paths)

    def test_terminal_env_preserves_existing_pythonpath(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            config = {
                "_capability_paths": {
                    "mod": ["/cap/path"],
                }
            }
            env = _build_terminal_env(config)
            self.assertIn("/cap/path", env["PYTHONPATH"])
            self.assertIn("/existing/path", env["PYTHONPATH"])

    def test_terminal_env_without_capabilities_has_no_pythonpath(self):
        with patch.dict(os.environ, {}, clear=True):
            config = {}
            env = _build_terminal_env(config)
            self.assertNotIn("PYTHONPATH", env)


class DiscoveryCapabilityTests(unittest.TestCase):
    def test_discovers_installed_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            lumen_dir = Path(tmp)
            cap_dir = lumen_dir / "capabilities" / "mylib"
            cap_dir.mkdir(parents=True)

            registry = Registry()
            _discover_capabilities(registry, [lumen_dir / "capabilities"])

            cap = registry.get(CapabilityKind.LIBRARY, "mylib")
            self.assertIsNotNone(cap)
            self.assertEqual(cap.status, CapabilityStatus.READY)
            self.assertIn("path", cap.metadata)


class CatalogCapabilityTests(unittest.TestCase):
    def test_catalog_lists_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.yaml"
            index.write_text(
                "capabilities:\n  - name: my-cap\n    description: A capability\n",
                encoding="utf-8",
            )
            catalog = Catalog(index)
            caps = catalog.list_capabilities()
            self.assertEqual(len(caps), 1)
            self.assertEqual(caps[0]["name"], "my-cap")

    def test_catalog_get_capability(self):
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index.yaml"
            index.write_text(
                "capabilities:\n  - name: find-me\n    path: /some/path\n",
                encoding="utf-8",
            )
            catalog = Catalog(index)
            cap = catalog.get_capability("find-me")
            self.assertIsNotNone(cap)
            self.assertEqual(cap["path"], "/some/path")


class ModuleRuntimeManagerCapabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_activate_sets_capability_paths_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lumen_dir = tmp / "lumen"
            cap_dir = lumen_dir / "capabilities" / "helper"
            cap_dir.mkdir(parents=True)
            (cap_dir / "__init__.py").write_text("", encoding="utf-8")

            module_dir = lumen_dir / "modules" / "test-mod"
            module_dir.mkdir(parents=True)
            (module_dir / "module.yaml").write_text(
                "name: test-mod\ncapabilities:\n  - helper\n",
                encoding="utf-8",
            )
            (module_dir / "connector.py").write_text(
                "def activate(ctx):\n    return None\n",
                encoding="utf-8",
            )

            connectors = ConnectorRegistry()
            config: dict = {}
            manager = ModuleRuntimeManager(
                pkg_dir=tmp / "pkg",
                lumen_dir=lumen_dir,
                config=config,
                connectors=connectors,
                memory=None,  # type: ignore[arg-type]
            )

            await manager._activate("test-mod", module_dir)
            self.assertIn("_capability_paths", config)
            self.assertIn("test-mod", config["_capability_paths"])
            self.assertEqual(
                config["_capability_paths"]["test-mod"], [str(cap_dir.resolve())]
            )
            await manager.close()

    async def test_unload_removes_capability_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            lumen_dir = tmp / "lumen"
            cap_dir = lumen_dir / "capabilities" / "helper"
            cap_dir.mkdir(parents=True)
            (cap_dir / "__init__.py").write_text("", encoding="utf-8")

            module_dir = lumen_dir / "modules" / "test-mod"
            module_dir.mkdir(parents=True)
            (module_dir / "module.yaml").write_text(
                "name: test-mod\ncapabilities:\n  - helper\n",
                encoding="utf-8",
            )
            (module_dir / "connector.py").write_text(
                "def activate(ctx):\n    return None\n",
                encoding="utf-8",
            )

            connectors = ConnectorRegistry()
            config: dict = {}
            manager = ModuleRuntimeManager(
                pkg_dir=tmp / "pkg",
                lumen_dir=lumen_dir,
                config=config,
                connectors=connectors,
                memory=None,  # type: ignore[arg-type]
            )

            await manager._activate("test-mod", module_dir)
            self.assertIn("test-mod", config.get("_capability_paths", {}))

            await manager.unload("test-mod")
            self.assertNotIn("test-mod", config.get("_capability_paths", {}))


if __name__ == "__main__":
    unittest.main()
