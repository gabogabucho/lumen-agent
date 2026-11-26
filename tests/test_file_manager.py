"""Handlers for the catalog file-manager module.

Files stay inside a tmp root. Nothing here talks to the network.
"""

from __future__ import annotations

import base64
import errno
import importlib.util
import os
from pathlib import Path

import pytest

from lumen.core.connectors import ConnectorRegistry
from lumen.core.discovery import _discover_connectors
from lumen.core.module_runtime import ModuleRuntimeContext
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry
from lumen.core.tool_policy import ToolPolicy

REPO = Path(__file__).resolve().parents[1]
CONNECTOR_PATH = REPO / "lumen" / "catalog" / "modules" / "file-manager" / "connector.py"
BUILTIN = REPO / "lumen" / "connectors" / "built-in.yaml"
MODULE_NAME = "x-lumen-file-manager"
SENTINEL = "SENTINEL-OUTSIDE-9f3a"


def load_connector():
    spec = importlib.util.spec_from_file_location("file_manager_under_test", CONNECTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


mod = load_connector()


@pytest.fixture(autouse=True)
def _clear_file_env(monkeypatch):
    for name in ("LUMEN_FILE_ROOT", "LUMEN_FILE_MAX_BYTES", "LUMEN_TRUSTED_TOOLS"):
        monkeypatch.delenv(name, raising=False)


def registry_and_policy():
    registry = ConnectorRegistry()
    registry.load(BUILTIN)
    policy = ToolPolicy()
    policy.load_defaults()
    registry.tool_policy = policy
    return registry, policy


def context(tmp_path, connectors=None, brain=None, **settings):
    return ModuleRuntimeContext(
        name=MODULE_NAME,
        module_dir=tmp_path,
        runtime_dir=tmp_path / "runtime",
        manifest={},
        config={"modules": {MODULE_NAME: settings}},
        connectors=connectors,
        brain=brain,
    )


async def activated(tmp_path, *, brain=True, **settings):
    registry, policy = registry_and_policy()
    brain_obj = type("Brain", (), {"tool_policy": policy})() if brain else None
    ctx = context(tmp_path, connectors=registry, brain=brain_obj, **settings)
    runtime = await mod.activate(ctx)
    return ctx, registry, policy, runtime


def _outside(tmp_path) -> Path:
    path = tmp_path / "outside"
    path.mkdir()
    (path / "secret.txt").write_text(SENTINEL, encoding="utf-8")
    return path


async def test_round_trip_list_write_read_move_delete(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ctx, registry, _policy, runtime = await activated(tmp_path, root=str(root))
    listed = await registry.execute("file", "list", {"path": "."})
    assert listed == {"status": "ok", "path": ".", "entries": []}

    written = await registry.execute(
        "file", "write", {"path": "notes/a.txt", "content": "hello", "noise": 1}
    )
    assert written["status"] == "ok"
    assert written["path"] == "notes/a.txt"
    assert written["overwrote"] is False
    assert (root / "notes" / "a.txt").read_text(encoding="utf-8") == "hello"

    again = await registry.execute_tool("file__write", {"path": "notes/a.txt", "content": "updated"})
    assert again["overwrote"] is True

    read = await registry.execute_tool("file__read", {"path": "notes/a.txt", "extra": True})
    assert read["content"] == "updated"
    assert read["encoding"] == "utf-8"
    assert read["truncated"] is False
    assert not Path(read["path"]).is_absolute()

    empty = await registry.execute("file", "write", {"path": "empty.txt", "content": ""})
    assert empty["status"] == "ok"
    assert (root / "empty.txt").read_bytes() == b""

    moved = await registry.execute(
        "file", "move", {"source": "notes/a.txt", "destination": "b.txt"}
    )
    assert moved["status"] == "ok"
    assert moved["source"] == "notes/a.txt"
    assert moved["destination"] == "b.txt"
    assert not (root / "notes" / "a.txt").exists()
    assert (root / "b.txt").read_text(encoding="utf-8") == "updated"

    (root / "dir" / "sub").mkdir(parents=True)
    (root / "dir" / "sub" / "c.txt").write_text("c", encoding="utf-8")
    moved_dir = await runtime.move(source="dir", destination="renamed")
    assert moved_dir["status"] == "ok"
    assert (root / "renamed" / "sub" / "c.txt").read_text(encoding="utf-8") == "c"

    removed = await registry.execute("file", "delete", {"path": "b.txt"})
    assert removed == {"status": "ok", "path": "b.txt", "deleted": True}
    assert not (root / "b.txt").exists()

    kept = await registry.execute("file", "delete", {"path": "renamed"})
    assert kept["error"] == "not_empty"
    assert (root / "renamed" / "sub" / "c.txt").exists()

    wiped = await registry.execute_tool("file__delete", {"path": "renamed", "recursive": "true"})
    assert wiped["deleted"] is True
    assert not (root / "renamed").exists()
    await mod.deactivate(ctx, runtime)


async def test_jail_blocks_traversal_absolute_paths_and_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = _outside(tmp_path)
    outside_file = outside / "secret.txt"
    ctx, registry, _policy, runtime = await activated(tmp_path, root=str(root))

    traversal = await runtime.write(path="../sentinel-escape.txt", content="nope")
    assert traversal["error"] == "path_outside_root"
    assert SENTINEL not in traversal["message"]
    assert not (tmp_path / "sentinel-escape.txt").exists()
    assert ".." not in traversal["message"] or "escapes" in traversal["message"]

    nested = await runtime.write(path="foo/../../sentinel-escape.txt", content="nope")
    assert nested["error"] == "path_outside_root"
    assert not (tmp_path / "sentinel-escape.txt").exists()

    absolute = await runtime.read(path=str(outside_file))
    assert absolute["error"] == "path_outside_root"
    assert SENTINEL not in str(absolute)

    null = await runtime.read(path="a\x00b")
    assert null["error"] == "invalid_path"

    hidden = await runtime.write(path="..hidden", content="ok")
    assert hidden["status"] == "ok"
    assert (root / "..hidden").read_text(encoding="utf-8") == "ok"

    link = root / "link"
    link.symlink_to(outside_file)
    for action in (
        runtime.read(path="link"),
        runtime.write(path="link", content="overwrite"),
        runtime.delete(path="link"),
    ):
        result = await action
        assert result["error"] == "symlink_forbidden"
    assert outside_file.read_text(encoding="utf-8") == SENTINEL
    assert link.is_symlink()

    linked_dir = root / "sub"
    linked_dir.symlink_to(outside)
    descended = await runtime.read(path="sub/secret.txt")
    assert descended["error"] == "symlink_forbidden"
    assert outside_file.read_text(encoding="utf-8") == SENTINEL

    listing = await runtime.list(path=".")
    kinds = {entry["name"]: entry["type"] for entry in listing["entries"]}
    assert kinds["link"] == "symlink"
    assert kinds["sub"] == "symlink"
    assert "secret.txt" not in kinds
    assert SENTINEL not in str(listing)
    assert "target" not in listing["entries"][0]

    listed_link = await runtime.list(path="sub")
    assert listed_link["error"] == "symlink_forbidden"

    (root / "stay.txt").write_text("stay", encoding="utf-8")
    escaped = await runtime.move(source="stay.txt", destination=str(tmp_path / "out.txt"))
    assert escaped["error"] == "path_outside_root"
    assert (root / "stay.txt").read_text(encoding="utf-8") == "stay"
    assert not (tmp_path / "out.txt").exists()

    inside = await runtime.write(path=str(root / "abs.txt"), content="abs")
    assert inside["status"] == "ok"
    assert inside["path"] == "abs.txt"
    assert (root / "abs.txt").read_text(encoding="utf-8") == "abs"
    await mod.deactivate(ctx, runtime)


async def test_symlink_inside_tree_blocks_recursive_delete(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = _outside(tmp_path)
    box = root / "box"
    box.mkdir()
    (box / "a.txt").write_text("keep", encoding="utf-8")
    (box / "link").symlink_to(outside / "secret.txt")
    ctx, _registry, _policy, runtime = await activated(tmp_path, root=str(root))

    refused = await runtime.delete(path="box", recursive=True)
    assert refused["error"] == "symlink_forbidden"
    assert (box / "a.txt").read_text(encoding="utf-8") == "keep"
    assert (box / "link").is_symlink()
    assert (outside / "secret.txt").read_text(encoding="utf-8") == SENTINEL

    sibling = tmp_path / "root2"
    sibling.mkdir()
    (sibling / "keep.txt").write_text("sibling", encoding="utf-8")
    plain = root / "plain"
    plain.mkdir()
    (plain / "a.txt").write_text("gone", encoding="utf-8")
    wiped = await runtime.delete(path="plain", recursive=1)
    assert wiped["deleted"] is True
    assert not plain.exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "sibling"

    await runtime.delete(path=".", recursive=True)
    assert root.is_dir()
    assert (box / "a.txt").exists()
    dot = await runtime.delete(path=".", recursive="sí")
    assert dot["error"] == "root_forbidden"
    assert root.is_dir()
    await mod.deactivate(ctx, runtime)


async def test_move_and_write_refuse_clobber_and_bad_content(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ctx, _registry, _policy, runtime = await activated(tmp_path, root=str(root))
    await runtime.write(path="a.txt", content="source")
    await runtime.write(path="b.txt", content="dest")
    clobber = await runtime.move(source="a.txt", destination="b.txt")
    assert clobber["error"] == "dest_exists"
    assert (root / "a.txt").read_text(encoding="utf-8") == "source"
    assert (root / "b.txt").read_text(encoding="utf-8") == "dest"

    same = await runtime.move(source="a.txt", destination="a.txt")
    assert same["error"] == "same_path"
    assert (await runtime.move(source="", destination="b.txt"))["error"] == "source_required"
    assert (await runtime.move(source="a.txt", destination=" "))["error"] == "destination_required"
    assert (await runtime.move(source="missing.txt", destination="c.txt"))["error"] == "not_found"

    dest_link = root / "alias"
    dest_link.symlink_to(root / "b.txt")
    refused = await runtime.move(source="a.txt", destination="alias")
    assert refused["error"] == "symlink_forbidden"
    assert (root / "a.txt").exists()
    assert dest_link.is_symlink()

    broken = root / "broken"
    broken.symlink_to(root / "no-such")
    refused_broken = await runtime.move(source="a.txt", destination="broken")
    assert refused_broken["error"] == "symlink_forbidden"
    assert broken.is_symlink()

    bad = await runtime.write(path="num.txt", content=123)
    assert bad["error"] == "invalid_content"
    assert not (root / "num.txt").exists()

    fifo = root / "pipe"
    os.mkfifo(fifo)
    assert (await runtime.read(path="pipe"))["error"] == "invalid_path"
    await mod.deactivate(ctx, runtime)


async def test_size_limit_binary_and_missing_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    ctx, _registry, _policy, runtime = await activated(tmp_path, root=str(root), max_bytes="4")
    # max_bytes is a module setting, not a FileManagerRuntime kwarg. activate
    # builds the runtime itself, so the limit comes from config.
    assert runtime.max_bytes() == 4

    too_big = await runtime.write(path="big.txt", content="hello")
    assert too_big["error"] == "too_large"
    assert not (root / "big.txt").exists()

    (root / "raw.bin").write_bytes(b"hello")
    read = await runtime.read(path="raw.bin")
    assert read["truncated"] is True
    assert read["content"] == "hell"
    assert read["bytes"] == 4

    (root / "bin.dat").write_bytes(b"a\x00bcd")
    binary = await runtime.read(path="bin.dat")
    assert binary["encoding"] == "base64"
    assert base64.b64decode(binary["content"]) == b"a\x00bc"
    assert binary["truncated"] is True

    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("LUMEN_FILE_ROOT", str(missing))
    # Config root wins over the environment, so a fresh runtime without config
    # is what observes the missing directory.
    bare = mod.FileManagerRuntime(context(tmp_path))
    missed = await bare.read(path="a.txt")
    assert missed["error"] == "root_not_found"
    assert "does not exist" in missed["message"]
    assert not missing.exists()

    file_root = tmp_path / "not-a-dir"
    file_root.write_text("x", encoding="utf-8")
    monkeypatch.setenv("LUMEN_FILE_ROOT", str(file_root))
    missed_file = await mod.FileManagerRuntime(context(tmp_path)).list(path=".")
    assert missed_file["error"] == "root_not_found"
    await mod.deactivate(ctx, runtime)


async def test_default_root_relative_root_and_config_beats_env(tmp_path, monkeypatch):
    ctx, registry, _policy, runtime = await activated(tmp_path)
    listed = await registry.execute("file", "list", {})
    assert listed["status"] == "ok"
    default_root = tmp_path / "runtime" / "files"
    assert default_root.is_dir()
    await registry.execute("file", "write", {"path": "n.txt", "content": "default"})
    assert (default_root / "n.txt").read_text(encoding="utf-8") == "default"
    await mod.deactivate(ctx, runtime)

    relative = tmp_path / "runtime" / "box"
    relative.mkdir(parents=True)
    env_root = tmp_path / "env-root"
    env_root.mkdir()
    monkeypatch.setenv("LUMEN_FILE_ROOT", str(env_root))
    ctx, _registry, _policy, runtime = await activated(tmp_path, root="box")
    written = await runtime.write(path="from-config.txt", content="config")
    assert written["status"] == "ok"
    assert (relative / "from-config.txt").read_text(encoding="utf-8") == "config"
    assert not (env_root / "from-config.txt").exists()
    await mod.deactivate(ctx, runtime)

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "root-link"
    link.symlink_to(real)
    monkeypatch.setenv("LUMEN_FILE_ROOT", str(link))
    engine = mod.FileManagerRuntime(context(tmp_path))
    pinned = await engine.write(path="pinned.txt", content="inside")
    assert pinned["status"] == "ok"
    assert (real / "pinned.txt").read_text(encoding="utf-8") == "inside"
    assert not (tmp_path / "pinned.txt").exists()


async def test_eloop_from_open_is_symlink_forbidden(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_text("x", encoding="utf-8")
    engine = mod.FileManagerRuntime(context(tmp_path), max_bytes=100)
    monkeypatch.setenv("LUMEN_FILE_ROOT", str(root))

    def boom(*_args, **_kwargs):
        raise OSError(errno.ELOOP, "loop")

    monkeypatch.setattr(os, "open", boom)
    assert (await engine.read(path="a.txt"))["error"] == "symlink_forbidden"
    assert (await engine.write(path="b.txt", content="y"))["error"] == "symlink_forbidden"
    assert not (root / "b.txt").exists()


async def test_metadata_policy_discovery_and_no_duplicate_tools(tmp_path):
    registry, policy = registry_and_policy()
    body = Registry()
    _discover_connectors(body, registry)
    before = body.get(CapabilityKind.CONNECTOR, "file")
    assert before.status == CapabilityStatus.MISSING_HANDLER
    assert list(before.provides) == ["read", "write", "list"]
    assert registry.get("file")._handlers == {}

    brain = type("Brain", (), {"tool_policy": policy})()
    ctx = context(tmp_path, connectors=registry, brain=brain, root=str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    active = await mod.activate(ctx)

    file_connector = registry.get("file")
    assert file_connector.actions == ["read", "write", "list", "move", "delete"]
    assert set(file_connector._handlers) == {"read", "write", "list", "move", "delete"}

    _discover_connectors(body, registry)
    after = body.get(CapabilityKind.CONNECTOR, "file")
    assert after.status == CapabilityStatus.READY
    assert after.provides == ["read", "write", "list", "move", "delete"]

    tools = registry.as_tools()
    names = [tool["function"]["name"] for tool in tools]
    for name in ("file__read", "file__list", "file__write", "file__move", "file__delete"):
        assert names.count(name) == 1
    assert "terminal" in names and "terminal__execute" in names

    by_name = {tool["function"]["name"]: tool for tool in tools}
    write_meta = by_name["file__write"]["function"]["annotations"]
    delete_meta = by_name["file__delete"]["function"]["annotations"]
    move_meta = by_name["file__move"]["function"]["annotations"]
    read_meta = by_name["file__read"]["function"]["annotations"]
    assert write_meta["effect"] == "write"
    assert write_meta["risk"] == "mutating"
    assert write_meta["confirm_required"] is False
    assert move_meta["effect"] == "move" and move_meta["confirm_required"] is False
    assert read_meta["effect"] == "read" and read_meta["confirm_required"] is False
    assert delete_meta["effect"] == "delete"
    assert delete_meta["risk"] == "destructive"
    assert delete_meta["confirm_required"] is True

    stored = registry._explicit_tools["file__delete"]["metadata"]
    assert stored["kind"] == "module"
    assert stored["module"] == MODULE_NAME
    assert stored["confirm_required"] is True

    # Reload matches channels/web.py: sync, then load_defaults, then load_config.
    policy.load_defaults()
    policy.load_config({"security": {"confirm_deletions": True}})
    delete_entry = policy.get_policy("file", "delete")
    assert delete_entry.risk == "destructive"
    assert policy.requires_confirmation(delete_entry) is True
    write_entry = policy.get_policy("file", "write")
    assert write_entry.risk == "mutating"
    assert policy.requires_confirmation(write_entry) is False

    policy.load_config({"security": {"confirm_deletions": False}})
    assert policy.requires_confirmation(policy.get_policy("file__delete", "delete")) is False
    unknown = policy.get_policy("no_such_connector", "explode")
    assert unknown.risk == "privileged"
    assert policy.requires_confirmation(unknown) is True

    await mod.deactivate(ctx, active)
    assert registry.get("file").actions == ["read", "write", "list"]
    assert registry.get("file")._handlers == {}
    assert "file__delete" not in registry._explicit_tools
    with pytest.raises(ValueError):
        await registry.execute("file", "move", {"source": "a", "destination": "b"})


async def test_activate_without_brain_still_registers_tools(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    ctx, registry, _policy, runtime = await activated(tmp_path, brain=False, root=str(root))
    assert "file__write" in registry._explicit_tools
    assert registry._explicit_tools["file__write"]["metadata"]["confirm_required"] is False
    written = await registry.execute("file", "write", {"path": "a.txt", "content": "x"})
    assert written["status"] == "ok"
    await mod.deactivate(ctx, runtime)


def test_catalog_install_copies_connector(tmp_path):
    from lumen.core.catalog import Catalog
    from lumen.core.installer import Installer
    from lumen.core.memory import Memory

    installer = Installer(
        pkg_dir=REPO / "lumen",
        connectors=ConnectorRegistry(),
        memory=Memory(tmp_path / "memory.db"),
        catalog=Catalog(),
        lumen_dir=tmp_path / "home",
    )
    result = installer.install_from_catalog("file-manager")
    copied = tmp_path / "home" / "modules" / "file-manager" / "connector.py"
    assert result["status"] == "installed"
    assert copied.is_file()
    assert "file__delete" in copied.read_text(encoding="utf-8")
    assert not (REPO / "lumen" / "modules" / "file-manager").exists()


def test_builtin_file_delete_is_destructive_in_defaults():
    policy = ToolPolicy()
    policy.load_defaults()
    entry = policy.get_policy("file", "delete")
    assert entry.risk == "destructive"
    assert policy.requires_confirmation(entry) is True
    policy.load_config({"security": {"confirm_deletions": False}})
    assert policy.requires_confirmation(policy.get_policy("file", "delete")) is False
    assert policy.requires_confirmation(policy.get_policy("file", "move")) is False
    assert policy.get_policy("file", "move").risk == "mutating"
