"""File manager — handlers for the built-in file connector.

Paths stay inside one configurable root. `..`, absolute paths outside that
root, and symlinks are refused. Write is mutating and does not open the
confirmation gate. Delete is destructive and does.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import logging
import os
import stat
from pathlib import Path

logger = logging.getLogger("lumen.module.file_manager")

MODULE_NAME = "x-lumen-file-manager"
DEFAULT_MAX_BYTES = 1_000_000

# effect / risk / confirm_required follow ToolPolicy: mutating writes do not
# ask, destructive deletes do (and honor confirm_deletions).
TOOL_POLICY = {
    "read": ("read", "read_only", False, "Read a file inside the file root."),
    "list": ("read", "read_only", False, "List entries inside the file root."),
    "write": ("write", "mutating", False, "Write a file inside the file root."),
    "move": ("move", "mutating", False, "Move a file or directory inside the file root."),
    "delete": ("delete", "destructive", True, "Delete a file or directory inside the file root."),
}


class PathJailError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _err(code: str, message: str, **extra) -> dict:
    payload = {"error": code, "message": message}
    payload.update(extra)
    return payload


def _clamp_int(value, default: int, low: int, high: int) -> int:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return default
    if number < low or number > high:
        return default
    return number


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "si", "sí"}
    return False


def _relative(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    text = rel.as_posix()
    return "." if text in {"", "."} else text


def confine(root: Path, user_path: str, *, allow_root: bool) -> Path:
    """Resolve `user_path` and refuse anything that leaves `root` or uses a symlink."""
    if user_path is None:
        raise PathJailError("path_required", "A path is required.")
    raw = str(user_path).strip()
    if not raw:
        raise PathJailError("path_required", "A path is required.")
    if "\x00" in raw or len(raw) > 4096:
        raise PathJailError("invalid_path", "The path is not valid.")
    root_real = root.resolve()
    if not root_real.is_dir():
        raise PathJailError("root_not_found", "The file root does not exist or is not a directory.")

    if os.path.isabs(raw):
        normalized = Path(os.path.normpath(raw))
        try:
            rel = normalized.relative_to(root_real)
        except ValueError as exc:
            raise PathJailError("path_outside_root", "Path escapes the file root.") from exc
    else:
        normalized_str = os.path.normpath(raw)
        if os.path.isabs(normalized_str):
            raise PathJailError("path_outside_root", "Path escapes the file root.")
        rel = Path(normalized_str)
        if any(part == ".." for part in rel.parts):
            raise PathJailError("path_outside_root", "Path escapes the file root.")

    parts = [part for part in rel.parts if part != "."]
    if not parts:
        if not allow_root:
            raise PathJailError("root_forbidden", "Refusing to operate on the file root itself.")
        return root_real

    current = root_real
    missing = False
    for index, part in enumerate(parts):
        if part in {"", ".", ".."}:
            raise PathJailError("path_outside_root", "Path escapes the file root.")
        current = current / part
        is_last = index == len(parts) - 1
        if missing:
            continue
        try:
            info = current.lstat()
        except FileNotFoundError:
            missing = True
            continue
        if stat.S_ISLNK(info.st_mode):
            raise PathJailError("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not is_last:
            raise PathJailError("not_a_directory", "A parent of the path is not a directory.")
        if not stat.S_ISREG(info.st_mode):
            raise PathJailError("invalid_path", "Only regular files and directories are allowed.")
    try:
        current.resolve().relative_to(root_real)
    except ValueError as exc:
        raise PathJailError("path_outside_root", "Path escapes the file root.") from exc
    return current


def _ensure_parent(path: Path, root: Path) -> None:
    parent = path.parent
    if parent == root:
        return
    rel = parent.relative_to(root)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise PathJailError("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
        if current.is_dir():
            continue
        if current.exists():
            raise PathJailError("not_a_directory", "A parent of the path is not a directory.")
        current.mkdir(mode=0o755)


def _contains_symlink(path: Path, root: Path) -> bool:
    if path.is_symlink():
        return True
    if not path.is_dir():
        return False
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        base = Path(dirpath)
        try:
            base.resolve().relative_to(root)
        except ValueError:
            return True
        for name in list(dirnames) + list(filenames):
            child = base / name
            if child.is_symlink():
                return True
            try:
                child.resolve().relative_to(root)
            except ValueError:
                return True
    return False


class FileManagerRuntime:
    def __init__(self, context, *, max_bytes: int | None = None):
        self.context = context
        self._max_bytes_override = max_bytes
        self._connector = None
        self._installed: list[tuple] = []
        self._added_actions: list[str] = []
        self._schema_names: list[str] = []

    def _setting(self, key: str, env_name: str | None = None) -> str | None:
        context = self.context
        if context is not None and hasattr(context, "resolve_setting"):
            value = context.resolve_setting(key, env_name)
            if value:
                return str(value)
        if env_name:
            raw = os.environ.get(env_name)
            if raw:
                return raw
        return None

    def max_bytes(self) -> int:
        if self._max_bytes_override is not None:
            return int(self._max_bytes_override)
        return _clamp_int(self._setting("max_bytes", "LUMEN_FILE_MAX_BYTES"), DEFAULT_MAX_BYTES, 1, 5_000_000)

    def resolve_root(self) -> tuple[Path | None, dict | None]:
        configured = self._setting("root", "LUMEN_FILE_ROOT")
        if configured:
            path = Path(configured).expanduser()
            if not path.is_absolute():
                base = Path(self.context.runtime_dir)
                path = base / path
            if path.is_symlink():
                resolved = path.resolve()
                if resolved.is_dir():
                    return resolved, None
            if not path.exists() or not path.is_dir():
                return None, _err(
                    "root_not_found",
                    "The configured file root does not exist or is not a directory.",
                )
            return path.resolve(), None
        default = Path(self.context.ensure_runtime_dir()) / "files"
        default.mkdir(parents=True, exist_ok=True)
        return default.resolve(), None

    def _call(self, fn, *args):
        try:
            return fn(*args)
        except PathJailError as exc:
            return _err(exc.code, exc.message)

    def _read_sync(self, path: str) -> dict:
        root, error = self.resolve_root()
        if error:
            return error
        target = confine(root, path, allow_root=False)
        if not target.exists():
            return _err("not_found", "The file does not exist.")
        if target.is_dir():
            return _err("not_a_file", "The path is a directory. Use file__list.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                return _err("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
            return _err("io_error", "The file could not be read.")
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return _err("not_a_file", "Only regular files can be read.")
            limit = self.max_bytes()
            chunks: list[bytes] = []
            remaining = limit + 1
            while remaining > 0:
                piece = os.read(fd, min(65536, remaining))
                if not piece:
                    break
                chunks.append(piece)
                remaining -= len(piece)
        finally:
            os.close(fd)
        data = b"".join(chunks)
        truncated = len(data) > limit
        if truncated:
            data = data[:limit]
        if b"\x00" in data:
            content = base64.b64encode(data).decode("ascii")
            encoding = "base64"
        else:
            content = data.decode("utf-8", errors="replace")
            encoding = "utf-8"
        return {
            "status": "ok",
            "path": _relative(root, target),
            "content": content,
            "encoding": encoding,
            "truncated": truncated,
            "bytes": len(data),
        }

    def _list_sync(self, path: str) -> dict:
        root, error = self.resolve_root()
        if error:
            return error
        raw = "." if str(path or "").strip() == "" else path
        target = confine(root, raw, allow_root=True)
        if not target.exists():
            return _err("not_found", "The directory does not exist.")
        if not target.is_dir():
            return _err("not_a_directory", "The path is not a directory.")
        entries = []
        with os.scandir(target) as scan:
            for entry in scan:
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "dir"
                elif entry.is_file(follow_symlinks=False):
                    kind = "file"
                else:
                    kind = "other"
                entries.append({"name": entry.name, "type": kind})
        entries.sort(key=lambda item: item["name"])
        return {"status": "ok", "path": _relative(root, target), "entries": entries}

    def _write_sync(self, path: str, content) -> dict:
        root, error = self.resolve_root()
        if error:
            return error
        if content is None:
            content = ""
        if not isinstance(content, str):
            return _err("invalid_content", "File content must be text.")
        data = content.encode("utf-8")
        if len(data) > self.max_bytes():
            return _err("too_large", "The content is larger than the configured limit.")
        target = confine(root, path, allow_root=False)
        if target.is_dir():
            return _err("not_a_file", "The path is a directory.")
        _ensure_parent(target, root)
        overwrote = target.exists()
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(target, flags, 0o644)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                return _err("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
            return _err("io_error", "The file could not be written.")
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        return {
            "status": "ok",
            "path": _relative(root, target),
            "bytes": len(data),
            "overwrote": overwrote,
        }

    def _move_sync(self, source: str, destination: str) -> dict:
        root, error = self.resolve_root()
        if error:
            return error
        if not str(source or "").strip():
            return _err("source_required", "A source path is required.")
        if not str(destination or "").strip():
            return _err("destination_required", "A destination path is required.")
        src = confine(root, source, allow_root=False)
        dest = confine(root, destination, allow_root=False)
        if src == dest:
            return _err("same_path", "Source and destination are the same path.")
        if not src.exists():
            return _err("not_found", "The source does not exist.")
        if dest.exists() or dest.is_symlink():
            if dest.is_symlink():
                return _err("symlink_forbidden", "Refusing to replace a symlink.")
            return _err("dest_exists", "The destination already exists.")
        _ensure_parent(dest, root)
        os.rename(src, dest)
        return {
            "status": "ok",
            "source": _relative(root, src),
            "destination": _relative(root, dest),
        }

    def _delete_sync(self, path: str, recursive) -> dict:
        root, error = self.resolve_root()
        if error:
            return error
        target = confine(root, path, allow_root=False)
        if target == root:
            return _err("root_forbidden", "Refusing to delete the file root.")
        if not target.exists() and not target.is_symlink():
            return _err("not_found", "The path does not exist.")
        if target.is_symlink():
            return _err("symlink_forbidden", "Refusing to delete a symlink.")
        if target.is_dir():
            if _contains_symlink(target, root):
                return _err("symlink_forbidden", "Refusing to delete a tree that contains a symlink.")
            if any(target.iterdir()):
                if not _as_bool(recursive):
                    return _err("not_empty", "The directory is not empty. Pass recursive=true to delete it.")
                self._remove_tree(target, root)
            else:
                target.rmdir()
        else:
            try:
                os.unlink(target)
            except OSError:
                return _err("io_error", "The file could not be deleted.")
        return {"status": "ok", "path": _relative(root, target), "deleted": True}

    def _remove_tree(self, path: Path, root: Path) -> None:
        for dirpath, dirnames, filenames in os.walk(path, topdown=False, followlinks=False):
            base = Path(dirpath)
            try:
                base.resolve().relative_to(root)
            except ValueError as exc:
                raise PathJailError("path_outside_root", "Path escapes the file root.") from exc
            for name in filenames:
                child = base / name
                if child.is_symlink():
                    raise PathJailError("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
                try:
                    child.resolve().relative_to(root)
                except ValueError as exc:
                    raise PathJailError("path_outside_root", "Path escapes the file root.") from exc
                child.unlink()
            for name in dirnames:
                child = base / name
                if child.is_symlink():
                    raise PathJailError("symlink_forbidden", "Refusing to follow a symlink inside the file root.")
                child.rmdir()
        path.rmdir()

    async def read(self, path: str = "", **_ignored):
        return await asyncio.to_thread(self._call, self._read_sync, path)

    async def list(self, path: str = "", **_ignored):
        return await asyncio.to_thread(self._call, self._list_sync, path)

    async def write(self, path: str = "", content: str = "", **_ignored):
        return await asyncio.to_thread(self._call, self._write_sync, path, content)

    async def move(self, source: str = "", destination: str = "", **_ignored):
        return await asyncio.to_thread(self._call, self._move_sync, source, destination)

    async def delete(self, path: str = "", recursive: bool = False, **_ignored):
        return await asyncio.to_thread(self._call, self._delete_sync, path, recursive)

    def bind(self, connector, schemas: dict):
        self._connector = connector
        if connector is None:
            return
        handlers = {
            "read": self.read,
            "write": self.write,
            "list": self.list,
            "move": self.move,
            "delete": self.delete,
        }
        for action, handler in handlers.items():
            if action not in connector.actions:
                connector.actions.append(action)
                self._added_actions.append(action)
            connector.register_handler(action, handler)
            self._installed.append((action, handler))
        setter = getattr(getattr(self.context, "connectors", None), "set_tool_schemas", None)
        if callable(setter) and schemas:
            setter(schemas)
            self._schema_names = list(schemas)

    def unbind(self):
        connector = self._connector
        if connector is not None:
            for action, handler in self._installed:
                if connector._handlers.get(action) is handler:
                    connector._handlers.pop(action, None)
            for action in self._added_actions:
                if action in connector.actions:
                    connector.actions.remove(action)
        schemas = getattr(getattr(self.context, "connectors", None), "_tool_schemas", None)
        if isinstance(schemas, dict):
            for name in self._schema_names:
                schemas.pop(name, None)
        self._installed = []
        self._added_actions = []
        self._schema_names = []
        self._connector = None


def _metadata(effect: str, risk: str, confirm_required: bool) -> dict:
    return {
        "kind": "module",
        "module": MODULE_NAME,
        "effect": effect,
        "risk": risk,
        "confirm_required": confirm_required,
    }


def _declare_policy(context, action: str, *, risk: str, confirm_required: bool, description: str) -> None:
    declare = getattr(context, "register_tool_policy", None)
    if not callable(declare):
        return
    try:
        declare("file", action, risk=risk, confirm_required=confirm_required, description=description)
    except RuntimeError:
        return


SCHEMAS = {
    "file__read": (
        "Read a text file inside the configured file root.",
        {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the file root."}},
            "required": ["path"],
        },
    ),
    "file__list": (
        "List files and directories inside the configured file root.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory relative to the file root. Defaults to the root."}
            },
        },
    ),
    "file__write": (
        "Create or replace a text file inside the configured file root.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the file root."},
                "content": {"type": "string", "description": "Full text content to write."},
            },
            "required": ["path", "content"],
        },
    ),
    "file__move": (
        "Move or rename a file inside the configured file root. Does not replace an existing destination.",
        {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Existing path relative to the file root."},
                "destination": {"type": "string", "description": "New path relative to the file root."},
            },
            "required": ["source", "destination"],
        },
    ),
    "file__delete": (
        "Delete a file or an empty directory inside the configured file root. "
        "Refuses symlinks. A non-empty directory needs recursive=true.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path relative to the file root."},
                "recursive": {
                    "type": "boolean",
                    "description": "Delete a non-empty directory and its contents. Default false.",
                },
            },
            "required": ["path"],
        },
    ),
}

_HANDLERS = {
    "file__read": "read",
    "file__list": "list",
    "file__write": "write",
    "file__move": "move",
    "file__delete": "delete",
}


async def activate(context):
    runtime = FileManagerRuntime(context)
    for tool_name, (description, parameters) in SCHEMAS.items():
        action = tool_name.split("__", 1)[1]
        effect, risk, confirm, policy_description = TOOL_POLICY[action]
        handler = getattr(runtime, _HANDLERS[tool_name])
        context.register_tool(
            tool_name,
            description,
            parameters,
            handler,
            metadata=_metadata(effect, risk, confirm),
        )
        _declare_policy(
            context,
            action,
            risk=risk,
            confirm_required=confirm,
            description=policy_description,
        )
    connector = None
    getter = getattr(getattr(context, "connectors", None), "get", None)
    if callable(getter):
        connector = getter("file")
    runtime.bind(
        connector,
        {name: {"description": spec[0], "parameters": spec[1]} for name, spec in SCHEMAS.items()},
    )
    return runtime


async def deactivate(context, runtime):
    if runtime is not None:
        runtime.unbind()
    unregister = getattr(context, "unregister_registered_tools", None)
    if callable(unregister):
        unregister()
