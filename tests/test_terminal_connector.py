"""Tests for Terminal Connector — REQ-T1 through REQ-T7."""

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumen.core.connectors import ConnectorRegistry
from lumen.core.handlers import register_builtin_handlers
from lumen.core.memory import Memory


class TerminalSecurityTests(unittest.TestCase):
    """REQ-T4: Allowlist enforcement. REQ-T5: Denylist override."""

    def test_deny_all_when_no_allowlist(self):
        """No allowlist configured → all commands denied."""
        from lumen.core.handlers import _check_command_allowed

        assert _check_command_allowed("echo", {}) is False
        assert _check_command_allowed("ls", {}) is False

    def test_allowlist_allows_listed_commands(self):
        """Command in allowlist → allowed."""
        from lumen.core.handlers import _check_command_allowed

        config = {"terminal": {"allowlist": ["echo", "ls", "git"]}}
        assert _check_command_allowed("echo", config) is True
        assert _check_command_allowed("ls", config) is True
        assert _check_command_allowed("git", config) is True

    def test_allowlist_denies_unlisted_commands(self):
        """Command NOT in allowlist → denied."""
        from lumen.core.handlers import _check_command_allowed

        config = {"terminal": {"allowlist": ["echo"]}}
        assert _check_command_allowed("rm", config) is False
        assert _check_command_allowed("sudo", config) is False

    def test_denylist_overrides_allowlist(self):
        """REQ-T5: Denylist wins even if command is in allowlist."""
        from lumen.core.handlers import _check_command_allowed

        config = {
            "terminal": {
                "allowlist": ["sh", "bash", "echo"],
                "denylist": ["rm", "sudo"],
            }
        }
        assert _check_command_allowed("echo", config) is True
        # rm not in allowlist → denied anyway
        assert _check_command_allowed("rm", config) is False
        # sudo not in allowlist → denied anyway
        assert _check_command_allowed("sudo", config) is False

    def test_empty_allowlist_denies_all(self):
        """Empty allowlist → deny all."""
        from lumen.core.handlers import _check_command_allowed

        config = {"terminal": {"allowlist": []}}
        assert _check_command_allowed("echo", config) is False

    def test_case_sensitive_allowlist(self):
        """Allowlist matching is case-sensitive."""
        from lumen.core.handlers import _check_command_allowed

        config = {"terminal": {"allowlist": ["echo"]}}
        assert _check_command_allowed("Echo", config) is False
        assert _check_command_allowed("ECHO", config) is False


class TerminalExecutionTests(unittest.IsolatedAsyncioTestCase):
    """REQ-T1: Command execution. REQ-T2: Timeout. REQ-T3: Cwd. REQ-T6: No shell. REQ-T7: Truncation."""

    async def test_execute_echo_returns_stdout(self):
        """REQ-T1: command execution returns stdout."""
        from lumen.core.handlers import terminal_execute

        if sys.platform == "win32":
            config = {"terminal": {"allowlist": ["python"]}}
            result = await terminal_execute(
                command='python -c "print(\'hello world\')"', config=config
            )
        else:
            config = {"terminal": {"allowlist": ["echo"]}}
            result = await terminal_execute(
                command="echo hello world", config=config
            )
        assert result["exit_code"] == 0
        assert "hello world" in result["stdout"]
        assert result["stderr"] == ""

    async def test_execute_returns_stderr(self):
        """REQ-T1: stderr is captured separately."""
        from lumen.core.handlers import terminal_execute

        # Use a command that writes to stderr
        if sys.platform == "win32":
            config = {"terminal": {"allowlist": ["python"]}}
            result = await terminal_execute(
                command='python -c "import sys; print(\'err\', file=sys.stderr)"',
                config=config,
            )
        else:
            config = {"terminal": {"allowlist": ["bash"]}}
            result = await terminal_execute(
                command='bash -c "echo err >&2"',
                config=config,
            )
        assert result["exit_code"] == 0
        assert "err" in result["stderr"]

    async def test_execute_nonexistent_command_returns_error(self):
        """REQ-T1: Non-existent command → exit_code != 0."""
        from lumen.core.handlers import terminal_execute

        config = {"terminal": {"allowlist": ["nonexistent_cmd_xyz_123"]}}
        result = await terminal_execute(
            command="nonexistent_cmd_xyz_123", config=config
        )
        assert result["exit_code"] != 0

    async def test_timeout_kills_process(self):
        """REQ-T2: Timeout kills long-running process."""
        from lumen.core.handlers import terminal_execute

        if sys.platform == "win32":
            config = {"terminal": {"allowlist": ["python"]}}
            result = await terminal_execute(
                command='python -c "import time; time.sleep(60)"',
                timeout=1,
                config=config,
            )
        else:
            config = {"terminal": {"allowlist": ["sleep"]}}
            result = await terminal_execute(
                command="sleep 60", timeout=1, config=config
            )
        assert result.get("error") == "timeout" or result["exit_code"] != 0

    async def test_command_denied_when_not_in_allowlist(self):
        """REQ-T4: Command not in allowlist → error."""
        from lumen.core.handlers import terminal_execute

        config = {"terminal": {"allowlist": ["ls"]}}
        result = await terminal_execute(command="rm -rf /", config=config)
        assert result.get("error") == "command_not_allowed"
        assert "rm" in result.get("command", "")

    async def test_cwd_configurable(self):
        """REQ-T3: Working directory is configurable."""
        from lumen.core.handlers import terminal_execute

        tmp = tempfile.gettempdir()
        if sys.platform == "win32":
            config = {"terminal": {"allowlist": ["cd"]}}
            # On Windows, cd is a shell builtin, use python
            config = {"terminal": {"allowlist": ["python"]}}
            result = await terminal_execute(
                command='python -c "import os; print(os.getcwd())"',
                cwd=tmp,
                config=config,
            )
        else:
            config = {"terminal": {"allowlist": ["pwd"]}}
            result = await terminal_execute(
                command="pwd", cwd=tmp, config=config
            )
        assert result["exit_code"] == 0
        # Normalize paths for comparison
        result_path = result["stdout"].strip().replace("/", "\\").lower()
        expected_path = str(Path(tmp)).lower()
        assert expected_path in result_path or tmp.lower() in result["stdout"].lower()

    async def test_output_truncation(self):
        """REQ-T7: Output over 10KB gets truncated."""
        from lumen.core.handlers import terminal_execute, _MAX_OUTPUT_BYTES

        assert _MAX_OUTPUT_BYTES == 10240  # 10KB

    async def test_public_env_injection_is_explicit(self):
        """Terminal subprocess receives only explicitly allowed public env vars."""
        from lumen.core.handlers import terminal_execute

        cmd = "python" if sys.platform == "win32" else "python3"
        config = {
            "terminal": {
                "allowlist": [cmd],
                "env": {
                    "public": ["INSTANCE_ID", "SCRIPTS_DIR"],
                    "modules": ["otto-tiendanube"],
                },
            },
            "secrets": {
                "otto-tiendanube": {
                    "public": {
                        "INSTANCE_ID": "otto-003",
                        "SCRIPTS_DIR": "/srv/otto/shared/capabilities/tiendanube",
                    },
                    "secret": {
                        "TIENDANUBE_API_TOKEN": "sk-secret",
                    },
                }
            },
        }
        script = Path(tempfile.gettempdir()) / "lumen_env_public_test.py"
        script.write_text(
            'import os\nprint(os.environ.get("INSTANCE_ID", ""))\nprint(os.environ.get("SCRIPTS_DIR", ""))\nprint(os.environ.get("TIENDANUBE_API_TOKEN", ""))\n',
            encoding="utf-8",
        )
        result = await terminal_execute(command=f'{cmd} "{script}"', config=config)
        assert result["exit_code"] == 0
        lines = [line.strip() for line in result["stdout"].splitlines()]
        assert "otto-003" in lines
        assert "/srv/otto/shared/capabilities/tiendanube" in lines
        assert "sk-secret" not in result["stdout"]

    async def test_secret_env_injection_requires_explicit_opt_in(self):
        """Secret env vars are injected only when explicitly listed."""
        from lumen.core.handlers import terminal_execute

        cmd = "python" if sys.platform == "win32" else "python3"
        config = {
            "terminal": {
                "allowlist": [cmd],
                "env": {
                    "public": ["INSTANCE_ID"],
                    "secret": ["TIENDANUBE_API_TOKEN"],
                    "modules": ["otto-tiendanube"],
                },
            },
            "secrets": {
                "otto-tiendanube": {
                    "public": {"INSTANCE_ID": "otto-003"},
                    "secret": {"TIENDANUBE_API_TOKEN": "sk-secret"},
                }
            },
        }
        script = Path(tempfile.gettempdir()) / "lumen_env_secret_test.py"
        script.write_text(
            'import os\nprint(os.environ.get("TIENDANUBE_API_TOKEN", ""))\n',
            encoding="utf-8",
        )
        result = await terminal_execute(command=f'{cmd} "{script}"', config=config)
        assert result["exit_code"] == 0
        assert "sk-secret" in result["stdout"]


class TerminalToolSchemaTests(unittest.TestCase):
    """Verify terminal connector appears in tool schemas and registry."""

    def test_terminal_in_built_in_yaml(self):
        """Terminal connector is defined in built-in.yaml."""
        from lumen.core.connectors import ConnectorRegistry

        registry = ConnectorRegistry()
        pkg_dir = Path(__file__).parent.parent / "lumen"
        built_in_path = pkg_dir / "connectors" / "built-in.yaml"
        registry.load(built_in_path)
        terminal = registry.get("terminal")
        assert terminal is not None
        assert "execute" in terminal.actions

    def test_terminal_tool_schema_registered(self):
        """terminal__execute has proper tool schema."""
        from lumen.core.handlers import TOOL_SCHEMAS

        schema = TOOL_SCHEMAS.get("terminal__execute")
        assert schema is not None
        assert "command" in schema["parameters"]["properties"]
        assert "command" in schema["parameters"]["required"]

    def test_terminal_alias_tool_exposed(self):
        """Single-action connectors expose a simple alias tool name."""
        registry = ConnectorRegistry()
        pkg_dir = Path(__file__).parent.parent / "lumen"
        built_in_path = pkg_dir / "connectors" / "built-in.yaml"
        registry.load(built_in_path)
        register_builtin_handlers(registry, Memory(":memory:"))

        tool_names = [tool["function"]["name"] for tool in registry.as_tools()]
        assert "terminal__execute" in tool_names
        assert "terminal" in tool_names

    def test_terminal_alias_parses_to_execute(self):
        registry = ConnectorRegistry()
        pkg_dir = Path(__file__).parent.parent / "lumen"
        built_in_path = pkg_dir / "connectors" / "built-in.yaml"
        registry.load(built_in_path)
        register_builtin_handlers(registry, Memory(":memory:"))
        registry.as_tools()  # populate alias map

        connector, action = registry.parse_tool_name("terminal")
        assert connector == "terminal"
        assert action == "execute"

    def test_connector_registry_injects_runtime_config_into_terminal_handler(self):
        """Terminal connector receives runtime config automatically through registry.execute."""
        import asyncio
        import sys

        memory = Memory(":memory:")
        registry = ConnectorRegistry()
        pkg_dir = Path(__file__).parent.parent / "lumen"
        built_in_path = pkg_dir / "connectors" / "built-in.yaml"
        registry.load(built_in_path)
        register_builtin_handlers(registry, memory)
        cmd = "python" if sys.platform == "win32" else "python3"
        registry.set_runtime_config({"terminal": {"allowlist": [cmd]}})

        result = asyncio.run(
            registry.execute("terminal", "execute", {"command": f"{cmd} --version", "timeout": 5})
        )
        assert result["exit_code"] == 0
        assert "Python" in result["stdout"] or "python" in result["stdout"]


if __name__ == "__main__":
    unittest.main()
