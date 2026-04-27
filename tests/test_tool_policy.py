import unittest
from lumen.core.tool_policy import (
    ToolPolicy, ToolRisk, ToolPolicyEntry, SecurityConfig,
    DEFAULT_TOOL_RISK, DEFAULT_CONFIRM_REQUIRED,
)


class TestToolRiskDefaults(unittest.TestCase):
    def test_read_only_tools_exist(self):
        read_only = [a for a in DEFAULT_TOOL_RISK["message"].values() if a == "read_only"]
        assert len(read_only) >= 1

    def test_destructive_tools_exist(self):
        destructive = []
        for connector, actions in DEFAULT_TOOL_RISK.items():
            for action, risk in actions.items():
                if risk == "destructive":
                    destructive.append(f"{connector}__{action}")
        assert "task__delete" in destructive
        assert "note__delete" in destructive

    def test_privileged_tools_exist(self):
        assert DEFAULT_TOOL_RISK["terminal"]["execute"] == "privileged"

    def test_default_confirm_required(self):
        assert DEFAULT_CONFIRM_REQUIRED["read_only"] is False
        assert DEFAULT_CONFIRM_REQUIRED["destructive"] is True
        assert DEFAULT_CONFIRM_REQUIRED["privileged"] is True


class TestToolPolicyLoadDefaults(unittest.TestCase):
    def test_loads_all_builtin_connectors(self):
        policy = ToolPolicy()
        policy.load_defaults()
        assert len(policy._entries) >= 7  # message, memory, web, file, task, note, terminal

    def test_task_delete_is_destructive(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("task", "delete")
        assert entry.risk == "destructive"

    def test_memory_read_is_read_only(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("memory", "read")
        assert entry.risk == "read_only"

    def test_terminal_execute_is_privileged(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("terminal", "execute")
        assert entry.risk == "privileged"


class TestRequiresConfirmation(unittest.TestCase):
    def test_read_only_no_confirm_by_default(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("memory", "read")
        assert policy.requires_confirmation(entry) is False

    def test_destructive_requires_confirm(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("task", "delete")
        assert policy.requires_confirmation(entry) is True

    def test_privileged_requires_confirm(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("terminal", "execute")
        assert policy.requires_confirmation(entry) is True

    def test_unknown_tool_requires_confirm(self):
        policy = ToolPolicy()
        policy.load_defaults()
        entry = policy.get_policy("unknown_tool", "some_action")
        assert policy.requires_confirmation(entry) is True

    def test_auto_approve_read_only_override(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy._security_config.auto_approve_read_only = True
        entry = policy.get_policy("memory", "read")
        assert policy.requires_confirmation(entry) is False

    def test_confirm_deletions_off(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy._security_config.confirm_deletions = False
        entry = policy.get_policy("task", "delete")
        assert policy.requires_confirmation(entry) is False

    def test_confirm_terminal_off(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy._security_config.confirm_terminal = False
        entry = policy.get_policy("terminal", "execute")
        assert policy.requires_confirmation(entry) is False


class TestSecurityConfig(unittest.TestCase):
    def test_from_empty_config(self):
        config = SecurityConfig.from_config(None)
        assert config.confirm_deletions is True
        assert config.confirmation_timeout == 60

    def test_from_dict(self):
        config = SecurityConfig.from_config({
            "security": {
                "confirm_deletions": False,
                "confirmation_timeout": 120,
            }
        })
        assert config.confirm_deletions is False
        assert config.confirmation_timeout == 120

    def test_to_dict_roundtrip(self):
        original = SecurityConfig(confirm_deletions=False, confirmation_timeout=90)
        d = original.to_dict()
        restored = SecurityConfig(**{k: v for k, v in d.items() if k != "privileged_tool_names"})
        assert restored.confirm_deletions is False
        assert restored.confirmation_timeout == 90

    def test_non_dict_security_returns_defaults(self):
        config = SecurityConfig.from_config({"security": "invalid"})
        assert config.confirm_deletions is True


class TestConfigOverrides(unittest.TestCase):
    def test_confirm_override(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy.load_config({
            "tool_policy": {
                "confirm_required": {
                    "task__create": True,
                }
            }
        })
        entry = policy.get_policy("task", "create")
        assert entry.confirm_required is True

    def test_risk_override(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy.load_config({
            "tool_policy": {
                "risk_overrides": {
                    "memory__write": "destructive",
                }
            }
        })
        entry = policy.get_policy("memory", "write")
        assert entry.risk == "destructive"


class TestGetAllPolicies(unittest.TestCase):
    def test_returns_all_entries(self):
        policy = ToolPolicy()
        policy.load_defaults()
        all_policies = policy.get_all_policies()
        assert len(all_policies) >= 7
        for p in all_policies:
            assert "tool" in p
            assert "risk" in p
            assert "confirm_required" in p


class TestGetSummary(unittest.TestCase):
    def test_summary_structure(self):
        policy = ToolPolicy()
        policy.load_defaults()
        summary = policy.get_summary()
        assert "total_tools" in summary
        assert "by_risk" in summary
        assert "needs_confirmation" in summary
        assert summary["total_tools"] >= 7


class TestRecordAction(unittest.TestCase):
    def test_record_does_not_raise(self):
        policy = ToolPolicy()
        policy.load_defaults()
        policy.record_action("task", "delete", approved=True)
