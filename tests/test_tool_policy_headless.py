"""Policy for tools that are not built in, and for agents nobody is watching.

Two facts drove these tests, both found by running a real agent headless:

1. `tool_policy.risk_overrides` and `tool_policy.confirm_required` were applied
   only to keys already present in `_entries` — that is, only to built-in
   connector actions. A tool registered by a module or an MCP server could not
   be configured at all. It was accepted in config, no warning was logged, and
   the tool went on asking for a confirmation the operator had turned off.

2. With no dashboard attached, every confirmation ends as "Rejected by user
   (timeout)". Confirmation on a headless agent is not a gate — it is a slow,
   guaranteed denial. The agent then reports a tool failure to the end user,
   which is honest and useless.

The safe default is unchanged and these tests say so: an unknown tool with
nothing declared still requires confirmation.
"""

import os
import unittest
from unittest import mock

from lumen.core.tool_policy import ToolPolicy


TOOL = "ambar__recordar"


def _policy(config=None) -> ToolPolicy:
    p = ToolPolicy()
    p.load_defaults()
    p.load_config(config)
    return p


class TestUnknownToolStaysGuarded(unittest.TestCase):
    """The default is not what changed."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_unknown_tool_still_requires_confirmation(self):
        p = _policy()
        entry = p.get_policy(TOOL)
        assert entry.risk == "privileged"
        assert p.requires_confirmation(entry) is True

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_terminal_is_untouched(self):
        p = _policy()
        entry = p.get_policy("terminal", "execute")
        assert entry.risk == "privileged"
        assert p.requires_confirmation(entry) is True


class TestConfigReachesToolsThatAreNotBuiltIn(unittest.TestCase):
    """The bug: overrides silently did nothing for the tools that need them."""

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_risk_override_applies_to_a_module_tool(self):
        p = _policy({"tool_policy": {"risk_overrides": {TOOL: "read_only"}}})
        entry = p.get_policy(TOOL)
        assert entry.risk == "read_only"
        assert p.requires_confirmation(entry) is False

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_confirm_required_override_applies_to_a_module_tool(self):
        p = _policy({"tool_policy": {"confirm_required": {TOOL: False}}})
        assert p.requires_confirmation(p.get_policy(TOOL)) is False

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_the_key_is_the_tool_name_once(self):
        """No `tool__action__action`.

        `requires_confirmation` recomposed `tool_name__action` over a name that
        already carried the action, so the whitelist key had to repeat it. With
        `action=""` the key a caller writes is the key that matches.
        """
        p = _policy({"tool_policy": {"risk_overrides": {TOOL: "read_only"}}})
        assert p.requires_confirmation(p.get_policy(TOOL)) is False
        # And the doubled form is not what works.
        otro = _policy({"tool_policy": {"risk_overrides": {
            f"{TOOL}__recordar": "read_only"}}})
        assert otro.requires_confirmation(otro.get_policy(TOOL)) is True


class TestTrustedFromEnvironment(unittest.TestCase):
    """The switch a headless deployment can actually set."""

    @mock.patch.dict(os.environ, {"LUMEN_TRUSTED_TOOLS": TOOL}, clear=True)
    def test_env_trusts_a_tool(self):
        p = _policy()
        assert p.requires_confirmation(p.get_policy(TOOL)) is False

    @mock.patch.dict(os.environ,
                     {"LUMEN_TRUSTED_TOOLS": f" {TOOL} , ambar__clima ,,"},
                     clear=True)
    def test_whitespace_and_empty_items_are_dropped(self):
        p = _policy()
        assert p.requires_confirmation(p.get_policy(TOOL)) is False
        assert p.requires_confirmation(p.get_policy("ambar__clima")) is False
        # A trailing comma must not trust the empty name.
        assert p.requires_confirmation(p.get_policy("")) is True

    @mock.patch.dict(os.environ, {"LUMEN_TRUSTED_TOOLS": TOOL}, clear=True)
    def test_env_wins_over_config(self):
        p = _policy({"tool_policy": {"confirm_required": {TOOL: True}}})
        assert p.requires_confirmation(p.get_policy(TOOL)) is False

    @mock.patch.dict(os.environ, {"LUMEN_TRUSTED_TOOLS": "   "}, clear=True)
    def test_blank_env_trusts_nothing(self):
        p = _policy()
        assert p.requires_confirmation(p.get_policy(TOOL)) is True

    @mock.patch.dict(os.environ, {"LUMEN_TRUSTED_TOOLS": "ambar__clima"},
                     clear=True)
    def test_trusting_one_tool_does_not_trust_another(self):
        """The switch is a list, not a mode."""
        p = _policy()
        assert p.requires_confirmation(p.get_policy("ambar__clima")) is False
        assert p.requires_confirmation(p.get_policy(TOOL)) is True

    @mock.patch.dict(os.environ, {"LUMEN_TRUSTED_TOOLS": "terminal__execute"},
                     clear=True)
    def test_it_can_trust_a_built_in_too(self):
        """Nothing special-cases built-ins: the operator owns the decision."""
        p = _policy()
        assert p.requires_confirmation(p.get_policy("terminal", "execute")) is False


if __name__ == "__main__":
    unittest.main()
