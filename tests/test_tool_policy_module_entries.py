"""Tests for module-declared tool policy (upstream plug E2).

Covers the scenarios of spec ``enchufes-upstream`` requirement E2: a module
declares the policy of its own tool and the Brain confirmation gate reads it;
a later config reload does not erase what the module declared.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import yaml

from lumen.core.brain import Brain
from lumen.core.catalog import Catalog
from lumen.core.confirmation_gate import ConfirmDecision, ConfirmResponse
from lumen.core.connectors import ConnectorRegistry
from lumen.core.memory import Memory
from lumen.core.module_runtime import ModuleRuntimeContext
from lumen.core.registry import Registry
from lumen.core.tool_policy import ToolPolicy


def _make_brain(tmp_path: Path) -> Brain:
    consciousness = tmp_path / "consciousness.yaml"
    consciousness.write_text(
        yaml.safe_dump({"identity": {"name": "TestLumen", "type": "test"}}),
        encoding="utf-8",
    )
    personality = tmp_path / "personality.yaml"
    personality.write_text(
        yaml.safe_dump({"identity": {"name": "TestLumen", "role": "Test"}}),
        encoding="utf-8",
    )
    catalog_path = tmp_path / "index.yaml"
    catalog_path.write_text(yaml.safe_dump({"modules": []}), encoding="utf-8")

    from lumen.core.consciousness import Consciousness
    from lumen.core.personality import Personality

    return Brain(
        consciousness=Consciousness(config_path=consciousness),
        personality=Personality(path=personality),
        memory=MagicMock(spec=Memory),
        connectors=ConnectorRegistry(),
        registry=Registry(),
        catalog=Catalog(catalog_path),
        model="test-model",
    )


class TestModuleDeclaredPolicy:
    async def test_module_policy_reaches_confirmation_gate(self, tmp_path, monkeypatch):
        """Spec scenario: the gate finds an entry declared by a module."""
        monkeypatch.delenv("LUMEN_TRUSTED_TOOLS", raising=False)
        brain = _make_brain(tmp_path)
        seen = []

        async def channel(request):
            seen.append(request)
            return ConfirmResponse(request.call_id, ConfirmDecision.APPROVED)

        brain.confirmation_gate.set_handler(channel)

        context = ModuleRuntimeContext(
            name="x-test",
            module_dir=tmp_path,
            runtime_dir=tmp_path / "runtime",
            manifest={},
            config={},
            connectors=brain.connectors,
            brain=brain,
        )

        async def publish(value: str):
            return value

        context.register_tool(
            "publicar_post",
            "Publishes",
            {"type": "object", "properties": {"value": {"type": "string"}}},
            publish,
        )
        context.register_tool_policy(
            "publicar_post",
            risk="mutating",
            confirm_required=True,
            description="declared by module",
        )

        decision = await brain._check_tool_confirmation(
            "publicar_post", "", {"value": "hola"}
        )

        assert decision == ConfirmDecision.APPROVED
        assert len(seen) == 1
        assert seen[0].tool_name == "publicar_post"
        assert seen[0].risk == "mutating"
        assert seen[0].description == "declared by module"

    def test_config_does_not_erase_module_entries(self, monkeypatch):
        """Spec scenario: config explicit > module > default, after reload too."""
        monkeypatch.delenv("LUMEN_TRUSTED_TOOLS", raising=False)
        policy = ToolPolicy()
        policy.load_defaults()
        policy.upsert_entry(
            "module_tool",
            risk="mutating",
            confirm_required=False,
            description="declared by module",
        )

        policy.load_config({"tool_policy": {"confirm_required": {"other_tool": True}}})
        entry = policy.get_policy("module_tool")
        assert entry.risk == "mutating"
        assert entry.confirm_required is False
        assert entry.description == "declared by module"
        assert policy.requires_confirmation(entry) is False

        policy.load_config({"tool_policy": {"confirm_required": {"module_tool": True}}})
        assert policy.requires_confirmation(policy.get_policy("module_tool")) is True

    def test_upsert_entry_with_action_matches_get_policy(self, monkeypatch):
        monkeypatch.delenv("LUMEN_TRUSTED_TOOLS", raising=False)
        policy = ToolPolicy()
        policy.upsert_entry(
            "task", "archive", risk="destructive", confirm_required=True
        )
        entry = policy.get_policy("task", "archive")
        assert entry.risk == "destructive"
        assert policy.requires_confirmation(entry) is True
