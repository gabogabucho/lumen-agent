from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.installer import Installer
from lumen.core.marketplace import Marketplace
from lumen.core.registry import Registry


FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return json.load(fh)


def _build_marketplace(config: dict | None = None) -> Marketplace:
    return Marketplace(
        catalog=Catalog(),
        registry=Registry(),
        connectors=ConnectorRegistry(),
        config=config or {},
    )


def _build_installer(tmp_path: Path) -> Installer:
    pkg_dir = tmp_path / "pkg"
    (pkg_dir / "modules").mkdir(parents=True, exist_ok=True)
    return Installer(
        pkg_dir=pkg_dir,
        connectors=ConnectorRegistry(),
        memory=None,
        catalog=Catalog(),
        lumen_dir=tmp_path / "lumen-home",
    )


def test_install_bridge_clawhub_missing_node_fails_gracefully():
    with tempfile.TemporaryDirectory() as tmp:
        installer = _build_installer(Path(tmp))

        with patch("lumen.core.installer.shutil.which", return_value=None):
            result = installer.install_marketplace_item(
                {
                    "name": "email-daily-summary",
                    "source_type": "clawhub",
                }
            )

        assert result["status"] == "error"
        assert "Install Node.js" in result["error"]


def test_install_bridge_clawhub_uses_npx_and_detects_new_module():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        installer = _build_installer(tmp_path)

        def fake_run(command, capture_output, text, check):
            assert command[0] == "npx"
            assert command[1:4] == ["clawhub@latest", "install", "email-daily-summary"]
            module_dir = installer.installed_dir / "email-daily-summary"
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "module.yaml").write_text(
                yaml.dump(
                    {
                        "name": "email-daily-summary",
                        "display_name": "Email Daily Summary",
                        "description": "Summarize email inboxes.",
                    }
                ),
                encoding="utf-8",
            )

            class Result:
                returncode = 0
                stderr = ""
                stdout = "ok"

            return Result()

        with patch("lumen.core.installer.shutil.which", return_value="npx"):
            with patch("lumen.core.installer.subprocess.run", side_effect=fake_run):
                with patch("lumen.core.installer.run_module_install_hook") as hook:
                    result = installer.install_marketplace_item(
                        {
                            "name": "email-daily-summary",
                            "source_type": "clawhub",
                            "description": "Summarize email inboxes.",
                        }
                    )

        assert result["status"] == "installed"
        assert result["name"] == "email-daily-summary"
        hook.assert_called_once()


def test_remote_mcp_registry_entry_marked_blocked_and_not_installable():
    market = _build_marketplace()
    payload = _load_fixture("mcp_registry.json")

    with patch.object(
        market,
        "_feed_configs",
        return_value=[
            {
                "name": "MCP Registry",
                "url": "https://registry.modelcontextprotocol.io/v0/servers?limit=3",
            }
        ],
    ):
        with patch.object(market, "_fetch_json", return_value=payload):
            snapshot = market.snapshot()

    card = next(
        item
        for item in snapshot["modules"]["items"]
        if item["name"] == "ac.inference.sh-mcp"
    )
    assert card["actions"]["can_install"] is False
    assert card["compatibility"]["status"] == "blocked"
    assert any(
        "remote MCP transport support" in reason
        for reason in card["compatibility"]["reasons"]
    )


def test_install_bridge_rejects_remote_mcp_transport():
    with tempfile.TemporaryDirectory() as tmp:
        installer = _build_installer(Path(tmp))

        result = installer.install_marketplace_item(
            {
                "name": "ac.inference.sh-mcp",
                "source_type": "mcp-registry",
                "remote_transport": {
                    "type": "streamable-http",
                    "url": "https://sh.inference.ac",
                },
            }
        )

    assert result["status"] == "error"
    assert "not installable yet" in result["error"]
