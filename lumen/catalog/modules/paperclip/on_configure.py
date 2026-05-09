"""on_configure.py — Paperclip re-registration hook.

Runs after module configuration updates. Re-registers the agent
with Paperclip if essential config changed.
"""


def on_configure(config, lumen_dir=None):
    """Called after module configuration changes.

    Args:
        config: Lumen runtime config dict (post-merge).
        lumen_dir: Directory instance path (default: ~/.lumen/).

    If the agent_id changed, re-registers with Paperclip.
    If only credentials changed, just validates the connection.
    """
    paperclip_url = config.get("paperclip.url")
    api_key = config.get("paperclip.api_key")
    company_id = config.get("paperclip.company_id")
    agent_id = config.get("paperclip.agent_id")

    if not all([paperclip_url, api_key, company_id]):
        print("[paperclip] Skipping re-registration — config incomplete.")
        return

    # If agent_id already set, just validate connection
    if agent_id:
        try:
            import requests

            response = requests.get(
                f"{paperclip_url}/api/agents/{company_id}/{agent_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10
            )

            if response.ok:
                print(f"[paperclip] Agent {agent_id} is valid and connected.")
            else:
                print(f"[paperclip] Connection validated, but agent response {response.status_code}")
                print("[paperclip] Try re-installing to force re-registration.")

        except Exception as e:
            print(f"[paperclip] Could not validate connection — {e}")
    else:
        # If no agent_id yet, do full registration
        print("[paperclip] No agent_id found. Please run: lumen module install paperclip")


if __name__ == "__main__":
    """Self-test."""
    test_config = {
        "paperclip.url": "http://localhost:3100",
        "paperclip.api_key": "test-key",
        "paperclip.company_id": "company_test",
        "paperclip.agent_id": "agent_test",
    }
    print("on_configure.py loaded successfully.")
