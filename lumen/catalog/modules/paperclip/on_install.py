"""on_install.py — Paperclip registration hook.

Runs once after installation. Registers this Lumen instance
as an agent in the Paperclip company.
"""

def on_install(config, lumen_dir=None):
    """Called once after module installation.

    Args:
        config: Lumen runtime config dict.
        lumen_dir: Directory instance path (default: ~/.lumen/).

    Registers this Lumen instance as an agent in Paperclip
    via the Paperclip registration API.
    """
    paperclip_url = config.get("paperclip.url")
    api_key = config.get("paperclip.api_key")
    company_id = config.get("paperclip.company_id")
    lumen_base_url = config.get("lumen.base_url", "http://localhost:3000")

    if not all([paperclip_url, api_key, company_id]):
        print("[paperclip] Skipping registration — config incomplete.")
        print("[paperclip] Run: lumen config set paperclip.url, api_key, company_id")
        return

    # Build callback URLs
    base = paperclip_url.rstrip("/")
    heartbeat_url = f"{base}/api/agents/{company_id}/heartbeat"

    try:
        import requests

        response = requests.post(
            f"{base}/api/agents/register",
            json={
                "name": lumen_dir.name if hasattr(lumen_dir, "name") else "lumen",
                "role": config.get("paperclip.agent_role", "Lumen Agent"),
                "company_id": company_id,
                "adapter": "http",
                "heartbeat_url": heartbeat_url,
            },
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )

        if response.ok:
            agent_id = response.json().get("agent_id")
            if agent_id:
                config.setdefault("paperclip.agent_id", agent_id)
                print(f"[paperclip] Registered as agent {agent_id}")
            else:
                print("[paperclip] Registration succeeded but no agent_id returned.")
        else:
            print(f"[paperclip] Registration failed — {response.status_code}")
            print("[paperclip] You can register manually via Paperclip dashboard.")

    except Exception as e:
        print(f"[paperclip] Could not reach Paperclip server — {e}")
        print("[paperclip] Configure paperclip.url and try: lumen reload")


if __name__ == "__main__":
    """Self-test."""
    test_config = {
        "paperclip.url": "http://localhost:3100",
        "paperclip.api_key": "test-key",
        "paperclip.company_id": "company_test",
    }
    print("on_install.py loaded successfully.")
