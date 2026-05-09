"""on_configure.py — Honcho reconnection hook.

Runs after module configuration changes. Reconnects to Honcho
with updated credentials.
"""


def on_configure(config, lumen_dir=None):
    """Called after module configuration changes.

    Args:
        config: Lumen runtime config dict (post-merge).
        lumen_dir: Directory instance path (default: ~/.lumen/).

    Tears down the existing connection and recreates it with
    updated credentials. Updates module_setup on success.
    """
    # Step 1: Reset existing connection
    from .honcho import reset_honcho_client

    reset_honcho_client()
    print("[honcho] Reconnecting with updated config...")

    # Step 2: Read config values
    workspace_id = (config.get("honcho") or {}).get("workspace_id", "")
    api_key = (config.get("secrets") or {}).get("honcho", {}).get("api_key", "")
    base_url = (config.get("honcho") or {}).get("base_url", "")
    session_strategy = (config.get("honcho") or {}).get("session_strategy", "per-session")
    recall_mode = (config.get("honcho") or {}).get("recall_mode", "hybrid")

    # Step 3: Validate
    if not workspace_id or not str(workspace_id).strip():
        print("[honcho] workspace_id is missing — cannot connect.")
        print("[honcho] Set your workspace ID: lumen config set honcho.workspace_id <id>")
        _update_module_setup(config, "error", "workspace_id missing")
        return

    if not api_key or not str(api_key).strip():
        print("[honcho] API key is missing — cannot connect.")
        print("[honcho] Configure an API key: lumen config set honcho.api_key <key>")
        _update_module_setup(config, "error", "api_key missing")
        return

    # Step 4: Create new client and connect
    from .honcho import HonchoConfig, HonchoClient

    try:
        honcho_config = HonchoConfig.from_config(config)
        client = HonchoClient(honcho_config)
        sdk = client.connect()

        if sdk is None:
            print("[honcho] Could not connect to Honcho API.")
            print("[honcho] Verify your credentials and network.")
            _update_module_setup(config, "error", "Connection failed")
            return

        print("[honcho] Connected to Honcho API.")

    except ImportError as e:
        print(f"[honcho] honcho-ai import failed: {e}")
        _update_module_setup(config, "error", f"Import error: {e}")
        return
    except Exception as e:
        print(f"[honcho] Client error: {e}")
        _update_module_setup(config, "error", str(e))
        return

    # Step 5: Check if peer_id changed (new workspace)
    current_peer_id = (config.get("module_setup") or {}).get("honcho", {}).get("peer_id", "")
    new_peer_id = client.register_peer()

    if new_peer_id:
        print(f"[honcho] Peer registered: {new_peer_id}")

        # If peer_id changed, update module_setup
        if new_peer_id != current_peer_id:
            save_peer_id(config, new_peer_id)
            print("[honcho] Updated peer_id in config.")
    else:
        print("[honcho] Peer registration returned empty — keeping current peer_id.")

    # Step 6: Log success
    print("[honcho] Reconnection successful. Honcho memory is active.")
    _update_module_setup(config, "connected", "Reconnected successfully.")


def _update_module_setup(config, status, reason=""):
    """Update module_setup[honcho] with current status."""
    config.setdefault("module_setup", {})
    setup = config["module_setup"].setdefault("honcho", {})
    setup["status"] = status
    setup["reason"] = reason
    if status == "connected":
        setup["connected"] = True
    else:
        setup["connected"] = False


def save_peer_id(config, peer_id):
    """Store the peer_id in module_setup config."""
    config.setdefault("module_setup", {})
    setup = config["module_setup"].setdefault("honcho", {})
    setup["peer_id"] = peer_id
    setup["connected"] = True
    setup["status"] = "connected"
    setup["reason"] = "Connected and ready."
