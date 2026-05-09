"""on_install.py — Honcho memory setup hook.

Running once after installation:
1. Validate configuration
2. Install honcho-ai dependency
3. Connect and register peer
4. Test connectivity
"""


def on_install(config, lumen_dir=None):
    """Called once after module installation.

    Args:
        config: Lumen runtime config dict.
        lumen_dir: Directory instance path (default: ~/.lumen/).

    Installs the honcho-ai dependency, validates config, registers
    a peer with Honcho, and stores the peer_id in module config.
    """
    # Step 1: Validate config
    workspace_id = (config.get("honcho") or {}).get("workspace_id", "")
    api_key = (config.get("secrets") or {}).get("honcho", {}).get("api_key", "")
    session_strategy = (config.get("honcho") or {}).get("session_strategy", "per-session")
    recall_mode = (config.get("honcho") or {}).get("recall_mode", "hybrid")

    errors = []

    if not workspace_id or not str(workspace_id).strip():
        errors.append("[honcho] workspace_id is required — set your Honcho workspace ID.")
        errors.append("[honcho] Ejecute: lumen config set honcho.workspace_id <workspace_id>")

    if not api_key or not str(api_key).strip():
        errors.append("[honcho] API key is required — set your Honcho API key as a secret.")
        errors.append("[honcho] Ejecute: lumen config set honcho.api_key <api_key> --secret")

    valid_strategies = ["per-session", "per-directory", "per-repo", "global"]
    if session_strategy not in valid_strategies:
        errors.append(
            f"[honcho] session_strategy must be one of {valid_strategies} — got '{session_strategy}'."
        )
        errors.append(
            f"[honcho] session_strategy debe ser uno de {valid_strategies} — se recibió '{session_strategy}'."
        )

    valid_modes = ["hybrid", "context", "tools"]
    if recall_mode not in valid_modes:
        errors.append(
            f"[honcho] recall_mode must be one of {valid_modes} — got '{recall_mode}'."
        )
        errors.append(
            f"[honcho] recall_mode debe ser uno de {valid_modes} — se recibió '{recall_mode}'."
        )

    if errors:
        for err in errors:
            print(err)
        # Mark module setup as error
        _update_module_setup(config, "error", errors[0])
        return

    # Step 2: Install honcho-ai dependency
    print("[honcho] Installing honcho-ai dependency...")
    try:
        import subprocess
        result = subprocess.run(
            ["pip", "install", "honcho-ai"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            err_text = result.stderr.strip() or result.stdout.strip()
            print(f"[honcho] Failed to install honcho-ai: {err_text}")
            _update_module_setup(config, "error", f"Install failed: {err_text}")
            return
        print("[honcho] honcho-ai installed successfully.")
    except subprocess.TimeoutExpired:
        print("[honcho] pip install timed out after 60 seconds.")
        _update_module_setup(config, "error", "Install timeout")
        return
    except Exception as e:
        print(f"[honcho] Could not install honcho-ai: {e}")
        _update_module_setup(config, "error", f"Install error: {e}")
        return

    # Step 3: Create client and connect
    from .honcho import HonchoConfig, HonchoClient
    from pathlib import Path

    try:
        honcho_path = Path(__file__).resolve().parent / "honcho.py"
        # We need to import from the module — use sys path hack
        import sys
        module_dir = Path(__file__).resolve().parent
        if str(module_dir) not in sys.path:
            sys.path.insert(0, str(module_dir))

        honcho_config = HonchoConfig.from_config(config)
        client = HonchoClient(honcho_config)
        sdk = client.connect()

        if sdk is None:
            print("[honcho] Could not connect to Honcho API.")
            print("[honcho] Check your workspace_id, api_key, and network connection.")
            print("[honcho] Comprueba tu workspace_id, api_key y conexión de red.")
            _update_module_setup(config, "error", "Connection failed")
            return

        print("[honcho] Connected to Honcho API.")

    except ImportError as e:
        print(f"[honcho] honcho-ai import failed: {e}")
        print("[honcho] Make sure honcho-ai is installed: pip install honcho-ai")
        _update_module_setup(config, "error", f"Import error: {e}")
        return
    except Exception as e:
        print(f"[honcho] Could not create client: {e}")
        _update_module_setup(config, "error", f"Client error: {e}")
        return

    # Step 4: Register peer
    peer_id = client.register_peer()
    if peer_id:
        print(f"[honcho] Registered as peer {peer_id}")
        save_peer_id(config, peer_id)
    else:
        print("[honcho] Peer registration failed — retry with lumen reload.")
        _update_module_setup(config, "error", "Peer registration failed")
        return

    # Step 5: Test connectivity
    try:
        sessions = client.list_sessions()
        print(f"[honcho] Connectivity test passed. Found {len(sessions.get('sessions', []))} sessions.")
    except Exception as e:
        print(f"[honcho] Connectivity test warning: {e}")
        # Not fatal — module still connected

    # Mark as connected
    _update_module_setup(config, "connected", "Connected and ready.")


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
