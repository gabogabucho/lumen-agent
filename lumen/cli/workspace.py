"""Workspace CLI wizard — `lumen workspace init`.

Interactive wizard that creates workspace.yaml, teams, and workspace_secret.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from lumen.core.workspace_auth import generate_workspace_secret, hash_pin, save_workspace_secret

workspace_app = typer.Typer(
    name="workspace",
    help="Manage workspace configuration.",
    no_args_is_help=True,
)

console = Console()

BRAND = "#3d3d6d"


# ── Prompt abstraction (for testability) ───────────────────────────


def _prompt(message: str, **kwargs) -> str:
    """Prompt wrapper — mocked in tests."""
    return Prompt.ask(message, **kwargs)


# ── Validation helpers ─────────────────────────────────────────────


def _validate_slug(slug: str, label: str = "slug") -> str:
    """Validate a slug: lowercase, alphanumeric + hyphens, no spaces."""
    if not re.match(r"^[a-z][a-z0-9-]*$", slug):
        raise ValueError(
            f"{label} inválido: debe empezar con letra minúscula, "
            "solo minúsculas, números y guiones (sin espacios)."
        )
    return slug


def _validate_email(email: str) -> str:
    """Basic email format validation."""
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise ValueError(f"Email inválido: '{email}'")
    return email


def _validate_pin(pin: str) -> str:
    """Validate PIN minimum length."""
    if len(pin) < 4:
        raise ValueError("El PIN debe tener al menos 4 caracteres.")
    return pin


def _confirm_pin(pin: str, confirm: str) -> str:
    """Check PIN matches confirmation."""
    if pin != confirm:
        raise ValueError("Los PINs no coinciden.")
    return pin


# ── Core init logic (pure-ish, testable) ───────────────────────────


def run_workspace_init(workspace_dir: Path, *, force: bool = False) -> dict:
    """Run the workspace init wizard.

    Creates workspace.yaml, workspace_secret, and teams/ structure.
    All prompts go through _prompt() for testability.

    Args:
        workspace_dir: Target directory for workspace files.
        force: Overwrite existing workspace.yaml if True.

    Returns:
        Dict with created file paths and summary info.

    Raises:
        FileExistsError: If workspace.yaml exists and force=False.
        ValueError: On invalid input.
    """
    ws_yaml = workspace_dir / "workspace.yaml"

    # Guard: existing workspace
    if ws_yaml.exists() and not force:
        raise FileExistsError(
            "Ya existe un workspace aquí. Usá --force para sobreescribir."
        )

    # Backup if --force
    if ws_yaml.exists() and force:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_path = ws_yaml.with_suffix(f".yaml.bak.{timestamp}")
        shutil.copy2(ws_yaml, backup_path)

    # ── Company setup ──
    console.print()
    console.print(Panel(
        "[bold cyan]Inicialización de Workspace[/bold cyan]\n\n"
        "Vamos a configurar tu workspace multi-tenant.",
        expand=False,
        border_style=BRAND,
    ))

    company_slug = _validate_slug(
        _prompt("Nombre del workspace (slug, ej: mi-empresa)"),
        label="Nombre del workspace",
    )
    company_display = _prompt("Nombre para mostrar (ej: Mi Empresa)")

    admin_email = _validate_email(
        _prompt("Email del administrador")
    )
    admin_name = _prompt("Nombre del administrador")

    admin_pin = _validate_pin(
        _prompt("PIN del administrador (mínimo 4 caracteres)")
    )
    admin_pin_confirm = _validate_pin(
        _prompt("Confirmar PIN")
    )
    _confirm_pin(admin_pin, admin_pin_confirm)

    # Collect emails for duplicate checking
    all_emails = {admin_email.lower()}

    # ── Team setup ──
    team_count = int(_prompt("¿Cuántos equipos querés crear ahora?", default="0"))

    teams: list[dict] = []
    for i in range(team_count):
        console.print(f"\n[bold]Equipo {i + 1}/{team_count}[/bold]")

        team_slug = _validate_slug(
            _prompt("Slug del equipo (ej: marketing)"),
            label="Slug del equipo",
        )
        team_display = _prompt("Nombre del equipo")

        team_email = _validate_email(
            _prompt("Email del administrador del equipo")
        )

        # Duplicate email check
        if team_email.lower() in all_emails:
            raise ValueError(
                f"Email duplicado: '{team_email}' ya está registrado."
            )
        all_emails.add(team_email.lower())

        team_admin_name = _prompt("Nombre del admin del equipo")

        team_pin = _validate_pin(
            _prompt("PIN del admin del equipo (mínimo 4 caracteres)")
        )
        team_pin_confirm = _validate_pin(
            _prompt("Confirmar PIN del admin del equipo")
        )
        _confirm_pin(team_pin, team_pin_confirm)

        teams.append({
            "slug": team_slug,
            "display_name": team_display,
            "admin_email": team_email,
            "admin_name": team_admin_name,
            "pin_hash": hash_pin(team_pin),
        })

    # ── Write workspace.yaml ──
    workspace_dir.mkdir(parents=True, exist_ok=True)

    workspace_data = {
        "name": company_slug,
        "display_name": company_display,
        "branding": {
            "logo": "",
            "primary_color": "#3d3d6d",
            "app_name": company_display,
        },
        "admins": [
            {
                "email": admin_email,
                "display_name": admin_name,
                "pin_hash": hash_pin(admin_pin),
            }
        ],
    }

    ws_yaml.write_text(
        yaml.dump(workspace_data, default_flow_style=False),
        encoding="utf-8",
    )

    # ── Write teams ──
    teams_dir = workspace_dir / "teams"
    teams_dir.mkdir(parents=True, exist_ok=True)

    created_teams: list[str] = []
    for team in teams:
        team_dir = teams_dir / team["slug"]
        team_dir.mkdir(parents=True, exist_ok=True)

        team_yaml_data = {
            "name": team["slug"],
            "display_name": team["display_name"],
            "enabled_skills": [],
            "users": [
                {
                    "email": team["admin_email"],
                    "role": "team_admin",
                    "display_name": team["admin_name"],
                    "pin_hash": team["pin_hash"],
                }
            ],
        }

        team_yaml = team_dir / "team.yaml"
        team_yaml.write_text(
            yaml.dump(team_yaml_data, default_flow_style=False),
            encoding="utf-8",
        )
        created_teams.append(str(team_yaml))

    # ── Generate workspace secret ──
    save_workspace_secret(workspace_dir, generate_workspace_secret())

    # ── Summary ──
    console.print()
    console.print(Panel(
        f"  Workspace: [bold]{company_display}[/bold] ({company_slug})\n"
        f"  Admin:     {admin_email}\n"
        f"  Equipos:   {team_count}",
        title="[green]Workspace creado[/green]",
        expand=False,
        border_style="green",
    ))

    console.print(f"\n  [bold]Archivos creados:[/bold]")
    console.print(f"    {ws_yaml}")
    console.print(f"    {workspace_dir / 'workspace_secret'}")
    for t in created_teams:
        console.print(f"    {t}")

    console.print()
    console.print(f"  [bold cyan]Próximos pasos:[/bold cyan]")
    console.print(f"    1. Agregar usuarios a los equipos")
    console.print(f"    2. Habilitar skills en cada equipo")
    console.print(f"    3. Ejecutar [bold]lumen reload[/bold] para aplicar")
    console.print()

    return {
        "workspace_yaml": str(ws_yaml),
        "workspace_secret": str(workspace_dir / "workspace_secret"),
        "teams": created_teams,
    }


# ── Typer command ──────────────────────────────────────────────────


@workspace_app.command("init")
def init(
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Sobreescribir workspace existente.",
    ),
    instance: str = typer.Option(
        None, "--instance", "-i",
        help="Instancia nombrada (directorio de datos aislado).",
    ),
    data_dir: str = typer.Option(
        None, "--data-dir", "-d",
        help="Directorio de datos personalizado.",
    ),
):
    """Inicializar un workspace multi-tenant.

    Crea workspace.yaml, teams/ y workspace_secret.
    """
    from lumen.core.paths import resolve_lumen_dir

    workspace_dir = resolve_lumen_dir(instance=instance, data_dir=data_dir)

    try:
        run_workspace_init(workspace_dir, force=force)
    except FileExistsError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
