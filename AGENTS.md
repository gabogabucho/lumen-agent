# AGENTS.md

## Estado actual

- Workspaces implementado por fases (1 a 5) con:
  - `workspace.yaml` + `teams/*/team.yaml`
  - auth multiusuario (`email + pin`)
  - ACL por team en tool calls
  - aislamiento de memoria por dominio (`user|team|global`) con namespacing en backend
  - governance UI/CLI mínima
- Documento de contrato de seguridad para adaptadores externos:
  - `docs/security/WORKSPACE-MEMORY-CONTRACT.md`

## Arranque local (sin Docker)

```powershell
python -m lumen.cli.main run --port 3000 --data-dir .lumen-local --no-wizard
```

Server mode:

```powershell
python -m lumen.cli.main server --host 0.0.0.0 --port 3000 --data-dir .lumen-local --no-wizard
```

## Workspace CLI

Inicializar workspace:

```powershell
python -m lumen.cli.main workspace init --name <slug> --display-name "<name>" --admin-email <email> --user-email <email>
```

Ver governance:

```powershell
python -m lumen.cli.main workspace show --data-dir .lumen-local
```

Gestión de usuarios:

```powershell
python -m lumen.cli.main workspace user-add --team <team> --email <email> --role member
python -m lumen.cli.main workspace user-remove --team <team> --email <email>
```

Gestión de skills por team:

```powershell
python -m lumen.cli.main workspace skill-enable --team <team> --skill <skill_key>
python -m lumen.cli.main workspace skill-disable --team <team> --skill <skill_key>
```

## Endpoints workspace relevantes

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/workspace/branding`
- `GET /api/workspace/governance`
- `GET /settings/workspace`

## Seguridad aplicada

- Hash de PIN con `pbkdf2_sha256` (sin placeholders).
- Cookie firmada con claims (`workspace`, `team`, `role`, `email`).
- ACL central en ejecución de tools (brain).
- Aislamiento de memoria por prefijos de dominio.
- Mutaciones de lessons restringidas a roles `admin` / `team_admin` en workspace.

## Nota operativa

- Tras cambios de código en rutas/auth/memory, reiniciar proceso de servidor para cargar cambios.
