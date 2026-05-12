# AGENTS.md

## Estado actual

### Backend / Workspaces
- Workspaces implementado por fases (1 a 5) con:
  - `workspace.yaml` + `teams/*/team.yaml`
  - auth multiusuario (`email + pin`)
  - ACL por team en tool calls
  - aislamiento de memoria por dominio (`user|team|global`) con namespacing en backend
  - governance UI/CLI operativa
- Contrato de seguridad de memoria para adaptadores externos:
  - `docs/security/WORKSPACE-MEMORY-CONTRACT.md`

### Frontend / Rediseño aplicado
- Rediseño integral del dashboard/settings/modules en modo light y dark.
- Menú lateral y paneles de settings unificados visualmente.
- Fondo de chat y capas visuales ajustadas para mejor legibilidad/UX.
- Sección **Modules** reestructurada:
  - toolbar superior con búsqueda + selects (`tipo`, `estado`, `categoría`, `orden`)
  - toggle de vista `grid/list`
  - cards uniformes con mejor jerarquía visual
  - acciones y metadatos movidos al footer de card (alineación consistente)
  - estados de compatibilidad con puntos CSS (sin iconografía fea legacy)
  - eliminación de duplicados visuales (`opaque`, metadatos repetidos)
- Persistencia de navegación interna:
  - al refrescar, conserva panel activo (`chat`, `marketplace`, `status`, `config`)
  - fallback seguro a `chat` si el panel guardado es inválido
- Navegación lateral unificada en vistas de operación:
  - `dashboard`, `memory` y `agent-status` comparten el mismo patrón visual (`Workspace / Monitoring / System`)
  - `Memory`, `Modules` y `Status` ya no fuerzan el “menú de settings”; `Settings` sí redirige al área de ajustes
- Branding del sidebar actualizado:
  - cabecera muestra solo `Lumen`
  - versión dinámica (`v{__version__}`) movida al footer junto a `Lumen is active`
  - favicon migrado a SVG del ojo de Lumen (`/static/favicon.svg`) y aplicado en templates web

### Accesibilidad y hardening UI
- Labels accesibles añadidos en controles de marketplace.
- `aria-label` aplicado a selects y toggles clave.
- Controles icon-only ajustados a touch target 44x44.
- Limpieza de código/DOM residual de filtros legacy.

### Auditoría de diseño
- `impeccable audit` re-ejecutado tras cambios.
- Resultado final: **20/20**.
- Reporte actualizado en:
  - `docs/phases/IMPECCABLE-AUDIT-REPORT.md`

### i18n UI (EN/ES)
- Infra i18n UI cargada desde JSON:
  - `lumen/locales/en/ui.json`
  - `lumen/locales/es/ui.json`
- Se normalizaron rutas de settings y páginas operativas para usar `ui_locale`.
- Dashboard EN/ES corregido para evitar mezcla de textos:
  - menú lateral (Chats/Memory/Modules/Status/Settings)
  - grupos de navegación (`Workspace / Monitoring / System`)
  - panel derecho de capacidades (`sense_native`, `sense_adapted`, `sense_opaque`)
- Se añadieron/ajustaron claves de locale para páginas:
  - `memory`, `agent_status`, `security`, `outputs`, `channels`, `confirmations_page`.

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

## Endpoints relevantes

- `POST /api/auth/login`
- `POST /api/auth/logout`
- `GET /api/workspace/branding`
- `GET /api/workspace/governance`
- `GET /settings/workspace`

## Seguridad aplicada

- Hash de PIN con `pbkdf2_sha256`.
- Cookie firmada con claims (`workspace`, `team`, `role`, `email`).
- ACL central en ejecución de tools (`brain`).
- Aislamiento de memoria por prefijos de dominio.
- Mutaciones de lessons restringidas a `admin` / `team_admin`.
- Logging reforzado para fallos LiteLLM en `brain.py` (`logger.exception` en fases críticas).

## Nota operativa

- Tras cambios en backend (auth/rutas/memory/brain), reiniciar servidor.
- Cambios solo de plantilla CSS/JS suelen verse con recarga, pero si hay duda reiniciar proceso.
