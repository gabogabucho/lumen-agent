# RFC: Lumen Workspaces

**Status:** ✅ IMPLEMENTED (v1.2.2)  
**Version:** 2.0.0  
**Target:** Lumen 1.2.x  
**Type:** Major Feature — Implemented  
**Author:** gabogabucho + neuronlabpro-coder  
**Date:** 2026-05-09  
**Closed:** 2026-05-11

---

## Principio de diseño

> **Lumen Workspace es Lumen — con gobernanza.**

Mismo Lumen. Misma instalación. Mismos módulos. Mismos skills. Solo que ahora múltiples personas pueden usarlo con identidad, roles y memoria propia desde el primer minuto.

Lo que cambia con Workspace:
- La webapp pide email + PIN por usuario
- Cada usuario tiene su memoria aislada
- Cada equipo ve solo los skills que el admin habilitó
- La empresa tiene su logo y nombre en la interfaz

Lo que NO cambia:
- Cómo se instalan módulos y skills
- Cómo funciona el Kit
- Cómo funciona Lumen internamente
- Todo lo demás

---

## 1. Resumen

Lumen Workspace convierte una instancia de Lumen en un agente corporativo compartido. Un solo Lumen sirve a toda la organización. Cada empleado tiene su perfil y su memoria. Cada equipo tiene acceso solo a los skills que le corresponden.

Los equipos pueden desarrollar sus propios skills y módulos — el admin los instala en Lumen como siempre y los habilita para el equipo que corresponda. Workspace no cambia ese flujo, solo agrega quién puede usarlo.

---

## 2. Arquitectura

### 2.1 Jerarquía

```
LUMEN INSTANCE
└── Workspace
    ├── workspace.yaml         # Branding + admin
    └── teams/
        ├── marketing/
        │   └── team.yaml      # Usuarios + skills habilitados
        ├── finanzas/
        │   └── team.yaml
        └── operaciones/
            └── team.yaml
```

### 2.2 Roles

| Role | Scope | Qué puede hacer |
|------|-------|-----------------|
| `admin` | Workspace | Todo. Gestiona equipos, usuarios y skills habilitados. |
| `team_admin` | Team | Gestiona usuarios de su equipo. |
| `member` | Team | Usa el agente con los skills de su equipo. |
| `viewer` | Team | Solo chat. Sin acceso a configuración. |

---

## 3. Archivos de configuración

### 3.1 workspace.yaml

```yaml
name: empresa-x
display_name: "Empresa X"

branding:
  logo: /assets/logo.png
  primary_color: "#1A1A2E"
  app_name: "Asistente Empresa X"

admins:
  - email: ceo@empresa-x.com
    display_name: "CEO"
    pin_hash: <bcrypt_hash>
```

### 3.2 team.yaml

```yaml
name: marketing
display_name: "Equipo de Marketing"

# Skills instalados en Lumen que este equipo puede usar
enabled_skills:
  - core-productivity
  - marketing-skill         # Desarrollado por el equipo de marketing
  - slack-skill             # Módulo de comunidad instalado por el admin

users:
  - email: director@empresa-x.com
    role: team_admin
    display_name: "Director de Marketing"
    pin_hash: <bcrypt_hash>

  - email: community@empresa-x.com
    role: member
    display_name: "Community Manager"
    pin_hash: <bcrypt_hash>
```

---

## 4. Auth

El PIN lo setea el admin directamente en el `team.yaml` y se lo comunica al usuario por el canal que prefiera. Sin SMTP, sin envío de emails — sin dependencias externas.

### Flujo de login

```
1. Usuario abre la URL de la instancia
2. Ve el branding de la empresa
3. Ingresa email + PIN
4. Lumen valida → genera JWT con { role, team }
5. La UI se adapta según su role
```

### Reset de PIN

El admin edita el `team.yaml` con el nuevo `pin_hash` y recarga Lumen:

```bash
lumen reload --instance empresa-x
```

No hay flujo automático de reset en v1 — el admin es el punto de contacto.

---

## 5. ACL de skills

Todos los skills instalados en la instancia están disponibles globalmente. Workspace agrega una capa de acceso: cada tool call valida que el skill esté en `enabled_skills` del equipo del usuario autenticado.

```
Usuario autenticado → role: member, team: marketing
Tool call: marketing-skill     → ✅ está en enabled_skills
Tool call: finance-skill       → ❌ no está en enabled_skills → respuesta limpia, sin exponer que existe
Role: admin                    → ✅ bypassa el ACL, accede a todo
```

El flujo de habilitación de un skill nuevo es el mismo de siempre:

```bash
# 1. Cualquiera desarrolla un skill
# 2. El admin lo instala en Lumen como siempre
lumen module install github:equipo-marketing/campanas-skill

# 3. El admin lo habilita en team.yaml
enabled_skills:
  - campanas-skill

# 4. Recarga
lumen reload --instance empresa-x
```

---

## 6. Dominios de memoria

Workspace no agrega sistemas de memoria externos. Usa el sistema de memoria que Lumen ya tiene, pero particionado: cada usuario y cada equipo arranca con una memoria limpia y aislada.

```
Memoria por usuario   → privada, solo la ve ese usuario
Memoria por equipo    → compartida entre miembros del equipo
Memoria global        → visible para todos (solo admin escribe)
```

Si la instancia usa Honcho u otro sistema de memoria externo, el adaptador de memoria de Lumen debe respetar estos dominios. Workspace define el modelo — la implementación de memoria es responsabilidad del conector que se use.

---

## 7. Branding

La webapp lee `GET /api/workspace/branding` al cargar — endpoint público, sin auth. Aplica logo, color y nombre antes de mostrar el login.

```
GET /api/workspace/branding
→ { logo_url, primary_color, app_name }
```

---

## 8. Bootstrap — wizard de instalación

```bash
lumen workspace init
```

Wizard de terminal que genera `workspace.yaml` + `teams/` con preguntas simples. Para configuración avanzada, el usuario edita los YAMLs directamente.

### Flujo del wizard

```bash
$ lumen workspace init

Nombre de la empresa: Empresa X
Email del admin: ceo@empresa-x.com
PIN del admin: ****

¿Cuántos equipos querés crear ahora? 2

  Equipo 1
  Slug: marketing
  Nombre: Equipo de Marketing
  Email del director: director@empresa-x.com
  PIN del director: ****

  Equipo 2
  Slug: finanzas
  Nombre: Equipo de Finanzas
  Email del director: cfo@empresa-x.com
  PIN del director: ****

✅ workspace.yaml generado
✅ teams/marketing/team.yaml generado
✅ teams/finanzas/team.yaml generado

Próximos pasos:
  1. Agregá usuarios a cada team.yaml
  2. Habilitá los skills que correspondan en enabled_skills
  3. lumen reload --instance <tu-instancia>
```

El wizard no toca módulos ni skills — eso es responsabilidad del admin como siempre.

---

## 9. Cambios requeridos en el codebase

### 9.1 Auth layer
- Reemplazar `owner_secret_hash` único por auth multi-usuario desde `workspace.yaml` + `teams/*/team.yaml`
- `POST /api/auth/login` → recibe `{ email, pin }`, devuelve JWT con `{ user_id, role, team }`
- Middleware JWT en todos los endpoints

### 9.2 Workspace loader
- Al startup, cargar `workspace.yaml` y todos los `teams/*/team.yaml`
- Índice en memoria: `email → { role, team, enabled_skills }`
- Recarga vía `lumen reload` sin restart

### 9.3 Skill ACL middleware
- Cada tool call valida `skill ∈ user.team.enabled_skills OR role == admin`
- Si no tiene permiso: respuesta limpia, sin exponer que el skill existe

### 9.4 Dominios de memoria
- Al crear sesión, resolver el dominio de memoria del usuario: `user:{email}`, `team:{team}`, `global`
- Memoria de usuario: aislada, solo visible para ese usuario
- Memoria de equipo: compartida entre miembros, el team_admin puede purgarla
- Memoria global: visible para todos, solo admin escribe

### 9.5 Branding API
- `GET /api/workspace/branding` → público, devuelve logo URL, color, app name
- La webapp aplica branding antes de renderizar el login

### 9.6 Personality resolver
- Si no hay `workspace.yaml`: comportamiento actual de Lumen sin cambios
- Si hay `workspace.yaml`: resolver personalidad según Kit activo + team del usuario

---

## 10. Breaking changes

> **Auth:** `owner_secret_hash` en `config.yaml` queda deprecated. Las instancias existentes sin `workspace.yaml` siguen funcionando igual — Workspace es opt-in.

> **UI:** Si existe `workspace.yaml`, la webapp muestra login antes de cargar. Sin `workspace.yaml`, comportamiento actual sin cambios.

---

## 11. Fuera de scope (v1)

- Envío de emails / SMTP
- Reset automático de PIN
- UI web avanzada (branding, multi-workspace, audit log)
- SSO / OAuth
- Audit log
- Múltiples Workspaces por instancia
- Branding por equipo (v1 es por Workspace)

---

## 12. v1.2.x — IMPLEMENTED ✅

### Checklist de cierre

| # | Requisito | Implementado | Commit |
|---|-----------|-------------|--------|
| 9.1 | Auth JWT, `POST /api/auth/login` con `{email, pin}` | ✅ Sí (collaborator `lumen/core/workspace.py` + nuestro `workspace_auth.py`) | d9dad48 |
| 9.2 | Workspace loader — `workspace.yaml` + `teams/*/team.yaml` | ✅ Sí (`core/workspace.py` + `runtime.py` `_load_workspace_index`) | merge |
| 9.3 | Skill ACL — tool call valida `skill ∈ enabled_skills` | ✅ Dos capas: session ACL (collaborator) + skill_acl.py (nuestro) | merge |
| 9.4 | Dominios de memoria — sesión escalada por workspace/team/email | Parcial — `_scoped_session_id` existe pero dominio de memoria persistente pendiente | brain.py |
| 9.5 | Branding API — `GET /api/workspace/branding` público | ✅ Sí (collaborator `web.py` + templates) | d9dad48 |
| 9.6 | Personality resolver — comportamiento dual (con/sin workspace.yaml) | ✅ Sí (`runtime.py` detecta workspace mode) | merge |
| 8   | CLI `lumen workspace init` wizard | ✅ Sí (`cli/workspace.py` — 303 líneas) | merge |
|     | CLI `lumen workspace user-add/remove/skill-enable/disable` | ⚠️ Esqueleto en `cli/workspace.py` (comandos registrados) | merge |
|     | `lumen reload` sin restart del servidor | ✅ `api/workspace/reload` en `web.py` | merge |
|     | `/settings/workspace` page | ✅ Template `settings_workspace.html` | merge |

**Veredicto: v1.2.2 listo.** Auth JWT, Teams, Skill ACL, Branding API y CLI workspace implementados y integrados con el código del collaborator (PR #13).

### Merge de v1.2.2

- PR original del collaborator: #13 → `d9dad48` (v1.2.1)
- Integración de workspaces propios: `0214bef` (merge con skill ACL layered, tests, CLI)
- Total: ~55k líneas añadidas por collaborator + ~6k líneas propias + ~5k líneas tests

---

## 13. Roadmap v1.3+ — Fusionado (RFC + Colaborador)

### v1.3.0 — Seguridad y fiabilidad (próximo sprint)

| # | Feature | Autor RFC | Autor | Descripción |
|---|---------|-----------|-------|-------------|
| 1 | **Dominios de memoria** | ✅ RFC §6 | — | Memoria aislada por usuario/equipo/global. Cada sesión resuelve su dominio al inicializarse. |
| 2 | **Version pinning + rollback** | — | Colaborador | Fijar versión por workspace + botón "volver a versión anterior" si algo falla. |
| 3 | **Sandbox de permisos por módulo** | — | Colaborador | Ver y limitar qué tools/endpoints puede usar cada módulo antes de instalar. |
| 4 | **Dry-run install / compat testing** | — | Colaborador | Validar dependencias y riesgos antes de aplicar cambios sin romper nada. |
| 5 | **Health checks por módulo** | — | Colaborador | Estado real (ready/degraded/down) con diagnóstico corto y última verificación. |

### v1.4.0 — Gobernanza

| # | Feature | Autor RFC | Autor | Descripción |
|---|---------|-----------|-------|-------------|
| 6 | **Centro de eventos y cambios** | — | Colaborador | Timeline de "quién instaló qué, cuándo y por qué" + export para compliance. |
| 7 | **Aprobación de instalación por rol/team** | — | Colaborador | Flujo "solicitar módulo" y aprobación por admin/team_admin con auditoría. |
| 8 | **Telemetría de uso por team** | — | Colaborador | Qué módulos aportan valor, uso por usuario/equipo, errores y latencia. |
| 9 | **Admin UI + CLI completa** | ✅ RFC §12 | — | UI web de gestión de equipos, usuarios, skills habilitados + CLI `workspace *` completa. |
|10 | **Plantillas de packs por departamento** | — | Colaborador | "Pack Sales", "Pack Soporte" — activar varios módulos de una vez. |

### v2.0.0 — Enterprise

| # | Feature | Autor RFC | Autor | Descripción |
|---|---------|-----------|-------|-------------|
|11 | **Firma y trust score** | — | Colaborador | Módulos firmados, publisher verificado, nivel de confianza visible en UI. |
|12 | **Políticas por entorno** | — | Colaborador | dev/staging/prod con reglas distintas de instalación y activación. |
|13 | **SSO / LDAP / OIDC** | ✅ RFC §11 | — | Autenticación corporativa más allá de email+pin. |
|14 | **Audit log completo** | ✅ RFC §11 | — | Registro de todas las acciones, exportable para compliance. |
|15 | **Multi-workspace por instancia** | ✅ RFC §11 | — | No múltiple workspace (v1 is single workspace). |

### Estimaciones acumuladas

| Versión | Alcance | Estimado | Dependencias |
|---------|---------|----------|--------------|
| **v1.2.2** | ✅ IMPLEMENTADO | — | — |
| **v1.3.0** | Dominios de memoria, version pinning, sandbox dry-run, health checks, compat testing | 3–4 semanas | Depende de v1.2.2 completo |
| **v1.4.0** | Centro de eventos, aprobación de instal., telemetría, admin UI+CLI, plantillas | 4–6 semanas | Depende de v1.3.0 |
| **v2.0.0** | Firma/trust score, políticas por entorno, SSO/audit log, multi-workspace | TBD — cuando v1.4.0 esté estable | Depende de v1.4.0 |

---

## 14. Contrato de integración con colaborador

El código de cada fase se integra de manera no destructiva:

- **Nuestro** (`workspace_auth.py`, `skill_acl.py`, `cli/workspace.py`, `core/workspace.py`) → capa de gobernanza + ACL
- **Colaborador** (`workspace.py` v1.2.1, `web.py` endpoints, templates, branding) → capa de auth + UI

Las dos capas operan en paralelo y se combinan en los puntos de integración:
- `web.py` — endpoints fusionados
- `runtime.py` — workspace index compartido
- `brain.py` — dos capas de ACL (session-based en collaborator + skill_acl.py en nuestro)
- `cli/main.py` — CLI del colaborador + CLI de workspaces propio

**Para nuevos colaboradores**: cada cambio nuevo debe mantener esta separación.

---

## 15. Cambios al RFC desde su versión draft

| Cambio | Razón |
|--------|-------|
| Status: Draft → ✅ IMPLEMENTED v1.2.2 | Auth, Teams, ACL, Branding, CLI completados |
| v1.2.0 → v1.2.2 | Incluyendo branding API, CLI init, reload endpoint |
| Roadmap fusionado RFC + propuestas colaborador | Las propuestas del colaborador cubren gaps del RFC |
| Dominios de memoria no movido a v1.2 | Requiere cambios en core de memoria, mejor en v1.3 |
| v2.0.0 ampliado con nuevas features | SSO + audit log del RFC + firma + trust score + env policies del colaborador |

(End of file - total 297 lines)
