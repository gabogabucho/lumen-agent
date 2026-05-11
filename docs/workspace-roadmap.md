# Lumen Workspaces — Roadmap

> Multi-tenant workspaces con jerarquía **Company → Team → User**.

---

## ¿Qué es un Workspace?

Un Workspace convierte a Lumen en un motor multipropósito: puede servir a una persona (modo personal) o a una organización completa (modo empresa).

Cada workspace gestiona:

- **Equips y usuarios** con roles (`admin`, `team_admin`, `member`)
- **Memoria aislada** por dominio (`user`, `team`, `global`)
- **ACL de skills** por equipo
- **Branding corporativo** (logo, colores, nombre de app)
- **Gobernanza** desde CLI y UI

---

## Estado actual 🟢 V1.2.2 — Todo implementado

Las 5 fases del plan original están completas:

### ✅ Fase 1 — Foundation

- [x] Loader de `workspace.yaml` + `teams/*/team.yaml`
- [x] Validación de estructura mínima y construcción de índice en memoria
- [x] CLI wizard: `lumen workspace init`
- [x] Branding API: `GET /api/workspace/branding`
- [x] Feature flag implícito (sin `workspace.yaml` → comportamiento legacy intacto)

### ✅ Fase 2 — Auth multiusuario

- [x] Login por `email + pin`: `POST /api/auth/login`
- [x] Cookie firmada con claims: `workspace`, `team`, `role`, `email`
- [x] Middleware de auth protegiendo dashboard/settings/tools
- [x] Compatibilidad con modo owner legacy
- [x] Logout, expiración y rotación de secretos

### ✅ Fase 3 — ACL de skills por team

- [x] Hook de autorización de tool call (admin bypass)
- [x] Respuesta de denegación limpia (sin leaks de capacidades)
- [x] Auditoría estructurada de denegaciones

### ✅ Fase 4 — Memoria por dominios

- [x] Resolución de dominio: `user:{email}`, `team:{team}`, `global`
- [x] Política de lectura/escritura por rol
- [x] Contrato para adaptadores externos (Honcho/Obsidian)

### ✅ Fase 5 — Admin UX / governance

- [x] UI: listado de equipos, usuarios, skills habilitados
- [x] CLI: agregar/quitar usuarios, habilitar skills

---

## API actual

| Endpoint | Description |
|----------|-------------|
| `POST /api/auth/login` | Login workspace (`email` + `pin`) |
| `POST /api/auth/logout` | Cierre de sesión |
| `GET /api/workspace/branding` | Branding corporativo |
| `GET /api/workspace/governance` | Estado de gobernanza (teams, users, skills) |

---

## Estructura generada

```
<workspace>/
├── workspace.yaml          # Nombre, branding, admin
├── workspace_secret        # JWT secret (gitignored)
├── teams/
│   ├── engineering/
│   │   └── team.yaml       # Skills, usuarios
│   └── marketing/
│       └── team.yaml
```

### `workspace.yaml`

```yaml
name: mi-empresa
display_name: "Mi Empresa"
branding:
  logo: "/static/logo.png"
  app_name: "Lumen"
  primary_color: "#3d3d6d"
admins:
  - email: admin@miempresa.com
    display_name: "Admin"
    pin_hash: "pbkdf2_sha256$..."
```

### `teams/<slug>/team.yaml`

```yaml
name: engineering
display_name: "Ingeniería"
enabled_skills:
  - code-review
  - api-design
users:
  - email: dev@miempresa.com
    display_name: "Dev"
    role: member
    pin_hash: "pbkdf2_sha256$..."
```

---

## Aislamiento de memoria

Cada dominio se resuelve en:

- `admin` → `global:{workspace}`
- `team_admin` → `team:{workspace}:{team}`
- `member` → `user:{workspace}:{team}:{email}`

Las reglas de acceso:

| Rol | Lectura | Escritura |
|-----|---------|-----------|
| `member` | `user:*`, `team:*`, `global:*` propia | `user:*` propia |
| `team_admin` | `user:*`, `team:*`, `global:*` propias | `team:*` |
| `admin` | Todo lo del workspace | `global:*` |

---

## Próximos pasos (ideas para futuras fases)

### Fase 6 — Multi-tenancy real

- [ ] Separar bases de datos por workspace (no solo por prefijo)
- [ ] Migrar memoria a backend externo (Postgres + namespace)
- [ ] Soporte para workspaces que comparten una instancia de Lumen

### Fase 7 — Advanced governance

- [ ] Workspaces hijos (multi-empresa organizacional)
- [ ] SSO / OAuth2 / SAML
- [ ] Rate limiting por equipo
- [ ] Usage analytics / reportes por empresa

### Fase 8 — Marketplace integrado

- [ ] Catálogo privado por workspace
- [ ] Instalación de módulos desde catálogo administrado por admin
- [ ] Control de versiones por workspace

### Fase 9 — Audit & Compliance

- [ ] Logs de auditoría inmutables
- [ ] Export/backup de workspace completo
- [ ] Retención de datos por dominio (GDPR)
- [ ] Compliance reports para admins

---

## Notas operativas

- Tras cambios de código en rutas/auth/memory, reiniciar el servidor para cargar cambios.
- Los archivos `workspace_secret`, `.env` y datos de `data/` están en `.gitignore`.
- El workspace es opcional: sin `workspace.yaml`, Lumen funciona en modo single-user (legacy).
