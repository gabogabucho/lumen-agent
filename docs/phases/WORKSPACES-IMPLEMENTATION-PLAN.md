# Workspaces Implementation Plan (Enterprise / Teams / Users)

## Objetivo

Implementar Workspaces en Lumen con jerarquía empresarial:

- `Company` (Workspace)
- `Team`
- `User/Profile`

Con:

- autenticación multiusuario
- gobernanza por roles
- ACL de skills por team
- memoria aislada por dominio (`user`, `team`, `global`)
- branding corporativo

## Lo que pide el cliente (interpretación)

- Una empresa puede tener varios departamentos/equipos.
- Cada usuario pertenece a un equipo.
- Cada usuario tiene su perfil e identidad.
- Los equipos usan solo capacidades habilitadas por gobernanza.
- Integraciones como Honcho/Obsidian deben respetar dominios de memoria.

## Estado actual del codebase

- Auth actual orientada a `owner` único (cookie firmada + PIN/password).
- No existe modelo multiusuario en runtime web.
- Existe infraestructura útil para:
  - recarga en caliente (`lumen reload`)
  - control de herramientas / policy
  - capacidades y registry
  - memoria persistente

## Propuesta de implementación por fases

### Fase 1 — Foundation (sin breaking de instancias actuales)

1. [x] Loader de Workspace:
   - cargar `workspace.yaml` + `teams/*/team.yaml`
   - validar estructura mínima
   - construir índice en memoria:
     - `email -> {team, role, enabled_skills, display_name}`

2. [x] CLI bootstrap:
   - `lumen workspace init`
   - generar estructura base de YAML

3. [x] Branding API pública:
   - `GET /api/workspace/branding`
   - fallback seguro cuando no hay workspace

4. [x] Feature flag implícito:
   - si no hay `workspace.yaml`: comportamiento actual intacto
   - si hay `workspace.yaml`: habilitar modo Workspace

### Fase 2 — Auth multiusuario

1. [x] Endpoint login workspace:
   - `POST /api/auth/login` con `{email, pin}`
   - emitir JWT/cookie con claims:
     - `sub` (email o user_id)
     - `workspace`
     - `team`
     - `role`

2. [x] Middleware de auth:
   - proteger endpoints dashboard/settings/tools
   - mantener compatibilidad con modo owner legacy cuando no hay workspace

3. [x] Logout y expiración:
   - cierre de sesión
   - rotación segura de secretos de firma

### Fase 3 — ACL de skills por team

1. [x] Hook de autorización de tool call:
   - permitir `admin` global (bypass)
   - para otros roles: `tool/skill in team.enabled_skills`

2. [x] Respuesta de denegación limpia:
   - no revelar capacidades no habilitadas

3. [x] Auditoría mínima:
   - log estructurado de denegaciones

### Fase 4 — Memoria por dominios

1. [x] Resolver dominio por sesión:
   - `user:{email}`
   - `team:{team_slug}`
   - `global`

2. [x] Política de lectura/escritura:
   - `member`: user + team (global read)
   - `team_admin`: user + team (team admin ops)
   - `admin`: global full

3. [x] Adaptadores externos (Honcho/otros):
   - contrato para que respeten namespace de dominio

### Fase 5 — Admin UX / governance UX

1. [x] UI mínima:
   - listado de equipos
   - usuarios por equipo
   - skills habilitados por equipo

2. [x] CLI equivalente:
   - agregar/quitar usuario
   - habilitar/deshabilitar skills por team

## Modelo de datos sugerido

### workspace.yaml

- `name`
- `display_name`
- `branding` (`logo`, `primary_color`, `app_name`)
- `admins[]` (`email`, `display_name`, `pin_hash`)

### team.yaml

- `name`
- `display_name`
- `enabled_skills[]`
- `users[]` (`email`, `display_name`, `role`, `pin_hash`)

## Seguridad y hashing

- Actualmente Lumen ya usa `pbkdf2_sha256` para secretos de owner.
- Recomendación pragmática para v1:
  - reutilizar `pbkdf2_sha256` consistente con codebase
  - opcional v2: migrar/soportar `bcrypt` con estrategia dual

## Integración con Obsidian/Honcho

- Obsidian:
  - tratar cada team como vault lógico o prefijo de namespace
  - separar rutas por `workspace/team/user` para evitar fugas
- Honcho:
  - mapear cada memoria al dominio `user|team|global`
  - bloquear cross-read fuera de scope de rol

## Criterios de aceptación (MVP)

1. Una instancia con `workspace.yaml` permite login por `email + PIN`.
2. Dos usuarios de distintos teams no comparten memoria de usuario.
3. Un user de Team A no puede usar skill habilitado solo en Team B.
4. Branding corporativo se aplica en login/dashboard.
5. Sin `workspace.yaml`, Lumen sigue funcionando igual que hoy.

## Riesgos

- Mezclar auth legacy y workspace en la misma ruta sin feature gate claro.
- Fugas de memoria si no se propaga dominio en todos los paths de memoria.
- ACL incompleta si hay tool calls indirectas sin middleware central.

## Siguiente ejecución recomendada

1. Implementar Fase 1 completa.
2. Implementar Fase 2 con tests de auth.
3. Implementar Fase 3 con tests de ACL.
4. Integrar Fase 4 por etapas con pruebas de aislamiento.
