# Workspace Memory Contract (Adapters)

This contract is mandatory for external adapters (Honcho, Obsidian, custom connectors) to prevent cross-tenant leaks.

## Required Identity Inputs

- `workspace`
- `role` (`admin`, `team_admin`, `member`)
- `team` (for `team_admin`/`member`)
- `email` (for `member`)

## Domain Resolution

Use `lumen.core.workspace.resolve_memory_domain(...)`:

- `admin` -> `global:{workspace}`
- `team_admin` -> `team:{workspace}:{team}`
- `member` -> `user:{workspace}:{team}:{email}`

## Enforcement Rules

1. Read scope:
- `member`: own `user:*`, own `team:*`, own `global:*`
- `team_admin`: own `user:*`, own `team:*`, own `global:*`
- `admin`: all workspace prefixes (`user:{workspace}:`, `team:{workspace}:`, `global:{workspace}:`)

2. Write scope:
- `member`: only `user:*`
- `team_admin`: only `team:*`
- `admin`: only `global:*`

3. Never accept raw unscoped session IDs from external systems.
4. Prefix every persisted key, memory row, or path with the resolved domain.
5. Reject writes outside allowed scope with explicit `forbidden` errors.
