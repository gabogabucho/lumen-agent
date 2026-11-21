# Proposal: WhatsApp Health Status in /health

## Intent

The `/health` endpoint currently reports brain status, modules ready, and provider health, but has no visibility into the WhatsApp bridge state. Operators cannot tell from the health check whether WhatsApp is connected, what number is in use, or if a QR scan is pending. This change surfaces WhatsApp connection metadata in the global health endpoint.

## Scope

### In Scope
- Extend `bridge.js` `/health` to return `connected`, `number`, and `session_status`.
- Extend `connector.py` `WhatsAppRuntime` to poll the bridge `/health` and cache state in `runtime_state`.
- Extend `web.py` `/health` to include a `whatsapp` field with the cached module state.

### Out of Scope
- Changes to WhatsApp send/receive behavior.
- Frontend UI for displaying WhatsApp status.
- WebSocket or push-based updates (keep existing polling model).

## Approach

1. **Bridge (`bridge.js`)**: Augment the `/health` endpoint to read `sock?.user?.id` (normalized to phone number), `connectionState`, and a `session_status` derived from whether auth creds exist and socket is open.
2. **Connector (`connector.py`)**: Add a `_poll_health` async task in `WhatsAppRuntime` that GETs `/health` every few seconds and writes a `whatsapp_health` blob into `runtime_state`. Start it alongside `_poll_loop`.
3. **Web (`web.py`)**: In `health_check()`, look up the WhatsApp module from `_brain.registry`, read its `runtime_state` via the module context, and inject a `whatsapp` dict into the JSON response. If the module is inactive, return `"status": "not_installed"`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `lumen/catalog/modules/x-lumen-comunicacion-whatsapp/bridge.js` | Modified | Extend `/health` endpoint with extra fields. |
| `lumen/catalog/modules/x-lumen-comunicacion-whatsapp/connector.py` | Modified | Add health polling task and state caching. |
| `lumen/channels/web.py` | Modified | Read WhatsApp state and include `whatsapp` key in `/health`. |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Bridge `/health` unreachable during startup | Low | Connector handles gracefully; marks state `"unknown"`. |
| Performance impact on `/health` | Low | Runtime state is a local file read; negligible overhead. |
| Backward compatibility break | Very Low | Adding a new top-level key `"whatsapp"` does not affect existing consumers. |

## Rollback Plan

1. Revert `bridge.js` `/health` to original payload shape.
2. Remove `_poll_health` task and `whatsapp_health` writes from `connector.py`.
3. Remove `whatsapp` key from `web.py` `health_check` return dict.

## Dependencies

- None external. Uses existing `urllib` patterns in connector.

## Success Criteria

- [ ] `GET /health` returns a `whatsapp` object when the WhatsApp module is active.
- [ ] `whatsapp` includes `status` (`"connected"`|`"disconnected"`|`"unknown"`|`"not_installed"`), `number`, and `session_status`.
- [ ] When the bridge is unreachable, `status` is `"unknown"` and the `/health` endpoint does not crash.
- [ ] No regression in existing `/health` fields.
