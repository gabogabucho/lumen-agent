# Plan — Pilar A: Contrato de Extensión v1

> **Alcance de esta sesión: M0 → M3.** M4–M6 quedan documentados para continuidad pero **fuera de alcance**.
>
> **Modo TDD estricto activo.** Cada hito es RED → GREEN. No se escribe implementación antes del test que la exige.
>
> **Regla de la cadena:** cada hito tiene una compuerta falsificable. Si la compuerta no pasa, **no se avanza — se retrocede**. No existe "más o menos funciona".

---

## 0. Contexto: por qué este trabajo y no otro

Decisiones de producto ya tomadas por el product owner. No están en discusión; se listan para que quien ejecute entienda el *por qué* de cada compuerta.

### Posicionamiento

> Hermes es el agente personal que crece con vos.
> **Lumen es el harness que despliegas para OTROS** — elegís la vertical, le pones tu marca, gobiernas quién puede qué, y lo entregás funcionando.

El comprador es **el integrador / la agencia**, no el usuario final. WordPress no ganó con blogs personales: ganó con agencias.

### Modelo de dos superficies

| Superficie | Audiencia | Regla de diseño |
|---|---|---|
| **CLI + archivos de config** | El integrador | Composable, scripteable, documentada, aburrida. **Jerga bienvenida.** |
| **Web UI** | El cliente final | Brandeable, simple, **sin jerga.** |

`README.md:69` dice *"No separate admin panel. No dev jargon."* Esa regla era correcta **para la superficie equivocada**. Se corrige en M6 (fuera de alcance).

### Por qué el contrato primero, y por qué la robustez es su primer cliente

Un contrato de extensión diseñado en abstracto sale mal. Este trae consumidor real incorporado: las ~1000 líneas de `brain.py` que compensan LLMs flojos se convierten en un módulo que consume **solo** el contrato.

**Si ese módulo no puede hacer su trabajo con el contrato solo, el contrato está incompleto — y lo aprendemos en M3, antes de publicárselo a terceros.** Ese es el valor de la secuencia.

### Qué NO hacer en esta sesión

Fuera de tesis. No agregar, ni "de paso":

- Más plataformas de mensajería (Hermes tiene 21 y 222k stars validándolo)
- TUI, app de escritorio
- Loop de auto-mejora de skills
- Multi-tenancy real / SSO (fases 6–7 del `workspace-roadmap.md`): van **después** del contrato. No sirve gobernanza empresarial sobre un ecosistema que no existe.
- Refactor de `brain.py` o `web.py` por pureza. Solo se toca lo que la cadena exige.

---

## 1. Hechos verificados del código

Auditados el 2026-07-29 sobre `feat/perfil-core`. **Si alguno cambió, detener y reportar** — el plan asume estos hechos.

### La superficie de extensión actual es de dos métodos

`ModuleRuntimeContext` ([lumen/core/module_runtime.py:119-245](../lumen/core/module_runtime.py)) expone a un módulo:

- `register_tool(name, description, parameters, handler, metadata)`
- `register_inbound_hook(channel, handler, *, timeout_seconds)`

Nada más. **Y una fuga:**

```python
brain: Any = None   # module_runtime.py:127
```

Un módulo recibe el brain crudo y puede monkey-patchear lo que quiera. Eso no es un contrato, es una puerta abierta. El docstring de `register_inbound_hook` lo confirma: *"should use this API instead of monkey-patching channel connectors"* — se le puso contrato al canal de entrada y se dejó el brain al aire.

### No existe versionado de contrato

```bash
grep -rn "manifest_version\|schema_version\|api_version" lumen/core/*.py
# 0 resultados
```

### Las heurísticas de robustez viven en 23 call sites, en 4 momentos distintos

| # | Momento | Call sites en `brain.py` | Funciones |
|---|---|---|---|
| 1 | **Pre-prompt** | 2159, 2162, 2200, 2203 | `_tool_enforcement_directive()`, `_suggest_relevant_tools()` (dentro de `_build_prompt()`) |
| 2 | **Post-respuesta cruda** | 2841, 3003, 3049, 3061, 3070, 3252, 3288, 3294, 3300, 3520 | `_extract_fallback_tool_calls`, `_sanitize_raw_tool_content`, `_resolve_tool_calls` |
| 3 | **Pre-ejecución de tool** | 3175, 3188, 3427, 3436 | `_coerce_args` |
| 4 | **Post-mensaje final** | 473, 1779, 1784, 1795, 1806 | `_guard_capability_claims`, `_detect_capability_denial`, `_build_contradiction_directive` |

Funciones auxiliares del mismo cuerpo: `_is_affirmative_claim`, `_is_denial_of_capability`, `_has_serialized_tool_call_shape`, `_is_usable_tool_arguments`, `_is_valid_tool_call`, `_has_usable_tool_calls`, `_tool_call_name`, `_tool_call_arguments`, `_allowed_tool_names`, `_sanitize_raw_tool_content`.

### Hay duplicación streaming / no-streaming

Los bloques **`brain.py:3049-3252`** y **`brain.py:3288-3520`** son el mismo código: una copia para streaming, otra para no-streaming. **Cada heurística está escrita dos veces.** Extraer el pipeline las unifica — ese es un beneficio adicional, no accidental.

### Provisioning

- Comandos raíz del CLI: `run`, `server`, `status`, `install`, `update`, `reload`, `doctor`. **No existe `lumen setup`.**
- El wizard es un prompt interactivo TTY que dispara `lumen run` cuando no hay config ([tests/test_cli_wizard.py:106](../tests/test_cli_wizard.py)), con flag `--no-wizard`.
- `_is_configured()` es `bool(config.get("model"))` ([lumen/channels/web.py:496](../lumen/channels/web.py)) — provisionar headless ya es posible vía `lumen config set` + `lumen model-set`, pero no es ciudadano de primera clase.

Relevante para M5 (fuera de alcance). **No tocar en esta sesión.**

### Tamaños relevantes

| Archivo | Líneas |
|---|---|
| `lumen/core/brain.py` | 3.570 |
| `lumen/channels/web.py` | 5.215 (81 rutas) |
| `lumen/cli/main.py` | 1.925 |
| `lumen/channels/templates/dashboard.html` | 2.633 |

Suite: 63 archivos de test, pytest, `asyncio_mode = auto`.

---

## 2. La cadena

```
M0 baseline ──► M1 contrato ──► M2 pipeline ──► M3 módulo ──╎ CORTE
  (confianza)     (promesa)       (forma)        (prueba)    ╎
                                                            ╎
                        M4 perfiles ──► M5 provisioning ──► M6 docs
                         (producto)        (negocio)      (ecosistema)
```

---

## M0 — Línea base confiable

**Depende de:** nada. Es el arranque.

### Por qué existe

M2 y M3 son refactor sobre el hot path del producto. Sin una base verde reproducible no se puede distinguir *"lo rompí yo"* de *"ya estaba roto"*. **No es formalidad:** la Fase 7 del checklist de `perfil-core` está stale, y en una sesión previa aparecieron 4 tests rotos preexistentes (cuelgue de websocket en `/ws` por `think_stream()`, y locks de tempdir en Windows).

### Trabajo

1. Correr la suite completa. **Registrar el conteo exacto de pass / fail / skip** en el reporte de cierre del hito. Este número es la referencia de M2.
2. Capturar RSS de línea base con `scripts/measure_rss.py`.
3. **Resolver `feat/perfil-core`**: verificar la Fase 7 contra el código real (el checklist no es confiable) y luego integrar a `main`.
4. Higiene del árbol, en **commit separado**:
   - destrackear `dev-archive/` (17 `.md` de manifiestos y planes viejos)
   - destrackear `.atl/skill-registry.md` (config de tooling, no del producto)
   - destrackear `README-SIN-DOCKER.md`
   - archivar el change `openspec/changes/honcho-memory-integration/` (el módulo honcho ya existe en el catálogo)
   - agregar `SKILL.md` y `.commit_msg` al `.gitignore` (hoy están untracked **sin** ignorar: entran al próximo `git add .`)

### Compuerta

- [ ] `pytest` verde en `main` con `perfil-core` integrado
- [ ] Conteo pass/fail/skip documentado
- [ ] RSS de línea base registrado
- [ ] `git status` limpio

### Nota de honestidad

La higiene **no es dependencia técnica** de la cadena. Se incluye porque cuesta poco y porque un árbol limpio hace legibles los diffs de M2/M3 — ahí sí es dependencia débil. Si se saca, la cadena no se rompe.

---

## M1 — Contrato de extensión v1 declarado

**Depende de M0:** con la suite verde, los tests nuevos son la única señal.

### Trabajo

1. **Versionado del manifest.** Campo `lumen_api: 1`. `lumen/core/module_manifest.py` valida y rechaza majors desconocidos con error legible. Módulos sin el campo: tratar como v1 con warning (compatibilidad hacia atrás con los módulos del catálogo actual).

2. **Declarar los 4 extension points como protocolos tipados**, uno por momento del loop (§1):
   - aumento de prompt
   - reparación de respuesta cruda
   - coerción de argumentos de tool
   - guard de mensaje final

3. **Semántica de fallo — crítica, no negociable.**

   **Fallo de un hook ⇒ descartar la transformación de ese hook y usar el valor ORIGINAL.**

   No se puede copiar la semántica de `invoke_inbound_hooks` (*"all failures pass through"*, loguea y sigue). Ahí es correcta: el peor caso es que el mensaje se rutee normal. En el pipeline de respuesta corrompe el turno:
   - si falla un hook de coerción de argumentos → se ejecuta una tool con argumentos basura
   - si falla un hook de sanitización → se filtra JSON crudo al usuario

4. **Timeouts obligatorios** por hook y **orden determinístico declarado**. Corre en cada turno: es hot path.

5. **Tests de contrato.** Un módulo falso registrando en los 4 puntos, verificando:
   - orden de invocación determinístico
   - aislamiento ante fallo (el valor original sobrevive)
   - aislamiento ante timeout
   - que un módulo escrito contra v1 sigue funcionando

6. **Política de compatibilidad escrita.** Qué se promete, qué puede romper, cómo se versiona. Documento corto y explícito.

### Compuerta

- [ ] Tests de contrato **verdes contra un pipeline stub** — todavía **sin cablear** al brain
- [ ] `module_manifest.py` valida `lumen_api`; módulos existentes del catálogo siguen cargando
- [ ] Política de compatibilidad escrita y en el repo
- [ ] Suite completa sigue verde (conteo de M0)

> Sin tests de contrato no hay contrato: hay documentación.

---

## M2 — Pipeline extraído y deduplicado

**Depende de M1:** no se puede extraer hacia un pipeline cuyo contrato no está definido.

### Trabajo

1. Colapsar los 23 call sites en un pipeline explícito, invocado desde el camino **streaming y no-streaming**.
2. **Eliminar la duplicación** de `brain.py:3049-3252` y `3288-3520`. El pipeline se define una vez; los dos caminos lo invocan.
3. Las heurísticas **siguen viviendo en el core**. Solo cambia por dónde pasan. **Cero cambio de comportamiento.**
4. Los tests de contrato de M1 pasan a correr contra el pipeline real, no el stub.

### Compuerta — la más dura de la cadena

- [ ] Suite verde **sin modificar una sola expectativa de test**
- [ ] Tests de contrato de M1 verdes contra el pipeline real
- [ ] Los rangos duplicados ya no existen

**Si un test necesita cambiar, el comportamiento cambió → se revierte.** Es binaria y no admite interpretación. Un test modificado en este hito es un fallo del hito, no un ajuste.

### Rollback

Un solo commit de refactor, revertible entero. No mezclar con otros cambios.

---

## M3 — Robustez extraída a módulo · **CORTE DE SESIÓN**

**Depende de M2:** el módulo se engancha al pipeline. Sin pipeline no hay dónde.

### Trabajo

1. Crear el módulo `x-lumen-model-robustness` que consume **solo el contrato v1**: sin imports del core, sin acceso a `brain`.
2. Mover las ~1000 líneas de heurísticas fuera de `brain.py` al módulo.
3. **Retirar `brain: Any` de `ModuleRuntimeContext`** ([module_runtime.py:127](../lumen/core/module_runtime.py)) o reducirlo a una fachada acotada y documentada.

### Compuerta — doble corrida

- [ ] **Con** el módulo instalado: los tests de robustez existentes pasan
- [ ] **Sin** el módulo: se marcan skip/xfail y **la suite del core sigue verde**
- [ ] `brain: Any` retirado o reducido a fachada acotada
- [ ] El módulo no importa nada de `lumen.core` fuera del contrato

Eso prueba modularidad de verdad, no de palabra.

### Este es el momento de la verdad del pilar A

**Si el módulo necesita tocar `brain` para hacer su trabajo, el contrato está incompleto.** No forzarlo: **reportarlo**. Descubrir un extension point faltante acá es el resultado exitoso del hito — es exactamente para eso que se eligió este cliente primero. Agregar un quinto extension point con su test de contrato es trabajo legítimo de M3. Filtrar acceso a `brain` para que "funcione" **anula el propósito de toda la cadena**.

### Por qué el corte está acá

**M3 es el único punto de la cadena donde el valor queda capturado aunque el trabajo se detenga.**

- Parar en M1 → un contrato sin cliente, no validado
- Parar en M2 → un refactor sin cliente
- Parar en M3 → contrato **validado por uso real**, core con ~1000 líneas menos, sin deuda abierta

---

## Fuera de alcance — para continuidad

Documentados para que no se pierdan. **No ejecutar en esta sesión.**

### M4 — Perfiles como producto (depende de M3)

| Perfil | Para quién | Contenido |
|---|---|---|
| `demo` / `eval` | El dev evaluando en 60 segundos | Modelo gratis + robustez + kit vertical precargado |
| `core` | Deploy real de cliente | Proveedor propio, sin robustez, `headless` |

Decidir el destino de `full`: ¿sigue existiendo o queda absorbido?

**Razonamiento:** el módulo de robustez va donde hay modelos flojos **por diseño**, que es el preset de evaluación — no "para el usuario final", que ya fue descartado como target. El modelo gratis no se borra: se reencuadra como **camino de evaluación del integrador**, que es hoy el único canal de adquisición.

**Compuerta:** medición, no opinión. `scripts/measure_rss.py` con `core` bajo objetivo; `demo` arranca y responde con modelo gratis; `core` arranca sin el módulo y no regresiona con un modelo bueno.

### M5 — Provisioning declarativo (depende de M4 y M1)

`lumen init --from provision.yaml` (modelo, proveedor, kit, canal, branding, workspace/teams). **Idempotente:** re-ejecutar converge, no duplica.

Instala kits y módulos en el arranque, por eso viaja sobre el contrato. Es el equivalente de `wp core install --url=… --admin_user=…`, el comando que hizo posible el negocio de las agencias.

**Compuerta:** desde un contenedor limpio, un comando produce una instancia vertical brandeada funcionando, verificada **dos veces** (idempotencia), sin abrir el browser.

### M6 — Documentación para integradores (depende de M5)

- "Cómo publicar un kit/módulo" como **portada**, no apéndice
- Referencia del contrato v1 + política de compatibilidad (escritas en M1, publicadas acá)
- Reencuadrar `README.md`: sacar la copy de consumer, mover *"no dev jargon"* (`README.md:69`) a la sección de UI cliente-facing, explicitar el modelo de dos superficies

**Compuerta — prueba de ácido:** un tercero escribe e instala un módulo contra v1 **leyendo solo la doc**. Si necesita abrir `brain.py`, la doc falló.

---

## 3. Criterios de auditoría

El trabajo de esta sesión será auditado hito por hito. **Las compuertas son los criterios** — no hay criterios blandos adicionales. Verificaciones que se van a correr:

| Hito | Verificación objetiva |
|---|---|
| M0 | Suite verde en `main` con `perfil-core`; conteo documentado; `git status` limpio; los 5 ítems de higiene aplicados |
| M1 | Tests de contrato existen y pasan contra stub; `grep lumen_api` da resultados en `module_manifest.py`; los módulos del catálogo siguen cargando; política escrita |
| M2 | `git diff` de los archivos de test en el commit de refactor debe ser **vacío**; los rangos 3049-3252 y 3288-3520 ya no están duplicados; suite verde con el conteo de M0 |
| M3 | Doble corrida (con y sin módulo) verde; `grep -rn "lumen.core" ` dentro del módulo solo toca el contrato; `brain: Any` ya no está expuesto |

Banderas rojas que van a hacer fallar la auditoría:

- Un test modificado en el commit de M2
- El módulo de M3 importando `brain` o cualquier interno del core
- Extension points agregados sin test de contrato
- Trabajo de M4/M5/M6 mezclado en la sesión
- Semántica de fallo "pass through" copiada al pipeline de respuesta

---

## 4. Decisiones y estado

### Cerradas

| Decisión | Valor |
|---|---|
| Posicionamiento | Integradores / agencias |
| Primer pilar | A — contrato de extensión versionado |
| Robustez de LLMs flojos | Módulo, no activo ni impuesto (contrapropuesta del owner) |
| Onboarding con modelo gratis | Se reencuadra como camino de **evaluación** del integrador, no modo por defecto |
| Corte de sesión | **M3** |

### Abiertas — resolver al inicio de M0

| # | Decisión | Default recomendado |
|---|---|---|
| 1 | `feat/perfil-core`: ¿merge o rebase? | Verificar la Fase 7 contra el código real (el checklist está stale) y **luego mergear**. No confiar en el checklist. |
| 2 | Higiene: ¿dentro de M0 o aparte? | **Dentro de M0, en commit separado.** No contamina la cadena y deja los diffs de M2 legibles. |

Si quien ejecuta no recibe confirmación explícita, **aplicar el default y dejarlo asentado en el reporte del hito**.

---

*Plan derivado de la auditoría arquitectónica del 2026-07-29 (Lumen vs. NousResearch/hermes-agent). Contexto completo en engram: `strategy/lumen-vs-hermes`, `strategy/lumen-extension-contract-v1`, `sdd/lumen-pilar-a/plan`.*
