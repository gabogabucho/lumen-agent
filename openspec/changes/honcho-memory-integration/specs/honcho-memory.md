# Honcho Memory Specification

## Purpose

This specification defines the requirements for integrating Honcho (honcho.dev) as Lumen's cross-session persistent memory layer. It covers module installation, environment configuration, memory initialization, read/write sync, API endpoints, awareness skill, error handling, and multi-mode support.

## Requirements

### Requirement: Module Installation & Registration

Lumen MUST provide a `lumen module install honcho` command that scaffolds the honcho module directory at `lumen/catalog/modules/honcho/`, installs required dependencies, and registers the module in the catalog index.

#### Scenario: Successful module installation

- GIVEN Honcho module is defined in the Lumen catalog
- WHEN a user runs `lumen module install honcho`
- THEN the directory `lumen/catalog/modules/honcho/` is created
- AND the module is registered in the catalog index
- AND `honcho-ai` dependency is installed

#### Scenario: Module appears in installed list

- GIVEN The honcho module has been successfully installed
- WHEN a user queries `GET /api/modules/installed`
- THEN The honcho module appears in the list with its provides and requires fields

### Requirement: Environment Configuration

Lumen MUST accept five honcho environment variables: `HONCHO_WORKSPACE_ID`, `HONCHO_API_KEY`, `HONCHO_BASE_URL`, `HONCHO_SESSION_STRATEGY`, `HONCHO_RECALL_MODE`. The API key SHALL be stored as a secret; others as plain config. On install, the system MUST validate all values against their constraints.

#### Scenario: API key stored as secret

- GIVEN Honcho module is being installed with `HONCHO_API_KEY=xxx`
- THEN the API key is stored in the secrets store (`_config["secrets"]["honcho"]["api_key"]`)
- AND plain config values are stored under `_config["honcho"]`

#### Scenario: Invalid workspace_id rejected

- GIVEN A user attempts to install with `HONCHO_WORKSPACE_ID=""`
- WHEN configuration validation runs
- THEN the install fails with error: workspace_id must be a non-empty string

#### Scenario: Invalid session_strategy rejected

- GIVEN A user attempts to install with `HONCHO_SESSION_STRATEGY=invalid`
- WHEN configuration validation runs
- THEN the install fails with error: session_strategy must be one of [per-session, per-directory, per-repo, global]

#### Scenario: Invalid recall_mode rejected

- GIVEN A user attempts to install with `HONCHO_RECALL_MODE=invalid`
- WHEN configuration validation runs
- THEN the install fails with error: recall_mode must be one of [hybrid, context, tools]

### Requirement: Memory Initialization

On install, Lumen MUST register itself as an AI peer with Honcho using the provided workspace_id, store the resulting Peer ID in module config, and validate the connection with a test API call.

#### Scenario: Peer registration succeeds

- GIVEN workspace_id and api_key are valid
- WHEN `on_install.py` runs
- THEN Lumen registers as an AI peer with Honcho
- AND the Peer ID is stored in module config
- AND a test call (`list_sessions`) confirms connectivity

#### Scenario: Connection failure reported

- GIVEN credentials are valid but the network is unreachable
- WHEN `on_install.py` runs
- THEN the install fails and reports the error reason (connection refused, timeout, etc.)

### Requirement: Memory Sync — Read (Context Injection)

Before each Lumen session, the system MUST fetch context from Honcho using the session key. Context includes session summary, peer representation, and recent facts. Injected context SHALL be passed to the brain as available context.

#### Scenario: Context fetched for session

- GIVEN The honcho module is configured and connected
- WHEN a Lumen session begins
- THEN context is fetched from Honcho using the session key
- AND the context (summary, card, representation, recent messages) is injected into the session

#### Scenario: Empty session gets fresh context

- GIVEN A brand new session with no prior data in Honcho
- WHEN context is fetched
- THEN an empty or minimal context block is returned
- AND the session proceeds without blocking

### Requirement: Memory Sync — Write (Conclusions)

At end of each Lumen session, the system MUST write relevant conclusions to Honcho. Conclusions are extracted from the session summary, user interactions, and tool usage. Write frequency is controlled by config (async / per-turn / per-session). Failed writes SHALL be logged but SHALL NOT fail the session.

#### Scenario: Conclusions written at session end

- GIVEN A session has completed with user interactions
- WHEN the session concludes
- THEN relevant conclusions are written to Honcho based on `recalls_mode` setting
- AND the write is logged for observability

#### Scenario: Failed write does not break session

- GIVEN Honcho API is unreachable
- WHEN session conclusions are being written
- THEN the failure is logged
- AND the session completes normally without error to the user

### Requirement: Search API Endpoint

The system MUST expose `POST /honcho/search` for semantic search against Honcho memory. The endpoint requires Bearer token auth and module installation guard.

#### Scenario: Valid search request

- GIVEN a valid Bearer token and the honcho module installed
- AND a search query is provided: `{"query": "mi proyecto de Python", "max_tokens": 800}`
- WHEN the request is processed
- THEN the response returns `{ "result": "...", "sessions": [...] }` with relevance-ranked excerpts

#### Scenario: Honcho API failure

- GIVEN a valid request to `/honcho/search`
- AND the Honcho API is unreachable
- THEN the endpoint returns HTTP 503 with a human-readable error message

#### Scenario: Unauthenticated request

- GIVEN no Bearer token is provided
- WHEN `POST /honcho/search` is called
- THEN the endpoint returns HTTP 401 (Unauthorized)

### Requirement: Context API Endpoint

The system MUST expose `GET /honcho/context` for retrieving full context blocks from Honcho. The endpoint requires Bearer token auth and module installation guard.

#### Scenario: Context retrieval succeeds

- GIVEN a valid Bearer token and the honcho module installed
- AND the query params `session_id=sess123&peer=user` are provided
- WHEN `GET /honcho/context` is called
- THEN the response returns the full context block: summary, card, representation, recent messages

#### Scenario: Session not found

- GIVEN a valid request with a non-existent session_id
- WHEN `GET /honcho/context` is called
- THEN the endpoint returns an empty context block

### Requirement: Conclude API Endpoint

The system MUST expose `POST /honcho/conclude` for writing conclusions to Honcho. The endpoint requires Bearer token auth and module installation guard.

#### Scenario: Conclusion written successfully

- GIVEN a valid Bearer token and the honcho module installed
- AND the request body is `{"content": "El usuario prefiere respuestas en markdown", "peer": "user"}`
- WHEN `POST /honcho/conclude` is called
- THEN the response returns `{ "success": true }`

#### Scenario: Conclusion too long

- GIVEN a request with a content field exceeding 25000 characters
- WHEN the endpoint processes the request
- THEN the content is truncated to 25000 characters gracefully
- AND the write proceeds with the truncated content

#### Scenario: Write failure returns error

- GIVEN the Honcho API is unreachable
- WHEN `POST /honcho/conclude` is called
- THEN the response returns `{ "success": false, "error": "<message>" }`

### Requirement: Memory Store API Endpoint

The system MUST expose `POST /honcho/memory` for storing arbitrary memory entries in Honcho. The endpoint requires Bearer token auth and module installation guard.

#### Scenario: Memory entry created successfully

- GIVEN a valid Bearer token and the honcho module installed
- AND the request body is `{"session": "sess123", "content": "Prefiero Python para backends", "type": "fact"}`
- WHEN `POST /honcho/memory` is called
- THEN the response returns `{ "id": "...", "success": true }`

#### Scenario: Memory entry without session

- GIVEN the request body omits the optional `session` field
- WHEN the endpoint processes the request
- THEN the memory is stored without an associated session

#### Scenario: Memory endpoint timeout

- GIVEN the Honcho API takes longer than 30 seconds to respond
- WHEN `POST /honcho/memory` is in progress
- THEN the endpoint returns HTTP 504 (Gateway Timeout)

### Requirement: Awareness Skill

The system MUST create `skills/honcho-awareness/SKILL.md` that provides Lumen AI with instructions for using Honcho-memory features.

#### Scenario: Skill contains search instructions

- GIVEN the honcho-awareness skill is installed
- WHEN Lumen AI reads the SKILL.md
- THEN it contains instructions on how to use `honcho_search`

#### Scenario: Skill contains conclude instructions

- GIVEN the honcho-awareness skill is installed
- WHEN Lumen AI reads the SKILL.md
- THEN it contains instructions on how to use `honcho_conclude`

#### Scenario: Skill contains context accessibility instructions

- GIVEN the honcho-awareness skill is installed
- THEN it instructs the AI on how to access Honcho context

#### Scenario: Skill contains memory retention best practices

- GIVEN the honcho-awareness skill is installed
- THEN it includes guidance on memory retention best practices
- AND includes Honcho-specific terminology and capabilities

### Requirement: Error Handling & Resilience

The system MUST enforce timeout, auth, and availability error handling across all Honcho endpoints.

#### Scenario: Service unavailable

- GIVEN any Honcho endpoint is called
- AND the Honcho service is unreachable
- THEN the endpoint returns HTTP 503 (Service Unavailable)

#### Scenario: Invalid credentials

- GIVEN a request with an invalid Bearer token
- THEN the endpoint returns HTTP 401 (Unauthorized)

#### Scenario: Timeout handling

- GIVEN a request to a Honcho endpoint
- AND the Honcho API does not respond within 30 seconds
- THEN the endpoint returns HTTP 504 (Gateway Timeout)

#### Scenario: Offline graceful degradation

- GIVEN the Honcho service is permanently unavailable
- THEN Lumen continues to operate normally
- AND the module reports available but no memory sync occurs

#### Scenario: Bilingual error messages

- GIVEN an error occurs on any Honcho endpoint
- THEN the error response includes a human-readable message in both Spanish and English

### Requirement: Multi-mode Support

The system MUST support both cloud mode (automatic connection to honcho.dev) and self-hosted mode (connection to HONCHO_BASE_URL) with automatic mode selection.

#### Scenario: Cloud mode auto-selected

- GIVEN `HONCHO_API_KEY` is set but `HONCHO_BASE_URL` is not set
- WHEN the module initializes
- THEN the client connects to honcho.dev (default cloud URL)
- AND the API key is used for authentication

#### Scenario: Self-hosted mode auto-selected

- GIVEN both `HONCHO_API_KEY` and `HONCHO_BASE_URL=https://honcho.internal` are set
- WHEN the module initializes
- THEN the client connects to https://honcho.internal instead of honcho.dev
- AND the mode is detected automatically based on whether HONCHO_BASE_URL is configured

#### Scenario: Self-hosted without auth

- GIVEN `HONCHO_BASE_URL` is set to a local URL with no API key configured
- THEN the client connects to the local instance without requiring authentication
