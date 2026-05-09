# Honcho awareness

## What is Honcho

Honcho is a persistent cross-session memory service. It stores facts, learns from
interactions, builds peer models, and provides semantic search and dialectic reasoning
across all your conversations. Lumen connects to Honcho to remember what matters
beyond the current session.

## When to search

Use `honcho_search` (POST /honcho/search) before answering when:

- The user asks about something discussed in a past session
- The user mentions a previous interaction you need details about
- The question may have been answered or explored before
- You need context from earlier conversations to give a good answer

Use `honcho_context` (GET /honcho/context) when:

- You start a new session and want to inject prior context into your thinking
- The session hasn't been started yet but you need historical knowledge

## How to use honcho_search

Make a POST request to `/honcho/search` with:

```json
{
  "query": "your search terms here",
  "max_tokens": 800
}
```

The response returns relevance-ranked excerpts:

```json
{
  "result": "the main matching content",
  "sessions": [{"session_id": "...", "title": "..."}, ...]
}
```

Interpret results by looking for relevance to the user's current question.
If no results match, the memory may not contain relevant data — proceed without it.

## How to use honcho_conclude

After completing a task or reaching a conclusion, make a POST request to
`/honcho/conclude` to persist what you learned:

```json
{
  "content": "El usuario prefiere respuestas en markdown con código bloqueado",
  "peer": "user"
}
```

The `content` field is truncated to 25000 characters automatically — do not
worry about truncation, just write your conclusion naturally.

Use `honcho_memory` (POST /honcho/memory) for storing structured facts:

```json
{
  "content": "User prefers Python for backend work",
  "session": "sess_abc123",
  "type": "fact"
}
```

## Context injection

Before beginning a new session, call `GET /honcho/context?session_id=<id>` to
retrieve accumulated context. The response includes:

- **context**: A formatted context block for injection
- **summary**: High-level summary of past interactions
- **card**: Peer card with identity information
- **representation**: The model Honcho has built of this peer
- **recent**: Recent memory entries

Inject the `context` field into your session by passing it to
`brain.set_context()` or equivalent — this makes past knowledge part of your
current reasoning.

## Memory retention best practices

### What to store

- User preferences and stated habits
- Technical decisions and their rationale
- Project structure and key files
- Recurring problems and their solutions
- Personal facts (names, roles, workflows)
- Learned patterns about user communication style
- Decisions made during conversation that shaped outcomes

### What to skip

- Temporary noise or throwaway information
- Redundant facts already implied by other memories
- Raw conversation logs (store summaries, not transcripts)
- Information that is already in the session context
- Credentials or sensitive data beyond what is necessary

### Summary vs detail tradeoffs

- Write concise summaries of outcomes instead of full transcripts
- Store the "what" and "why" but skip the "how" of intermediate steps
- Prefer high-signal facts over low-signal observations
- If in doubt, write a brief conclusion — it's cheap to store and easy to forget

## Error awareness

Honcho may be unavailable due to network issues, rate limits, or service outages.

If you receive a **503 Service Unavailable** or **504 Gateway Timeout**:

1. Continue the task without Honcho memory — it was never strictly required
2. Log the failure but do not fail the user's request
3. Retry the operation on the next interaction if the information is critical
4. Inform the user only if the lack of memory affects the quality of the answer

Honcho is an augmentation, not a dependency. Lumen operates normally without it.
