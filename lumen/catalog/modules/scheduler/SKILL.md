---
name: scheduler
description: "Schedule reminders and recurring tasks that fire automatically. Use scheduler__create to save them."
min_capability: tier-2
provides: [scheduler__create, scheduler__list, scheduler__cancel]
requires:
  connectors: [memory]
---
# Scheduler

You can schedule reminders that are DELIVERED AUTOMATICALLY at the right time
through the conversation channel. A background loop fires them — you do not
need to do anything at delivery time.

## How to handle reminder requests

When the user says things like:
- "Remind me to call Maria at 3pm"
- "Every Monday, remind me to check reports"
- "In 30 minutes, tell me to take a break"

Do the following:
1. Extract WHAT to remind (keep the user's words)
2. Resolve WHEN to an absolute local date-time 'YYYY-MM-DD HH:MM'
   (compute relative times like "in 30 minutes" yourself)
3. Call scheduler__create with text + when (+ recurrence for recurring:
   'daily' or 'weekly:mon'..'weekly:sun')
4. Confirm back to the user with the exact date and time

## Managing reminders

- "What reminders do I have?" → scheduler__list
- "Cancel the X reminder" → scheduler__list to find the job id, then scheduler__cancel

## Rules

- NEVER claim a reminder is set without calling scheduler__create and getting
  back a job id. If the call returns an error, tell the user honestly.
- Always confirm times in the user's timezone.
