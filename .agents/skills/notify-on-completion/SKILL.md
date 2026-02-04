---
name: notify-on-completion
description: Send local completion notifications to the Agent Notifications app via POST http://127.0.0.1:60766/agent/notify after finishing a task; include title, content, and agent details so the desktop app shows a system notification.
---

# Notify on Completion

## Endpoint

- POST `http://127.0.0.1:60766/agent/notify`
- Header: `Content-Type: application/json`
- Scope: Loopback only; requires the Agent Notifications app to be running and listening.

## Payload properties

- `title` (string, required): Concise heading for the notification title (e.g., "Build succeeded", "Tests failed").
- `content` (string, required): One to two sentences summarizing the outcome. Include key facts such as what finished, duration, artifact paths, or a brief error summary. Keep under ~950 characters because the server truncates the displayed body to 1000 characters after prefixing the agent.
- `agent` (string, required): Short identifier for the calling agent or workflow (e.g., "codex", "ci-run"). Avoid blanks and trailing spaces.

The server renders the notification body as `<agent>: <content>`.

## Construction rules

1. Trim all fields; do not send empty strings.
2. Make success/failure explicit in `title` and `content`.
3. Include the most useful signal in `content` (duration, counts, artifact hints, or top error line).
4. Stay within the soft 950-character limit to avoid truncation.
5. Build JSON safely (use `jq -n` or `printf`); escape quotes to avoid invalid payloads.

## cURL template

```bash
agent_name="codex"
title="Feature implemented"
content="The feature was implemented in ${elapsed}s; bundle at dist/."

curl --fail --silent --show-error \
  -X POST http://127.0.0.1:60766/agent/notify \
  -H "Content-Type: application/json" \
  -d "$(jq -n --arg title "$title" --arg content "$content" --arg agent "$agent_name" '{title:$title, content:$content, agent:$agent}')"
```

## When to send

- After long-running builds, tests, deploys, or data jobs finish.
- Before ending a session when the user may be away from the terminal.
- When a task fails; set a failure-focused `title` and summarize the error in `content`.
