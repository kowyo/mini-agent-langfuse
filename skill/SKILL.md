---
name: langfuse
description: |
  Langfuse observability integration for mini-agent. Traces sessions, records turns, tracks token usage. Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.
---

# Langfuse Plugin for mini-agent

## Setup

```bash
pip install mini-agent-langfuse
```

Add to `~/.mini-agent/.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

Run `mini` — the plugin loads automatically.

## Architecture

The plugin implements `MiniAgentPlugin` and hooks into the agent lifecycle:

| Hook | Action |
|---|---|
| `on_agent_init()` | Validates credentials |
| `on_session_start(id)` | Opens `propagate_attributes(session_id=...)` scope |
| `on_turn_complete(id, history, usage)` | Records `generation` observation |
| `on_session_end(id, history, usage)` | Closes scope, calls `flush()` |

Session IDs are mapped 1:1 (mini-agent UUID → Langfuse session_id). Each turn records model, token usage, input, and output.

## Troubleshooting

- **No traces**: Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set.
- **Plugin not loading**: Run `mini --plugins` to verify discovery.
