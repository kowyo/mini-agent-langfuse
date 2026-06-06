# mini-agent-langfuse

Langfuse session tracing plugin for [mini-agent](https://github.com/kowyo/mini-agent).

## Install

```bash
uv tool install mini-agent --with mini-agent-langfuse
```

Or for local development:

```bash
git clone https://github.com/kowyo/mini-agent-langfuse
cd mini-agent-langfuse
uv sync
```

## Setup

```bash
# ~/.mini-agent/.env (loaded automatically)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
```

## Usage

Run `mini`. The plugin is auto-discovered. Each session and turn is traced to Langfuse.

```bash
mini
> /plugins
# Active plugins (1):
#   - LangfusePlugin
```

## What gets traced

| Event | Langfuse entity |
|---|---|
| Session start | `propagate_attributes(session_id=...)` scope |
| Each turn | `generation` observation (input, output, tokens, model) |
| Session end | `flush()` |

## Plugin entry point

Registered under `mini_agent.plugins` in `pyproject.toml`:

```toml
[project.entry-points."mini_agent.plugins"]
langfuse = "mini_agent_langfuse.plugin:create_plugin"
```
