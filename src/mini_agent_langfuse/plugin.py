"""Langfuse plugin for mini-agent."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any

from mini_agent.plugin import MiniAgentPlugin

from .session import (
    LangfuseSessionScope,
    is_langfuse_configured,
)
from .tracing import flush, record_turn_observation

if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from mini_agent.cli.token import Usage

_DEBUG = os.getenv("LANGFUSE_DEBUG", "1") not in ("0", "false", "False", "")

def _log(msg: str) -> None:
    if _DEBUG:
        print(f"[langfuse] {msg}", file=sys.stderr)


class LangfusePlugin(MiniAgentPlugin):
    """Hooks into mini-agent lifecycle to trace sessions to Langfuse."""

    def __init__(self) -> None:
        self._scope: LangfuseSessionScope | None = None
        self._enabled = False

    def on_agent_init(self) -> None:
        self._enabled = is_langfuse_configured()
        if self._enabled:
            _log("Langfuse credentials found, plugin active")
            try:
                from langfuse import get_client
                get_client()
                _log("Langfuse client initialized")
            except Exception as e:
                _log(f"Failed to init Langfuse client: {e}")
                self._enabled = False
        else:
            _log("No Langfuse credentials (set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)")

    def on_session_start(self, session_id: str) -> None:
        if not self._enabled:
            return
        _log(f"session start: {session_id[:12]}...")
        self._close_scope()
        model = _get_current_model()
        _log(f"  model={model}")
        scope = LangfuseSessionScope(session_id=session_id, model=model)
        try:
            scope.__enter__()
            self._scope = scope
            _log("  propagate_attributes scope opened")
            print(file=sys.stderr)
        except Exception as e:
            _log(f"  FAILED to open scope: {e}")

    def on_turn_complete(
        self,
        session_id: str,
        history: list[MessageParam],
        round_usages: list[Usage] | None,
    ) -> None:
        if not self._enabled or not history:
            return
        try:
            _record_latest_turn(history, round_usages)
        except Exception as e:
            _log(f"turn record failed: {e}")

    def on_session_end(
        self,
        session_id: str,
        history: list[MessageParam],
        round_usages: list[Usage] | None,
    ) -> None:
        if not self._enabled:
            return
        _log(f"session end: {session_id[:12]}...")
        self._close_scope()
        try:
            flush()
            _log("flushed events to Langfuse")
        except Exception as e:
            _log(f"flush failed: {e}")

    def _close_scope(self) -> None:
        if self._scope is not None:
            import contextlib
            with contextlib.suppress(Exception):
                self._scope.__exit__(None, None, None)
            self._scope = None


def create_plugin() -> LangfusePlugin:
    """Entry point factory for mini_agent.plugins."""
    return LangfusePlugin()


def _get_current_model() -> str | None:
    try:
        from mini_agent.config import config
        return config.get_model()
    except Exception:
        return None


def _block_type(block: Any) -> str:
    """Get the type from either a dict or an Anthropic SDK object."""
    if isinstance(block, dict):
        return block.get("type", "")
    return str(getattr(block, "type", ""))


def _block_text(block: Any) -> str:
    """Get text from a dict or Anthropic SDK object (ParsedTextBlock, ThinkingBlock)."""
    if isinstance(block, dict):
        return str(block.get("text", block.get("thinking", "")))
    return str(getattr(block, "text", getattr(block, "thinking", "")))


def _get_text_content(content: str | list[Any]) -> str:
    """Extract all text and thinking content from a message."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            _block_text(b) for b in content
            if _block_type(b) in ("text", "thinking")
        ]
        return " ".join(parts) if parts else ""
    return str(content)


def _is_tool_only(content: str | list[Any]) -> bool:
    """True if the message only has tool-related blocks (no text/thinking)."""
    return isinstance(content, list) and not any(
        _block_type(b) in ("text", "thinking") and _block_text(b).strip()
        for b in content
    )


def _collect_blocks(content: str | list[Any], block_type: str) -> list[Any]:
    if not isinstance(content, list):
        return []
    return [b for b in content if _block_type(b) == block_type]


def _record_latest_turn(
    history: list[MessageParam],
    round_usages: list[Usage] | None,
) -> None:
    # Find the last real user query (skip tool_result-only messages).
    user_idx = None
    for i in range(len(history) - 1, -1, -1):
        if history[i].get("role") == "user" and not _is_tool_only(
            history[i].get("content", "")
        ):
            user_idx = i
            break

    if user_idx is None:
        _log("  no user message found in last turn, skipping")
        return

    user_msg = _get_text_content(history[user_idx].get("content", ""))

    # Find the last assistant response after the user query.
    assistant_idx = None
    for i in range(len(history) - 1, user_idx, -1):
        if history[i].get("role") == "assistant":
            assistant_idx = i
            break

    assistant_msg = None
    tool_calls = []
    tool_results = []

    if assistant_idx is not None:
        # Collect the assistant message text from the last assistant response.
        content = history[assistant_idx].get("content", "")
        text = _get_text_content(content)
        tool_calls_in_last = _collect_blocks(content, "tool_use")
        if text.strip():
            assistant_msg = text
        elif tool_calls_in_last:
            names = [_block_text(tc) or _block_type(tc) for tc in tool_calls_in_last[:3]]
            assistant_msg = f"[tool_calls: {', '.join(names)}]"

    # Scan ALL messages from user_idx+1 to the end for tool_use and tool_result
    # blocks — they may be in earlier assistant/user messages within this turn.
    all_tool_calls = []
    all_tool_results = []
    for i in range(user_idx + 1, len(history)):
        msg = history[i]
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            all_tool_calls.extend(_collect_blocks(content, "tool_use"))
        elif role == "user":
            all_tool_results.extend(_collect_blocks(content, "tool_result"))

    tool_calls = all_tool_calls or None
    tool_results = all_tool_results or None

    # Count turns by real user messages (not tool results).
    turn_number = sum(
        1 for m in history
        if m.get("role") == "user" and not _is_tool_only(m.get("content", ""))
    )

    usage_kwargs: dict[str, int] = {}
    if round_usages and len(round_usages) > 0:
        last = round_usages[-1]
        for attr, kw in [
            ("input_tokens", "input_tokens"),
            ("output_tokens", "output_tokens"),
            ("cache_creation_input_tokens", "cache_creation_input_tokens"),
            ("cache_read_input_tokens", "cache_read_input_tokens"),
        ]:
            val = getattr(last, attr, 0) or 0
            if val:
                usage_kwargs[kw] = val

    model = _get_current_model()

    _log(f"  recording turn {turn_number}: user_msg={user_msg[:60]!r}...")
    if usage_kwargs:
        _log(f"  usage: {usage_kwargs}")
    if tool_calls:
        _log(f"  tool_calls: {len(tool_calls)} tools")

    try:
        record_turn_observation(
            trace_name=f"agent-turn-{turn_number}",
            user_message=user_msg,
            assistant_message=assistant_msg,
            tool_calls=tool_calls,
            tool_results=tool_results,
            model=model,
            **usage_kwargs,
        )
        _log(f"  turn {turn_number} recorded")
        print(file=sys.stderr)
    except Exception as e:
        _log(f"  FAILED to record turn {turn_number}: {e}")
