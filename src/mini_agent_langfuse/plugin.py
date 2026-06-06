"""Langfuse plugin for mini-agent."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

from .session import (
    LangfuseSessionScope,
    is_langfuse_configured,
)
from .tracing import flush, record_turn_observation

if TYPE_CHECKING:
    from anthropic.types import MessageParam
    from mini_agent.cli.token import Usage

_DEBUG = True

def _log(msg: str) -> None:
    if _DEBUG:
        print(f"[langfuse] {msg}", file=sys.stderr)


class LangfusePlugin:
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


def _get_text_content(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return " ".join(parts) if parts else str(content)
    return str(content)


def _is_tool_only(content: str | list[dict[str, Any]]) -> bool:
    """Check if a message content only has tool blocks (no user/assistant text)."""
    return isinstance(content, list) and not any(
        isinstance(b, dict) and b.get("type") == "text" and b.get("text", "").strip()
        for b in content
    )


def _collect_blocks(
    content: str | list[dict[str, Any]], block_type: str
) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [b for b in content if isinstance(b, dict) and b.get("type") == block_type]


def _record_latest_turn(
    history: list[MessageParam],
    round_usages: list[Usage] | None,
) -> None:
    user_msg = None
    assistant_msg = None
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []

    # Walk backwards. Skip intermediate tool-only messages to find the
    # real user query and final assistant response.
    for msg in reversed(history):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and assistant_msg is None:
            text = _get_text_content(content)
            if text.strip():
                assistant_msg = text
                tool_calls = _collect_blocks(content, "tool_use")

        elif role == "user" and user_msg is None:
            # Skip messages that are only tool results (bash output, etc.)
            if _is_tool_only(content):
                tool_results = _collect_blocks(content, "tool_result")
                continue
            user_msg = _get_text_content(content)

        if user_msg is not None and assistant_msg is not None:
            break

    if user_msg is None:
        _log("  no user message found in last turn, skipping")
        return

    # Only count turns with actual text responses (skip tool_use-only rounds)
    turn_number = sum(
        1 for m in history
        if m.get("role") == "assistant" and _get_text_content(m.get("content", "")).strip()
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

    try:
        record_turn_observation(
            trace_name=f"agent-turn-{turn_number}",
            user_message=user_msg,
            assistant_message=assistant_msg,
            tool_calls=tool_calls if tool_calls else None,
            tool_results=tool_results if tool_results else None,
            model=model,
            **usage_kwargs,
        )
        _log(f"  turn {turn_number} recorded")
    except Exception as e:
        _log(f"  FAILED to record turn {turn_number}: {e}")
