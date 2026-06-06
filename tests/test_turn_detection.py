"""Verify _record_latest_turn logic with realistic history scenarios.

Run: uv run pytest tests/test_turn_detection.py -v
"""

import json
from unittest import mock

from mini_agent_langfuse.plugin import (
    _collect_blocks,
    _get_text_content,
    _is_tool_only,
    _record_latest_turn,
)


# ── helpers ──────────────────────────────────────────────────────────

def user(text: str) -> dict:
    return {"role": "user", "content": text}


def user_tool_result(output: str) -> dict:
    return {"role": "user", "content": [{"type": "tool_result", "content": output}]}


def asst(text: str) -> dict:
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def asst_thinking(text: str, think: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": think},
            {"type": "text", "text": text},
        ],
    }


def asst_tool(name: str, **kwargs) -> dict:
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "name": name, "input": kwargs}],
    }


def asst_text_and_tool(text: str, name: str, **kwargs) -> dict:
    return {
        "role": "assistant",
        "content": [
            {"type": "text", "text": text},
            {"type": "tool_use", "name": name, "input": kwargs},
        ],
    }


class Capture:
    """Captures the last call to record_turn_observation."""
    def __init__(self):
        self.calls = []
    def __call__(self, **kw):
        self.calls.append(kw)


# ── tests ─────────────────────────────────────────────────────────────

def test_tool_call_turn():
    """Turn with tool_use in assistant (tool is recorded off multi-round history).

    History:
        user: "list files"
        asst: tool_use(bash)          ← tool_use lives here
        user: tool_result(bash output)
        asst: "Here are the files..." ← algorithm finds assistant_msg here
    """
    history = [
        user("list files"),
        asst_tool("bash", command="ls -la"),
        user_tool_result("total 164\ndrwxr-xr-x ..."),
        asst("Here are the files."),
    ]

    cap = Capture()
    with mock.patch(
        "mini_agent_langfuse.plugin.record_turn_observation", cap
    ):
        _record_latest_turn(history, None)

    assert len(cap.calls) == 1, f"Expected 1 recording, got {len(cap.calls)}"
    call = cap.calls[0]

    assert call["user_message"] == "list files", (
        f"Expected 'list files', got {call['user_message']!r}"
    )
    assert call["assistant_message"] == "Here are the files.", (
        f"Expected final text, got {call['assistant_message']!r}"
    )
    assert len(call.get("tool_calls", [])) == 1, (
        f"Expected 1 tool_call, got {len(call.get('tool_calls', []))}"
    )
    if call.get("tool_calls"):
        assert call["tool_calls"][0]["name"] == "bash"


def test_simple_turn_no_tools():
    """Plain Q&A — no tools involved."""
    history = [
        user("What is Python?"),
        asst("Python is a language."),
    ]

    cap = Capture()
    with mock.patch(
        "mini_agent_langfuse.plugin.record_turn_observation", cap
    ):
        _record_latest_turn(history, None)

    assert len(cap.calls) == 1
    assert cap.calls[0]["user_message"] == "What is Python?"
    assert cap.calls[0]["assistant_message"] == "Python is a language."
    assert cap.calls[0].get("tool_calls") is None


def test_tool_call_with_thinking():
    """Turn with thinking + tool_use + tool result + final text."""
    history = [
        user("check git status"),
        asst_thinking("I'll run git status", "User wants repo status"),
        asst_tool("bash", command="git status"),
        user_tool_result("On branch main"),
        asst("The repo is on main branch."),
    ]

    cap = Capture()
    with mock.patch(
        "mini_agent_langfuse.plugin.record_turn_observation", cap
    ):
        _record_latest_turn(history, None)

    assert len(cap.calls) == 1
    assert cap.calls[0]["user_message"] == "check git status"
    assert "The repo is on main" in cap.calls[0]["assistant_message"]
    assert len(cap.calls[0].get("tool_calls", [])) == 1


def test_multi_tool_turn():
    """Turn with multiple tool calls in a single round."""
    history = [
        user("deploy"),
        asst_text_and_tool("Checking status...", "bash", command="git status"),
        user_tool_result("clean"),
        asst_text_and_tool("Building...", "bash", command="npm run build"),
        user_tool_result("build success"),
        asst("Deployed!"),
    ]

    cap = Capture()
    with mock.patch(
        "mini_agent_langfuse.plugin.record_turn_observation", cap
    ):
        _record_latest_turn(history, None)

    assert len(cap.calls) == 1
    assert cap.calls[0]["user_message"] == "deploy"
    # Should have collected tool_calls from ALL intermediate assistants
    assert len(cap.calls[0].get("tool_calls", [])) == 2, (
        f"Expected 2 tool_calls, got {len(cap.calls[0].get('tool_calls', []))}"
    )


def test_tool_only_assistant():
    """Assistant message that has ONLY tool_use (no text)."""
    history = [
        user("run ls"),
        asst_tool("bash", command="ls"),
        user_tool_result("file1\nfile2"),
        asst("Done."),
    ]

    cap = Capture()
    with mock.patch(
        "mini_agent_langfuse.plugin.record_turn_observation", cap
    ):
        _record_latest_turn(history, None)

    assert len(cap.calls) == 1, f"Expected 1, got {len(cap.calls)}"
    assert cap.calls[0]["user_message"] == "run ls"
    # tool_calls should include the bash tool even though the text
    # response ("Done.") has no tool_use in it
    assert len(cap.calls[0].get("tool_calls", [])) == 1, (
        f"Expected 1 tool_call, got {len(cap.calls[0].get('tool_calls', []))}"
    )


# ── unit tests for helpers ─────────────────────────────────────────

def test_is_tool_only():
    assert _is_tool_only(user_tool_result("stuff")["content"])
    assert not _is_tool_only(user("hello")["content"])
    assert not _is_tool_only(asst("hello")["content"])


def test_get_text_content():
    assert _get_text_content("hello") == "hello"
    assert _get_text_content(asst("hi")["content"]) == "hi"
    text = _get_text_content(asst_thinking("answer", "think")["content"])
    assert "answer" in text
    assert "think" in text
    assert _get_text_content(asst_tool("bash")["content"]) == ""


def test_collect_blocks():
    content = asst_text_and_tool("hi", "bash", command="ls")["content"]
    tools = _collect_blocks(content, "tool_use")
    assert len(tools) == 1
    assert tools[0]["name"] == "bash"
    assert tools[0]["input"]["command"] == "ls"
