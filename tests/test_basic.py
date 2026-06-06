import os
from unittest import mock

from mini_agent_langfuse.session import (
    LANGFUSE_ENV_PUBLIC_KEY,
    LANGFUSE_ENV_SECRET_KEY,
    build_session_metadata,
    is_langfuse_configured,
)


def test_is_langfuse_configured():
    with mock.patch.dict(os.environ, {}, clear=True):
        assert is_langfuse_configured() is False
    with mock.patch.dict(
        os.environ,
        {LANGFUSE_ENV_PUBLIC_KEY: "pk-lf-test", LANGFUSE_ENV_SECRET_KEY: "sk-lf-test"},
        clear=True,
    ):
        assert is_langfuse_configured() is True
    with mock.patch.dict(os.environ, {LANGFUSE_ENV_PUBLIC_KEY: "pk-lf-test"}, clear=True):
        assert is_langfuse_configured() is False


def test_build_session_metadata():
    m = build_session_metadata(model="claude-sonnet-4-6", cwd="/home/user", tags=["test"])
    assert m["source"] == "mini-agent"
    assert m["model"] == "claude-sonnet-4-6"
    assert m["cwd"] == "/home/user"
    assert m["tags"] == "test"
    assert "started_at" in m


def test_build_session_metadata_minimal():
    m = build_session_metadata()
    assert m["source"] == "mini-agent"
    assert "model" not in m


def test_plugin_entry_point():
    from mini_agent_langfuse.plugin import create_plugin, LangfusePlugin
    assert isinstance(create_plugin(), LangfusePlugin)


def test_plugin_lifecycle():
    from mini_agent_langfuse.plugin import LangfusePlugin
    p = LangfusePlugin()
    p.on_agent_init()
    p.on_session_start("test-id")
    p.on_turn_complete("test-id", [], [])
    p.on_session_end("test-id", [], [])
