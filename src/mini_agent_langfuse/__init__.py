"""Langfuse session tracing plugin for mini-agent.

Usage::

    uv tool install mini-agent --with mini-agent-langfuse
    # Add to ~/.mini-agent/.env:
    #   LANGFUSE_PUBLIC_KEY=pk-lf-...
    #   LANGFUSE_SECRET_KEY=sk-lf-...
    mini
"""

from .plugin import LangfusePlugin

__all__ = ["LangfusePlugin"]
