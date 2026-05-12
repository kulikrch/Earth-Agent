"""Utilities for marking a subset of MCP tools as 'development' tools.

We cannot attach arbitrary metadata to FastMCP tools directly, but we can embed a stable tag
into the tool description. Downstream (LangChain runners) can filter tools by this tag.

Tag format: "[DEV_TOOL]"
"""

from __future__ import annotations

from typing import Callable, Any

DEV_TOOL_TAG = "[DEV_TOOL]"


def dev_mcp_tool(mcp: Any, description: str, *, tag: str = DEV_TOOL_TAG, **kwargs: Any) -> Callable:
    """A thin wrapper around `mcp.tool` that prefixes description with a stable dev tag."""

    # Keep original description for humans, but make tag machine-detectable.
    marked_description = f"{tag}\n{description.strip()}"
    return mcp.tool(description=marked_description, **kwargs)
