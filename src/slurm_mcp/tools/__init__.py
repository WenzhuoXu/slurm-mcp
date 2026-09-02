"""MCP tool groups (design section 4). ``register_all(mcp, service)`` wires every group module that exists.

Each group module exposes ``register(mcp, service)`` and defines ``async`` tool functions with the exact names,
arguments and defaults of section 4, ``_mcp`` annotations, structured Pydantic results and descriptions under
2 KB. Groups built by other slices (``jobs``, ``events``, ``transfers``, ``placement``, ``alloc``) are imported
when present so parallel builders never touch this file. ``run_tool`` converts ``SlurmMcpError`` (and any other
exception) into the ``ToolError`` text of section 9.1.
"""
from __future__ import annotations

import importlib
import logging
from collections.abc import Awaitable
from typing import Any, TypeVar

from .._mcp import MCPServer, ToolError
from ..errors import SlurmMcpError, to_tool_error

log = logging.getLogger("slurm_mcp.tools")

T = TypeVar("T")
CORE_GROUPS: tuple[str, ...] = ("clusters", "files", "config")
OPTIONAL_GROUPS: tuple[str, ...] = ("jobs", "events", "transfers", "placement", "alloc")
BIG_RESULT_META: dict[str, Any] = {"anthropic/maxResultSizeChars": 60000}
MAX_DESCRIPTION_CHARS = 2048


async def run_tool(aw: Awaitable[T]) -> T:
    """Await a Service call; ``SlurmMcpError`` -> ``ToolError(str(e))``; anything else -> ``E_SSH``-style text."""
    try:
        return await aw
    except ToolError:
        raise
    except SlurmMcpError as e:
        raise ToolError(str(e)) from e
    except Exception as e:  # never leak a raw traceback through a masked UnexpectedToolError
        log.exception("tool failed: %s", e)
        raise to_tool_error(e) from e


def register_all(mcp: MCPServer, service: Any) -> list[str]:
    """Register the core groups and every optional group that imports cleanly; returns the group names."""
    loaded: list[str] = []
    for name in CORE_GROUPS + OPTIONAL_GROUPS:
        try:
            module = importlib.import_module(f"{__name__}.{name}")
        except ImportError as e:
            if name in CORE_GROUPS:
                raise
            if getattr(e, "name", None) not in (f"{__name__}.{name}", None) and name not in str(e):
                log.warning("tool group %s not loaded (import error: %s)", name, e)
            continue
        register = getattr(module, "register", None)
        if not callable(register):
            log.warning("tool group %s has no register(mcp, service)", name)
            continue
        register(mcp, service)
        loaded.append(name)
    return loaded


__all__ = ["run_tool", "register_all", "CORE_GROUPS", "OPTIONAL_GROUPS", "BIG_RESULT_META", "MAX_DESCRIPTION_CHARS"]
