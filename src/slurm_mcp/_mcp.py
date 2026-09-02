"""Single import surface for the MCP SDK.

We target mcp 2.x (MCPServer). If the project ever needs to move back to the 1.x maintenance line
(FastMCP), this is the only module that changes. Everything else imports from here.
"""
from __future__ import annotations

try:  # mcp >= 2.0
    from mcp.server.mcpserver import Context, MCPServer  # type: ignore
    SDK_GENERATION = 2
except ModuleNotFoundError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import Context, FastMCP as MCPServer  # type: ignore
    SDK_GENERATION = 1

from mcp.types import ToolAnnotations  # both generations

try:  # mcp >= 2.0
    from mcp.server.mcpserver.exceptions import ToolError  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp.exceptions import ToolError  # type: ignore


def read_only() -> ToolAnnotations:
    return _ann(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def mutating() -> ToolAnnotations:
    """Creates or changes remote state but is not destructive (submit, upload)."""
    return _ann(read_only_hint=False, destructive_hint=False, idempotent_hint=False)


def destructive() -> ToolAnnotations:
    """Cancels jobs, deletes files, releases allocations."""
    return _ann(read_only_hint=False, destructive_hint=True, idempotent_hint=False)


def _ann(**kw) -> ToolAnnotations:
    kw.setdefault("open_world_hint", True)
    if SDK_GENERATION == 1:  # 1.x uses camelCase field names
        camel = {"read_only_hint": "readOnlyHint", "destructive_hint": "destructiveHint",
                 "idempotent_hint": "idempotentHint", "open_world_hint": "openWorldHint"}
        kw = {camel.get(k, k): v for k, v in kw.items()}
    return ToolAnnotations(**kw)


async def log(ctx: Context | None, level: str, message: str) -> None:
    """Best-effort protocol log (Claude Code does not display these; stderr is the real log)."""
    if ctx is None:
        return
    try:
        if SDK_GENERATION == 2:
            await ctx.log(level, message)  # type: ignore[arg-type]
        else:  # pragma: no cover
            await getattr(ctx, level)(message)
    except Exception:
        pass


__all__ = ["Context", "MCPServer", "ToolAnnotations", "ToolError", "SDK_GENERATION", "read_only",
           "mutating", "destructive", "log"]
