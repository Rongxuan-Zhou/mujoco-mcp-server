"""Central FastMCP instance — imported by both server.py and tool modules.

Isolating mcp here breaks the circular import between server.py and tools/*.py:
  Before: server.py → tools/simulation.py → server.py (circular)
  After:  server.py → _registry.py ← tools/simulation.py (no cycle)

lifespan is wired in server.py via _set_lifespan() before mcp.run() is called.

Root cause fix: FastMCP creates _mcp_server at __init__ time with the lifespan
value — setting mcp.settings.lifespan afterwards does NOT update _mcp_server.
Solution: pass a deferred wrapper at init time that calls _lifespan_fn lazily.
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

from mcp.server.fastmcp import FastMCP

# Set by server.py via _set_lifespan() before mcp.run() is called.
_lifespan_fn: Any = None


@asynccontextmanager
async def _deferred_lifespan(server: "FastMCP") -> AsyncIterator[Any]:
    """Wrapper that delegates to the actual lifespan set in server.py."""
    if _lifespan_fn is not None:
        async with _lifespan_fn(server) as ctx:
            yield ctx
    else:
        yield {}


def _set_lifespan(fn: Any) -> None:
    """Called by server.py to register the real app_lifespan."""
    global _lifespan_fn
    _lifespan_fn = fn


mcp = FastMCP(
    "mujoco-sim",
    lifespan=_deferred_lifespan,
    dependencies=[
        "mujoco>=2.3.0", "numpy", "Pillow",
        "pandas", "matplotlib", "packaging",
    ],
)
