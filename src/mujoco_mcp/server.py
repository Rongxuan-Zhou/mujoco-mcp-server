"""
MuJoCo MCP Server entry point.

IMPORT ORDER IS CRITICAL:
1. detect_and_set_gl_backend()  ← sets MUJOCO_GL env var
2. import mujoco                ← reads MUJOCO_GL once, cannot change after
3. everything else (compat, sim_manager, tools)

The FastMCP instance lives in _registry.py so tool modules can import it
without creating a circular dependency back to this module.
"""

import os
import logging

from .utils.gl_setup import detect_and_set_gl_backend

_gl_backend = detect_and_set_gl_backend()  # STEP 1: set MUJOCO_GL before mujoco import

import mujoco  # noqa: E402  # STEP 2: reads MUJOCO_GL once

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP

from ._registry import mcp, _set_lifespan
from .sim_manager import SimManager
from .compat import get_version_info

logger = logging.getLogger(__name__)


@dataclass
class AppContext:
    sim_manager: SimManager


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    ver = get_version_info()
    logger.info(f"MuJoCo MCP Server starting: {ver}")
    logger.info(f"GL backend: {_gl_backend}")

    manager = SimManager(
        enable_rendering=(_gl_backend != "none"),
        render_width=int(os.environ.get("MUJOCO_MCP_RENDER_WIDTH", "640")),
        render_height=int(os.environ.get("MUJOCO_MCP_RENDER_HEIGHT", "480")),
    )
    manager.init_rendering()

    try:
        yield AppContext(sim_manager=manager)
    finally:
        manager.cleanup()
        logger.info("MuJoCo MCP Server stopped.")


# Wire the lifespan into the shared FastMCP instance via the deferred wrapper
_set_lifespan(app_lifespan)

# Register all tool modules (importing them causes @mcp.tool() decorators to fire)
from .tools import simulation, rendering, meta  # noqa: E402, F401
from .tools import analysis, model, batch, export, workflows, viewer  # noqa: E402, F401
from . import resources, prompts  # noqa: E402, F401

# Keep backward-compatible alias
mcp_server = mcp


def main():
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="MuJoCo MCP Server")
    p.add_argument("--transport", default="stdio",
                   choices=["stdio", "streamable-http"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
        )
