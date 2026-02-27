"""Meta tools (tool 27): server_diagnostics."""

import asyncio
import json
import os
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import get_version_info
from ..constants import BATCH_MAX_WORKERS_DEFAULT
from ..utils.gl_setup import get_gl_diagnostics
from . import safe_tool


@mcp.tool()
@safe_tool
async def server_diagnostics(ctx: Context) -> str:
    """Return server health info: MuJoCo version, GL backend, loaded slots, env config.

    Run this first when debugging rendering or compatibility issues.
    Note: GL probe sub-tests run in a thread pool to avoid blocking the event loop.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager

    # C-4: use snapshot_slots() to avoid iterating an unlocked dict
    slots = mgr.snapshot_slots()
    slots_info = {
        name: {
            "nq":           s.model.nq,
            "nv":           s.model.nv,
            "nu":           s.model.nu,
            "time":         s.data.time,
            "recording":    s.recording,
            "traj_frames":  len(s.trajectory),
            "has_renderer": s.renderer is not None,
        }
        for name, s in slots.items()
    }

    # I-3: get_gl_diagnostics() spawns subprocesses (up to 30s).
    # Run in a thread pool so the async event loop stays unblocked.
    loop = asyncio.get_running_loop()
    gl_info = await loop.run_in_executor(None, get_gl_diagnostics)

    return json.dumps({
        "version": get_version_info(),
        "gl": gl_info,
        "rendering": {
            "enabled":   mgr._rendering_enabled,
            "available": mgr.can_render,
            "width":     mgr.render_width,
            "height":    mgr.render_height,
        },
        "env": {
            "MUJOCO_GL":              os.environ.get("MUJOCO_GL", "unset"),
            "MUJOCO_MCP_NO_RENDER":   os.environ.get("MUJOCO_MCP_NO_RENDER", "0"),
            "MUJOCO_MCP_MAX_WORKERS": os.environ.get("MUJOCO_MCP_MAX_WORKERS", str(BATCH_MAX_WORKERS_DEFAULT)),
        },
        "active_slot": mgr.active_slot,
        "slots":       slots_info,
    }, indent=2)
