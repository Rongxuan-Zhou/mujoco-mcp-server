"""Viewer tools: viewer_open, viewer_sync, viewer_close.

Bridges the MCP server to a live mujoco.viewer.launch_passive() window.
After viewer_open, every state-modifying tool (sim_step, modify_model, etc.)
automatically calls viewer.sync() so the window stays current.
"""

import json
import asyncio
import os

from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


def _has_display() -> bool:
    """Return True if a graphical display is available."""
    return bool(
        os.environ.get("DISPLAY")
        or os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("MJV_DISPLAY_OVERRIDE")   # escape hatch for testing
    )


def _close_viewer(slot) -> None:
    """Close and detach viewer from slot. Safe to call multiple times."""
    v = slot.passive_viewer
    if v is None:
        return
    try:
        v.close()
    except Exception:
        pass
    slot.passive_viewer = None


# ---------------------------------------------------------------------------
# viewer_open
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def viewer_open(
    ctx: Context,
    sim_name: str | None = None,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> str:
    """Open a live MuJoCo viewer window tied to a simulation slot.

    Once open, the viewer refreshes automatically after every state-changing
    tool call (sim_step, modify_model, sim_reset, sim_set_state, etc.) —
    no extra action needed.

    Requires a local graphical display (DISPLAY or WAYLAND_DISPLAY env var).
    Not available on headless servers; use render_snapshot there instead.

    Args:
        show_left_ui:  Show MuJoCo's left control panel (model tree, options).
        show_right_ui: Show MuJoCo's right info panel (sim stats).

    Example workflow:
        sim_load(xml_path="franka.xml")
        viewer_open()
        modify_model([{"element":"body","name":"franka_base","field":"pos","value":[0.3,0,0]}])
        # → viewer window updates instantly, no extra call needed
    """
    try:
        import mujoco.viewer as mjv
    except ImportError:
        return json.dumps({
            "error": "mujoco.viewer not available",
            "hint": "Ensure mujoco>=2.3.3 is installed with viewer support.",
        })

    if not _has_display():
        return json.dumps({
            "error": "No display found",
            "detail": (
                "DISPLAY / WAYLAND_DISPLAY not set. "
                "viewer_open requires a local desktop session. "
                "Use render_snapshot for headless rendering instead."
            ),
        })

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    target_name = sim_name or mgr.active_slot

    # Close any existing viewer for this slot
    _close_viewer(slot)

    loop = asyncio.get_event_loop()

    def _open():
        return mjv.launch_passive(
            slot.model,
            slot.data,
            show_left_ui=show_left_ui,
            show_right_ui=show_right_ui,
        )

    try:
        viewer = await loop.run_in_executor(None, _open)
    except Exception as e:
        return json.dumps({
            "error": "viewer_open_failed",
            "message": str(e),
            "hint": "Check that your display server is running and DISPLAY is set correctly.",
        })

    slot.passive_viewer = viewer
    # Initial sync to display current state
    viewer.sync()

    return json.dumps({
        "ok":      True,
        "slot":    target_name,
        "message": (
            "Viewer opened. "
            "The window updates automatically after sim_step, modify_model, "
            "sim_reset, sim_set_state, and reload_from_xml."
        ),
    }, indent=2)


# ---------------------------------------------------------------------------
# viewer_sync
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def viewer_sync(
    ctx: Context,
    sim_name: str | None = None,
) -> str:
    """Manually push the current simulation state to the viewer window.

    Most tools trigger this automatically. Call explicitly only when you need
    an immediate refresh outside a tool call (e.g. after reading sensor data
    and deciding to pause without stepping).
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)

    if slot.passive_viewer is None:
        return json.dumps({
            "ok":    False,
            "message": "No viewer open for this slot. Call viewer_open first.",
        })

    if not slot.passive_viewer.is_running():
        slot.passive_viewer = None
        return json.dumps({
            "ok":    False,
            "message": "Viewer window was closed by the user.",
        })

    slot.passive_viewer.sync()
    return json.dumps({"ok": True, "message": "Viewer synced."})


# ---------------------------------------------------------------------------
# viewer_close
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def viewer_close(
    ctx: Context,
    sim_name: str | None = None,
) -> str:
    """Close the live viewer window for a simulation slot."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)

    if slot.passive_viewer is None:
        return json.dumps({"ok": False, "message": "No viewer open."})

    _close_viewer(slot)
    return json.dumps({"ok": True, "message": "Viewer closed."})
