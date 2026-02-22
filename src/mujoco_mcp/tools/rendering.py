"""Rendering tools (tools 9–10): render_snapshot, render_depth."""

import json
import base64
import mujoco
import numpy as np
from io import BytesIO
from PIL import Image
from mcp.types import TextContent, ImageContent
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_camera, update_scene
from . import safe_tool


@mcp.tool()
@safe_tool
async def render_snapshot(
    ctx: Context,
    camera: str | None = None,
    width: int | None = None,
    height: int | None = None,
    show_contacts: bool = False,
    sim_name: str | None = None,
) -> list:
    """Render the current simulation state as a PNG image.

    Returns an ImageContent (PNG) plus a TextContent with physics summary.
    Raises a clear error if no GL backend is available.

    Args:
        camera: Named camera. None = free camera.
        width / height: Resize output (preserves aspect if only one given).
        show_contacts: Overlay contact force arrows.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    renderer = mgr.require_renderer(slot)
    m, d = slot.model, slot.data

    cam_id = resolve_camera(m, camera)
    opt = None
    if show_contacts:
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    update_scene(renderer, d, camera_id=cam_id, scene_option=opt)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    if width or height:
        new_w = width or img.width
        new_h = height or img.height
        img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        ImageContent(type="image", data=b64, mimeType="image/png"),
        TextContent(
            type="text",
            text=(
                f"t={d.time:.4f}s | {d.ncon} contacts | "
                f"E=[{d.energy[0]:.3f}, {d.energy[1]:.3f}]"
            ),
        ),
    ]


@mcp.tool()
@safe_tool
async def render_depth(
    ctx: Context,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Render a depth map as a grayscale PNG image.

    Returns an ImageContent (grayscale PNG) plus depth statistics.
    Closer objects appear darker; distant objects appear lighter.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    renderer = mgr.require_renderer(slot)
    cam_id = resolve_camera(slot.model, camera)

    # I-4: use try/finally so renderer state is always restored on exception
    renderer.enable_depth_rendering()
    try:
        update_scene(renderer, slot.data, camera_id=cam_id)
        depth = renderer.render()
    finally:
        renderer.disable_depth_rendering()

    # C-2: guard against all-zero depth buffer (empty scene or far-plane clipping)
    d_max = float(depth.max())
    if d_max > 0:
        d_norm = ((1.0 - depth / d_max) * 255).clip(0, 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(depth, dtype=np.uint8)
    img = Image.fromarray(d_norm, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        ImageContent(type="image", data=b64, mimeType="image/png"),
        TextContent(
            type="text",
            text=f"Depth: min={float(depth.min()):.3f} max={float(depth.max()):.3f}",
        ),
    ]
