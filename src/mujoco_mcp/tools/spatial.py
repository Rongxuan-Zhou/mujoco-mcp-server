"""Phase 4a: Spatial reasoning tools for natural-language scene setup.

These tools let Claude query object positions and compute placement coordinates
without manually calculating numbers — bridging natural language to MuJoCo coords.
"""

import json
import numpy as np
import mujoco
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


# ─── Private helpers ─────────────────────────────────────────────────────────

def _body_name_to_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve body name to ID. Raises ValueError with available names if not found."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
            for i in range(model.nbody)
        ]
        raise ValueError(f"Body {name!r} not found. Available: {available}")
    return bid


def _collect_subtree(model: mujoco.MjModel, root_id: int) -> list[int]:
    """BFS: collect all body IDs in the subtree rooted at root_id (inclusive)."""
    result = []
    queue = [root_id]
    while queue:
        bid = queue.pop(0)
        result.append(bid)
        for child in range(model.nbody):
            if child != bid and model.body_parentid[child] == bid:
                queue.append(child)
    return result


def _geom_aabb_world(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> tuple:
    """Return (lo, hi) AABB of one geom in world coordinates. Returns (None,None) for planes."""
    pos = data.geom_xpos[geom_id].copy()
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]

    if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        return None, None  # Infinite — skip

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = size[:3].copy()
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        half = np.array([r, r, h])
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = size[0], size[1]
        half = np.array([r, r, h + r])
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        aabb = model.mesh_aabb[mesh_id]          # [cx, cy, cz, hx, hy, hz] in mesh local frame
        center_local = aabb[:3]
        half_m = aabb[3:6]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    lp = center_local + np.array([sx * half_m[0], sy * half_m[1], sz * half_m[2]])
                    corners.append(pos + mat @ lp)
        arr = np.array(corners)
        return arr.min(axis=0), arr.max(axis=0)
    else:
        # Ellipsoid, SDF, etc. — use bounding sphere
        half = np.full(3, size[0])

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(pos + mat @ np.array([sx * half[0], sy * half[1], sz * half[2]]))
    arr = np.array(corners)
    return arr.min(axis=0), arr.max(axis=0)


def _body_aabb_impl(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, include_children: bool
) -> tuple:
    """World-frame AABB for a body (optionally including its subtree)."""
    root_id = _body_name_to_id(model, body_name)
    body_ids = _collect_subtree(model, root_id) if include_children else [root_id]

    lo_list, hi_list = [], []
    for bid in body_ids:
        n_geoms = model.body_geomnum[bid]
        start = model.body_geomadr[bid]
        for gid in range(start, start + n_geoms):
            lo, hi = _geom_aabb_world(model, data, gid)
            if lo is not None:
                lo_list.append(lo)
                hi_list.append(hi)

    if not lo_list:
        # Body has no geoms — return zero-size box at body CoM
        xpos = data.xpos[root_id]
        return xpos.copy(), xpos.copy()

    return np.min(lo_list, axis=0), np.max(hi_list, axis=0)


_SURFACE_AXIS = {
    "top": 2, "bottom": 2,
    "+x": 0, "-x": 0,
    "+y": 1, "-y": 1,
}
_SURFACE_SIGN = {
    "top": +1, "bottom": -1,
    "+x": +1, "-x": -1,
    "+y": +1, "-y": -1,
}


def _surface_anchor_impl(
    lo: np.ndarray, hi: np.ndarray, surface: str, anchor: str
) -> np.ndarray:
    """Convert AABB + surface/anchor description to a world-frame point."""
    if surface not in _SURFACE_AXIS:
        raise ValueError(f"surface must be one of {list(_SURFACE_AXIS)}, got {surface!r}")

    anchor_map = {
        "center": {},
        "+x": {0: hi[0]},  "-x": {0: lo[0]},
        "+y": {1: hi[1]},  "-y": {1: lo[1]},
        "+z": {2: hi[2]},  "-z": {2: lo[2]},
        "+x+y": {0: hi[0], 1: hi[1]},
        "+x-y": {0: hi[0], 1: lo[1]},
        "-x+y": {0: lo[0], 1: hi[1]},
        "-x-y": {0: lo[0], 1: lo[1]},
    }
    if anchor not in anchor_map:
        raise ValueError(f"anchor must be one of {list(anchor_map)}, got {anchor!r}")

    center = (lo + hi) / 2.0
    pt = center.copy()

    # Set the primary face axis
    axis = _SURFACE_AXIS[surface]
    pt[axis] = hi[axis] if _SURFACE_SIGN[surface] > 0 else lo[axis]

    # Apply anchor offsets within the face (skip primary axis)
    for dim, val in anchor_map[anchor].items():
        if dim != axis:
            pt[dim] = val

    return pt


# ─── MCP Tools ───────────────────────────────────────────────────────────────

@mcp.tool()
@safe_tool
async def scene_map(ctx: Context, sim_name: str | None = None) -> str:
    """Return a hierarchical map of all bodies in the scene with world positions.

    Use this first before any spatial query to learn body names and scene layout.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"bodies": [{"id": int, "name": str, "parent": str|null, "pos": [x,y,z]}]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data

    bodies = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
        parent_id = model.body_parentid[i]
        parent_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or f"<body:{parent_id}>"
            if i > 0 else None
        )
        bodies.append({"id": i, "name": name, "parent": parent_name, "pos": data.xpos[i].tolist()})

    return json.dumps({"bodies": bodies}, indent=2)


@mcp.tool()
@safe_tool
async def body_aabb(
    ctx: Context,
    body_name: str,
    sim_name: str | None = None,
    include_children: bool = True,
) -> str:
    """Compute axis-aligned bounding box (AABB) of a body in world coordinates.

    Args:
        body_name: Name of the body (use scene_map to find available names).
        sim_name: Slot name (default slot if None).
        include_children: Include child bodies in the AABB (default True).

    Returns:
        JSON: {"min": [x,y,z], "max": [x,y,z], "center": [x,y,z],
               "size": [sx,sy,sz], "top_z": float}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, body_name, include_children)
    center = ((lo + hi) / 2).tolist()
    size = (hi - lo).tolist()
    return json.dumps({
        "body": body_name,
        "min": lo.tolist(),
        "max": hi.tolist(),
        "center": center,
        "size": size,
        "top_z": float(hi[2]),
    }, indent=2)


@mcp.tool()
@safe_tool
async def surface_anchor(
    ctx: Context,
    body_name: str,
    surface: str,
    anchor: str = "center",
    sim_name: str | None = None,
) -> str:
    """Get world coordinates of a specific point on a body's surface.

    Args:
        body_name: Name of the body.
        surface: Face -- "top"|"bottom"|"+x"|"-x"|"+y"|"-y".
        anchor: Point within the face -- "center"|"+x"|"-x"|"+y"|"-y"|
                "+x+y"|"+x-y"|"-x+y"|"-x-y".
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"body": str, "surface": str, "anchor": str, "pos": [x, y, z]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, body_name, include_children=True)
    pt = _surface_anchor_impl(lo, hi, surface, anchor)
    return json.dumps({
        "body": body_name,
        "surface": surface,
        "anchor": anchor,
        "pos": pt.tolist(),
    }, indent=2)


@mcp.tool()
@safe_tool
async def compute_placement(
    ctx: Context,
    target_body: str,
    surface: str,
    anchor: str = "center",
    object_half_height: float = 0.0,
    sim_name: str | None = None,
) -> str:
    """Compute the world position to place an object on a body's surface.

    Pure spatial computation -- does NOT modify the simulation.
    Use the returned placement_pos with modify_model() or sim_set_state().

    Args:
        target_body: Body to place the object on.
        surface: Surface face -- "top"|"bottom"|"+x"|"-x"|"+y"|"-y".
        anchor: Anchor point within face -- "center"|"+x"|"-x"|"+y"|"-y"|etc.
        object_half_height: Half-height of the object being placed (m).
                            Added in the surface normal direction to avoid interpenetration.
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"placement_pos": [x,y,z], "surface_pos": [x,y,z],
               "target_body": str, "surface": str, "anchor": str}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, target_body, include_children=True)
    surface_pt = _surface_anchor_impl(lo, hi, surface, anchor)

    # Offset in the surface normal direction
    axis = _SURFACE_AXIS[surface]
    sign = _SURFACE_SIGN[surface]
    placement = surface_pt.copy()
    placement[axis] += sign * object_half_height

    return json.dumps({
        "target_body": target_body,
        "surface": surface,
        "anchor": anchor,
        "surface_pos": surface_pt.tolist(),
        "placement_pos": placement.tolist(),
    }, indent=2)
