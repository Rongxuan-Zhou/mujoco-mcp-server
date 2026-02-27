"""Model tools (tools 17–18): modify_model, reload_from_xml."""

import json
import mujoco
import numpy as np
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool, _viewer_sync

# Maps element name → (mjtObj type, MjModel array prefix)
_ELEM_INFO = {
    "geom":     (mujoco.mjtObj.mjOBJ_GEOM,     "geom"),
    "body":     (mujoco.mjtObj.mjOBJ_BODY,     "body"),
    "joint":    (mujoco.mjtObj.mjOBJ_JOINT,    "jnt"),
    "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator"),
    "site":     (mujoco.mjtObj.mjOBJ_SITE,     "site"),
}


@mcp.tool()
@safe_tool
async def modify_model(
    ctx: Context,
    modifications: list[dict],
    sim_name: str | None = None,
) -> str:
    """Modify compiled MjModel fields in-place via numpy writes. No recompilation needed.

    Each modification dict must contain:
      - element: "geom" | "body" | "joint" | "actuator" | "site" | "option"
      - field:   numpy array field name (e.g. "friction", "mass", "timestep")
      - value:   new value (scalar or list)
      - name:    element name (omit for element="option")

    mj_forward is called automatically after all modifications.

    Examples:
      {"element": "geom",   "name": "box_geom", "field": "friction", "value": [0.5, 0.005, 0.001]}
      {"element": "body",   "name": "box",      "field": "mass",     "value": 2.0}
      {"element": "joint",  "name": "slider",   "field": "range",    "value": [-1.0, 1.0]}
      {"element": "option", "field": "timestep", "value": 0.002}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    results = []

    for mod in modifications:
        elem  = mod.get("element", "")
        field = mod.get("field", "")
        value = mod.get("value")
        try:
            if elem == "option":
                # Direct access on m.opt
                attr = getattr(m.opt, field)
                if isinstance(attr, np.ndarray):
                    old = attr.copy().tolist()
                    attr[:] = np.asarray(value, dtype=attr.dtype)
                else:
                    old = attr
                    setattr(m.opt, field, type(attr)(value))
                results.append({"elem": "option", "field": field, "old": old, "new": value, "ok": True})

            elif elem in _ELEM_INFO:
                name = mod.get("name")
                if not name:
                    raise ValueError(f"'name' required for element '{elem}'")
                obj_type, prefix = _ELEM_INFO[elem]
                eid = mujoco.mj_name2id(m, obj_type, name)
                if eid < 0:
                    raise ValueError(f"{elem} '{name}' not found in model")

                # Access via MjModel numpy array: e.g. m.geom_friction[eid]
                arr = getattr(m, f"{prefix}_{field}")
                row = arr[eid]
                if isinstance(row, np.ndarray):
                    old = row.copy().tolist()
                    row[:] = np.asarray(value, dtype=arr.dtype)
                else:
                    old = float(row)
                    arr[eid] = type(row)(value)
                results.append({
                    "elem": elem, "name": name, "field": field,
                    "old": old, "new": value, "ok": True,
                })
            else:
                raise ValueError(
                    f"Unknown element '{elem}'. "
                    f"Use: geom, body, joint, actuator, site, option"
                )
        except Exception as e:
            results.append({"elem": elem, "field": field, "error": str(e), "ok": False})

    mujoco.mj_forward(m, d)
    _viewer_sync(slot)
    all_ok = all(r.get("ok") for r in results)
    return json.dumps({"ok": all_ok, "modifications": results}, indent=2)


@mcp.tool()
@safe_tool
async def reload_from_xml(
    ctx: Context,
    xml_string: str,
    sim_name: str | None = None,
) -> str:
    """Full model reload from XML string. Use for structural changes (add/remove bodies/joints).

    Resets simulation state to t=0. For parameter-only changes (friction, mass, etc.)
    prefer modify_model — it is instant and preserves simulation state.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)          # None → active_slot，与其他工具一致
    summary = mgr.load(slot.name, xml_string=xml_string)
    # Viewer is invalidated on full reload (new model/data pointers).
    # Inform the caller so they can call viewer_open again if needed.
    summary["viewer_note"] = "Viewer closed — call viewer_open() to reopen with the new model."
    return json.dumps(summary, indent=2)
