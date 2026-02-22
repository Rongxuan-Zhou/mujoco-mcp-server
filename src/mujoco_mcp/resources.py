"""MCP Resources — expose simulation state and model info as readable URIs."""

import json
import mujoco
from mcp.server.fastmcp import Context

from ._registry import mcp
from .compat import list_named


@mcp.resource("sim://slots")
async def resource_slots(ctx: Context) -> str:
    """List all loaded simulation slots and their key dimensions."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slots = mgr.snapshot_slots()
    return json.dumps({
        "active": mgr.active_slot,
        "slots": {
            name: {
                "nq": s.model.nq,
                "nv": s.model.nv,
                "nu": s.model.nu,
                "time": s.data.time,
                "recording": s.recording,
                "traj_frames": len(s.trajectory),
                "has_renderer": s.renderer is not None,
            }
            for name, s in slots.items()
        },
    }, indent=2)


@mcp.resource("sim://state/{slot_name}")
async def resource_state(slot_name: str, ctx: Context) -> str:
    """Current qpos, qvel, ctrl, time, contacts, energy for a named slot."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(slot_name)
    d = slot.data
    return json.dumps({
        "slot":   slot_name,
        "time":   d.time,
        "qpos":   d.qpos.tolist(),
        "qvel":   d.qvel.tolist(),
        "ctrl":   d.ctrl.tolist(),
        "energy": [float(d.energy[0]), float(d.energy[1])],
        "n_contacts": int(d.ncon),
    }, indent=2)


@mcp.resource("sim://model/{slot_name}")
async def resource_model(slot_name: str, ctx: Context) -> str:
    """Named elements (bodies, joints, geoms, actuators, sensors, cameras) for a slot."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(slot_name)
    m = slot.model
    obj = mujoco.mjtObj
    return json.dumps({
        "slot":      slot_name,
        "nq": m.nq, "nv": m.nv, "nu": m.nu,
        "timestep":  m.opt.timestep,
        "bodies":    list_named(m, obj.mjOBJ_BODY,     m.nbody),
        "joints":    list_named(m, obj.mjOBJ_JOINT,    m.njnt),
        "geoms":     list_named(m, obj.mjOBJ_GEOM,     m.ngeom),
        "actuators": list_named(m, obj.mjOBJ_ACTUATOR, m.nu),
        "sensors":   list_named(m, obj.mjOBJ_SENSOR,   m.nsensor),
        "cameras":   list_named(m, obj.mjOBJ_CAMERA,   m.ncam),
        "sites":     list_named(m, obj.mjOBJ_SITE,     m.nsite),
    }, indent=2)


@mcp.resource("sim://trajectory/{slot_name}")
async def resource_trajectory(slot_name: str, ctx: Context) -> str:
    """Recorded trajectory summary for a slot (first/last 5 frames + stats)."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(slot_name)
    traj = slot.trajectory
    if not traj:
        return json.dumps({"slot": slot_name, "frames": 0, "recording": slot.recording})
    return json.dumps({
        "slot":      slot_name,
        "frames":    len(traj),
        "recording": slot.recording,
        "t_start":   traj[0]["t"],
        "t_end":     traj[-1]["t"],
        "preview": {
            "first_5": traj[:5],
            "last_5":  traj[-5:],
        },
    }, indent=2)
