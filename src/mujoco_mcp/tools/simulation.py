"""Atomic simulation tools (tools 1–8): load, step, forward, reset, get/set state, record, list."""

import asyncio
import json
import mujoco
import numpy as np
from mcp.server.fastmcp import Context

from ..constants import MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL
from .._registry import mcp
from ..compat import list_named
from . import safe_tool, _viewer_sync


@mcp.tool()
@safe_tool
async def sim_load(
    ctx: Context,
    xml_path: str | None = None,
    xml_string: str | None = None,
    name: str = "default",
) -> str:
    """Load a MuJoCo MJCF model into a named simulation slot.

    Provide either xml_path (absolute path to .xml file) or xml_string (raw XML).
    Returns model summary: dimensions, named elements, renderer availability.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    summary = mgr.load(name, xml_path=xml_path, xml_string=xml_string)
    return json.dumps(summary, indent=2)


@mcp.tool()
@safe_tool
async def sim_step(
    ctx: Context,
    n_steps: int = 1,
    ctrl: list[float] | None = None,
    sim_name: str | None = None,
) -> str:
    """Advance physics simulation by n_steps timesteps (max 100 000 per call).

    Optionally set control vector before stepping. Returns state at final step.
    While recording is active, each step appends to the trajectory buffer.
    """
    if not 1 <= n_steps <= MAX_SIM_STEPS:
        return json.dumps({"error": f"n_steps must be 1–{MAX_SIM_STEPS}"})

    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data

    if ctrl is not None:
        if len(ctrl) != m.nu:
            return json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"})
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        if slot.recording:
            slot.trajectory.append({
                "t":    float(d.time),
                "qpos": d.qpos.copy().tolist(),
                "qvel": d.qvel.copy().tolist(),
            })
        if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop

    _viewer_sync(slot)
    return json.dumps({
        "time":       d.time,
        "qpos":       d.qpos.tolist(),
        "qvel":       d.qvel.tolist(),
        "energy":     [float(d.energy[0]), float(d.energy[1])],
        "n_contacts": int(d.ncon),
    }, indent=2)


@mcp.tool()
@safe_tool
async def sim_forward(ctx: Context, sim_name: str | None = None) -> str:
    """Recompute derived quantities (positions, forces, contacts) without advancing time.

    Call after sim_set_state or modify_model to update dependent quantities.
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    mujoco.mj_forward(slot.model, slot.data)
    d = slot.data
    _viewer_sync(slot)
    return json.dumps({
        "time":       d.time,
        "n_contacts": int(d.ncon),
        "energy":     [float(d.energy[0]), float(d.energy[1])],
    })


@mcp.tool()
@safe_tool
async def sim_reset(ctx: Context, sim_name: str | None = None) -> str:
    """Reset simulation to t=0 with default qpos/qvel.

    Clears and stops trajectory recording. Start recording again with sim_record.
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    mujoco.mj_resetData(slot.model, slot.data)
    mujoco.mj_forward(slot.model, slot.data)
    slot.trajectory.clear()
    slot.recording = False
    _viewer_sync(slot)
    return json.dumps({"status": "reset", "time": 0.0})


@mcp.tool()
@safe_tool
async def sim_get_state(
    ctx: Context,
    include_bodies: bool = False,
    include_sites: bool = False,
    sim_name: str | None = None,
) -> str:
    """Read current simulation state: time, qpos, qvel, ctrl.

    Set include_bodies=True to include world-frame body positions (excludes 'world' body).
    Set include_sites=True to include world-frame site positions.
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    state: dict = {
        "time": d.time,
        "qpos": d.qpos.tolist(),
        "qvel": d.qvel.tolist(),
        "ctrl": d.ctrl.tolist(),
    }
    if include_bodies:
        # Exclude 'world' body (index 0) — it has no meaningful xpos
        state["body_xpos"] = {
            n: d.body(n).xpos.tolist()
            for n in list_named(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)
            if n != "world"
        }
    if include_sites:
        state["site_xpos"] = {
            n: d.site(n).xpos.tolist()
            for n in list_named(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite)
        }
    return json.dumps(state, indent=2)


@mcp.tool()
@safe_tool
async def sim_set_state(
    ctx: Context,
    qpos: list[float] | None = None,
    qvel: list[float] | None = None,
    ctrl: list[float] | None = None,
    keyframe: int | None = None,
    sim_name: str | None = None,
) -> str:
    """Write simulation state. Calls mj_forward automatically after changes.

    Use keyframe (int index ≥ 0) to load a named keyframe from the model.
    Otherwise supply any combination of qpos (length nq), qvel (length nv), ctrl (length nu).
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data

    if keyframe is not None:
        if not 0 <= keyframe < m.nkey:
            return json.dumps({
                "error": f"keyframe {keyframe} out of range (model has {m.nkey} keyframes)"
            })
        mujoco.mj_resetDataKeyframe(m, d, keyframe)
    else:
        if qpos is not None:
            if len(qpos) != m.nq:
                return json.dumps({"error": f"qpos len {len(qpos)} != nq {m.nq}"})
            d.qpos[:] = np.array(qpos, dtype=np.float64)
        if qvel is not None:
            if len(qvel) != m.nv:
                return json.dumps({"error": f"qvel len {len(qvel)} != nv {m.nv}"})
            d.qvel[:] = np.array(qvel, dtype=np.float64)
        if ctrl is not None:
            if len(ctrl) != m.nu:
                return json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"})
            d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    mujoco.mj_forward(m, d)
    _viewer_sync(slot)
    return json.dumps({"status": "ok", "time": d.time})


@mcp.tool()
@safe_tool
async def sim_record(
    ctx: Context,
    action: str = "start",
    sim_name: str | None = None,
) -> str:
    """Control trajectory recording. action: 'start' | 'stop' | 'clear'.

    While recording, every sim_step appends {t, qpos, qvel} to the trajectory buffer.
    Use export_csv to save the recorded trajectory.
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    if action == "start":
        slot.recording = True
    elif action == "stop":
        slot.recording = False
    elif action == "clear":
        slot.trajectory.clear()
        slot.recording = False
    else:
        return json.dumps({"error": "action must be 'start', 'stop', or 'clear'"})
    return json.dumps({
        "recording": slot.recording,
        "frames": len(slot.trajectory),
    })


@mcp.tool()
@safe_tool
async def sim_list(ctx: Context) -> str:
    """List all loaded simulation slots with their status."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    # snapshot_slots() acquires lock internally — safe against concurrent load()
    slots = mgr.snapshot_slots()
    return json.dumps({
        "active": mgr.active_slot,
        "slots": {
            n: {
                "nq":           s.model.nq,
                "nv":           s.model.nv,
                "time":         s.data.time,
                "recording":    s.recording,
                "traj_frames":  len(s.trajectory),
                "has_renderer": s.renderer is not None,
            }
            for n, s in slots.items()
        },
    }, indent=2)
