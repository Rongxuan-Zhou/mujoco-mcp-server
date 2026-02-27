"""Workflow tools (tools 19–23): high-level research-oriented compositions of atomic tools."""

import asyncio
import base64
import json
import mujoco
import numpy as np
from io import BytesIO
from PIL import Image
from mcp.types import TextContent, ImageContent
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_camera, update_scene, ensure_energy, restore_energy, contact_geoms
from ..constants import MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL
from . import safe_tool


def _encode_png(pixels: np.ndarray) -> str:
    """Encode HxWx3 uint8 array to base64 PNG string."""
    img = Image.fromarray(pixels)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _snapshot(renderer, data, cam_id: int = -1, opt=None) -> ImageContent:
    update_scene(renderer, data, camera_id=cam_id, scene_option=opt)
    pixels = renderer.render()
    return ImageContent(type="image", data=_encode_png(pixels), mimeType="image/png")


# ---------------------------------------------------------------------------
# Tool 19: run_and_analyze
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def run_and_analyze(
    ctx: Context,
    n_steps: int = 1000,
    ctrl: list[float] | None = None,
    capture_every_n: int = 200,
    track: list[str] | None = None,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Run simulation and return a time-series + sparse keyframe images.

    Primary tool for observing a trajectory. Returns physics data at every
    timestep plus visual keyframes every capture_every_n steps.

    Args:
        n_steps: Total steps (max 100k).
        ctrl: Constant control vector applied before stepping. None = keep current.
        capture_every_n: Render an image every N steps (0 = no images).
        track: Quantities to record per step. Options:
            "qpos", "qvel", "energy", "contact_count",
            "sensor:<name>", "body_xpos:<name>"
            Default: ["energy", "contact_count"]
        camera: Named camera for keyframes. None = free camera.

    Returns:
        [TextContent(JSON time-series), ImageContent, ImageContent, ...]
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data
    track = track or ["energy", "contact_count"]
    n_steps = min(n_steps, MAX_SIM_STEPS)

    if ctrl is not None:
        if len(ctrl) != m.nu:
            return [TextContent(type="text",
                text=json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"}))]
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    cam_id = resolve_camera(m, camera)
    can_render = mgr.can_render and slot.renderer is not None and capture_every_n > 0

    # Enable energy computation for the duration of this tool call if tracking energy
    energy_was_enabled = ensure_energy(m) if "energy" in track else True

    timeseries: list[dict] = []
    keyframes: list[ImageContent] = []

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop every 1000 steps

        row: dict = {"t": round(d.time, 6)}
        for t in track:
            if t == "energy":
                row["E_pot"] = float(d.energy[0])
                row["E_kin"] = float(d.energy[1])
            elif t == "contact_count":
                row["ncon"] = int(d.ncon)
            elif t == "qpos":
                row["qpos"] = d.qpos.tolist()
            elif t == "qvel":
                row["qvel"] = d.qvel.tolist()
            elif t.startswith("sensor:"):
                sname = t.split(":", 1)[1]
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
                if sid >= 0:
                    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
                    row[t] = d.sensordata[adr: adr + dim].tolist()
            elif t.startswith("body_xpos:"):
                bname = t.split(":", 1)[1]
                row[t] = d.body(bname).xpos.tolist()
        timeseries.append(row)

        if can_render and step % capture_every_n == 0:
            keyframes.append(_snapshot(slot.renderer, d, cam_id))

    restore_energy(m, energy_was_enabled)

    summary = {
        "n_steps": len(timeseries),
        "sim_time": [timeseries[0]["t"], timeseries[-1]["t"]] if timeseries else [0, 0],
        "final_state": {
            "qpos": d.qpos.tolist(),
            "qvel": d.qvel.tolist(),
            "n_contacts": int(d.ncon),
            "energy": [float(d.energy[0]), float(d.energy[1])],
        },
        "timeseries": timeseries,
    }
    return [TextContent(type="text", text=json.dumps(summary))] + keyframes


# ---------------------------------------------------------------------------
# Tool 20: debug_contacts
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def debug_contacts(
    ctx: Context,
    n_steps: int = 500,
    ctrl: list[float] | None = None,
    force_threshold: float | None = None,
    penetration_threshold: float = 0.005,
    capture_on_change: bool = True,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Run simulation with event-triggered contact capture.

    Renders a snapshot ONLY when contact topology changes or force/penetration
    thresholds are exceeded. Records the full contact force time-series regardless.

    Args:
        n_steps: Simulation steps.
        force_threshold: Capture snapshot when |normal_force| exceeds this. None = off.
        penetration_threshold: Capture snapshot when penetration depth exceeds this (m).
        capture_on_change: Capture snapshot whenever contact pairs change.
        camera: Named camera. None = free camera.

    Returns:
        [TextContent(event log + force traces), ImageContent, ...]
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data
    n_steps = min(n_steps, MAX_SIM_STEPS)

    if ctrl is not None:
        if len(ctrl) != m.nu:
            return [TextContent(type="text",
                text=json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"}))]
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    cam_id = resolve_camera(m, camera)
    can_render = mgr.can_render and slot.renderer is not None

    events: list[dict] = []
    traces: list[dict] = []
    keyframes: list[ImageContent] = []
    prev_pairs: set[tuple] = set()

    contact_opt = mujoco.MjvOption()
    contact_opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    for _step_i, _ in enumerate(range(n_steps)):
        mujoco.mj_step(m, d)
        if (_step_i + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop every 1000 steps
        cur_pairs: set[tuple] = set()
        step_forces: dict = {}
        should_capture = False

        for ci in range(d.ncon):
            c = d.contact[ci]
            gid1, gid2 = contact_geoms(c)
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid1) or str(gid1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid2) or str(gid2)
            pair = tuple(sorted([g1, g2]))
            cur_pairs.add(pair)

            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, force)
            key = f"{pair[0]}↔{pair[1]}"
            step_forces[key] = {"fn": float(force[0]), "dist": float(c.dist)}

            if force_threshold is not None and abs(force[0]) > force_threshold:
                events.append({
                    "t": round(d.time, 6), "event": "force_exceeded",
                    "pair": list(pair), "fn": float(force[0]),
                })
                should_capture = True
            if c.dist < -penetration_threshold:
                events.append({
                    "t": round(d.time, 6), "event": "penetration",
                    "pair": list(pair), "depth": round(-float(c.dist), 5),
                })
                should_capture = True

        new_pairs = cur_pairs - prev_pairs
        lost_pairs = prev_pairs - cur_pairs
        if new_pairs or lost_pairs:
            for p in new_pairs:
                events.append({"t": round(d.time, 6), "event": "contact_made", "pair": list(p)})
            for p in lost_pairs:
                events.append({"t": round(d.time, 6), "event": "contact_lost",  "pair": list(p)})
            if capture_on_change:
                should_capture = True
        prev_pairs = cur_pairs

        traces.append({"t": round(d.time, 6), "ncon": d.ncon, "forces": step_forces})

        if should_capture and can_render:
            keyframes.append(_snapshot(slot.renderer, d, cam_id, opt=contact_opt))

    result = {
        "n_events":             len(events),
        "events":               events,
        "contact_trace_length": len(traces),
        "traces":               traces,
    }
    return [TextContent(type="text", text=json.dumps(result))] + keyframes


# ---------------------------------------------------------------------------
# Tool 21: evaluate_trajectory
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def evaluate_trajectory(
    ctx: Context,
    trajectory: list[dict] | None = None,
    trajectory_csv: str | None = None,
    n_keyframes: int = 6,
    energy_threshold: float = 1.0,
    penetration_limit: float = 0.01,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Replay an external trajectory and check physical plausibility.

    Sets qpos/qvel per frame using mj_forward (NOT mj_step), then checks:
    energy conservation, joint limits, penetration depth, velocity smoothness.

    Trajectory format (list of dicts or CSV):
      - "qpos": list of floats (length nq) OR columns "qpos_0", "qpos_1", ...
      - "qvel": list of floats (length nv) OR columns "qvel_0", "qvel_1", ...
      - "time" or "t": timestamp (optional)

    Returns:
        [TextContent(plausibility report), ImageContent keyframes...]
    """
    import pandas as pd

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    if trajectory_csv:
        loop = asyncio.get_running_loop()
        df = await loop.run_in_executor(None, pd.read_csv, trajectory_csv)
        frames = df.to_dict("records")
    elif trajectory:
        frames = trajectory
    else:
        return [TextContent(type="text",
            text=json.dumps({"error": "Provide trajectory (list) or trajectory_csv (path)"}))]

    if not frames:
        return [TextContent(type="text", text=json.dumps({"error": "Empty trajectory"}))]

    cam_id = resolve_camera(m, camera)
    can_render = mgr.can_render and slot.renderer is not None
    kf_indices = set(np.linspace(0, len(frames) - 1, n_keyframes, dtype=int).tolist())

    # Energy check requires the flag to be set (mj_forward computes energy only when enabled)
    energy_was_enabled = ensure_energy(m)

    violations: list[dict] = []
    energy_trace: list[dict] = []
    keyframes: list[ImageContent] = []
    prev_E: float | None = None

    for i, frame in enumerate(frames):
        if (i + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop every 1000 frames
        # --- Set state ---
        if "qpos" in frame and isinstance(frame["qpos"], list):
            d.qpos[:] = np.array(frame["qpos"], dtype=np.float64)
        else:
            qpos_cols = sorted(k for k in frame if k.startswith("qpos_"))
            if qpos_cols:
                d.qpos[:] = np.array([frame[k] for k in qpos_cols], dtype=np.float64)

        if "qvel" in frame and isinstance(frame["qvel"], list):
            d.qvel[:] = np.array(frame["qvel"], dtype=np.float64)
        else:
            qvel_cols = sorted(k for k in frame if k.startswith("qvel_"))
            if qvel_cols:
                d.qvel[:] = np.array([frame[k] for k in qvel_cols], dtype=np.float64)

        mujoco.mj_forward(m, d)
        t = float(frame.get("time", frame.get("t", i * m.opt.timestep)))
        E = float(d.energy[0] + d.energy[1])

        # Energy conservation check
        if prev_E is not None and abs(E - prev_E) > energy_threshold:
            violations.append({
                "t": round(t, 6), "type": "energy_jump",
                "delta": round(abs(E - prev_E), 4),
            })
        prev_E = E
        energy_trace.append({"t": round(t, 6), "E": round(E, 6)})

        # Penetration check
        for ci in range(d.ncon):
            c = d.contact[ci]
            if c.dist < -penetration_limit:
                gid1, gid2 = contact_geoms(c)
                g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid1) or "?"
                g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid2) or "?"
                violations.append({
                    "t": round(t, 6), "type": "penetration",
                    "pair": [g1, g2], "depth": round(-float(c.dist), 5),
                })

        # Joint limit check
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                q = float(d.qpos[m.jnt_qposadr[j]])
                lo, hi = float(m.jnt_range[j, 0]), float(m.jnt_range[j, 1])
                if q < lo - 1e-4 or q > hi + 1e-4:
                    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or str(j)
                    violations.append({
                        "t": round(t, 6), "type": "joint_limit",
                        "joint": jname, "value": round(q, 4), "range": [lo, hi],
                    })

        # Keyframe capture
        if i in kf_indices and can_render:
            keyframes.append(_snapshot(slot.renderer, d, cam_id))

    restore_energy(m, energy_was_enabled)

    score = max(0, 100 - len(violations) * 5)
    report = {
        "plausibility_score": score,
        "n_frames":           len(frames),
        "n_violations":       len(violations),
        "violations":         violations[:50],  # cap output size
        "energy_trace":       energy_trace,
    }
    return [TextContent(type="text", text=json.dumps(report))] + keyframes


# ---------------------------------------------------------------------------
# Tool 22: compare_trajectories
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def compare_trajectories(
    ctx: Context,
    slot_a: str,
    slot_b: str,
    metric: str = "rmse",
    camera: str | None = None,
    n_keyframes: int = 4,
) -> list:
    """Compare recorded trajectories from two simulation slots.

    Both slots must have recorded trajectories (use sim_record + sim_step first).
    Computes per-DOF RMSE and returns a side-by-side summary with keyframes.

    Args:
        slot_a: Name of first slot.
        slot_b: Name of second slot.
        metric: "rmse" (only option currently).
        n_keyframes: Number of keyframes to capture from each slot at matching times.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    sa = mgr.get(slot_a)
    sb = mgr.get(slot_b)

    ta = sa.trajectory
    tb = sb.trajectory
    if not ta:
        return [TextContent(type="text",
            text=json.dumps({"error": f"Slot '{slot_a}' has no recorded trajectory"}))]
    if not tb:
        return [TextContent(type="text",
            text=json.dumps({"error": f"Slot '{slot_b}' has no recorded trajectory"}))]

    # Align by index (shorter trajectory determines length)
    n = min(len(ta), len(tb))
    qpos_a = np.array([f["qpos"] for f in ta[:n]])
    qpos_b = np.array([f["qpos"] for f in tb[:n]])

    diff = qpos_a - qpos_b
    rmse_per_dof = np.sqrt((diff ** 2).mean(axis=0)).tolist()
    rmse_total   = float(np.sqrt((diff ** 2).mean()))

    t_a = [f["t"] for f in ta[:n]]
    t_b = [f["t"] for f in tb[:n]]

    report = {
        "slot_a":       slot_a,
        "slot_b":       slot_b,
        "n_frames":     n,
        "time_range_a": [t_a[0], t_a[-1]] if t_a else [],
        "time_range_b": [t_b[0], t_b[-1]] if t_b else [],
        "rmse_total":   round(rmse_total, 6),
        "rmse_per_dof": [round(v, 6) for v in rmse_per_dof],
    }

    # Keyframes from slot_a at evenly spaced indices
    keyframes: list[ImageContent] = []
    if mgr.can_render and n_keyframes > 0:
        cam_id_a = resolve_camera(sa.model, camera)
        cam_id_b = resolve_camera(sb.model, camera)
        kf_indices = set(np.linspace(0, n - 1, n_keyframes, dtype=int).tolist())
        for ki in sorted(kf_indices):
            # Replay frame into each slot and capture
            frame_a = ta[ki]
            sa.data.qpos[:] = np.array(frame_a["qpos"])
            sa.data.qvel[:] = np.array(frame_a["qvel"])
            mujoco.mj_forward(sa.model, sa.data)
            if sa.renderer:
                keyframes.append(_snapshot(sa.renderer, sa.data, cam_id_a))

            frame_b = tb[ki]
            sb.data.qpos[:] = np.array(frame_b["qpos"])
            sb.data.qvel[:] = np.array(frame_b["qvel"])
            mujoco.mj_forward(sb.model, sb.data)
            if sb.renderer:
                keyframes.append(_snapshot(sb.renderer, sb.data, cam_id_b))

    return [TextContent(type="text", text=json.dumps(report))] + keyframes
