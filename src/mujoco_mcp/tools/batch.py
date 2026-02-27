"""Batch tools (tool 24): run_sweep — parallel parameter sweep."""

import json
import csv
import os
import asyncio
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

import numpy as np
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..constants import BATCH_TASK_TIMEOUT, BATCH_MAX_WORKERS_DEFAULT
from . import safe_tool


# ──────────────────────────────────────────────
# Top-level helpers (must be picklable for subprocess)
# ──────────────────────────────────────────────

def _apply_param(m, param: str, value) -> None:
    """Apply a dot-notation parameter to a compiled MjModel.

    Format: "element.name.field"  or  "option.field"
    Examples:
      "geom.box_geom.friction"  → m.geom_friction[geom_id]  = value
      "body.box.mass"           → m.body_mass[body_id]       = value
      "joint.slider.range"      → m.jnt_range[jnt_id]        = value
      "option.timestep"         → m.opt.timestep             = value
    """
    import mujoco

    _ELEM_MAP = {
        "geom":     (mujoco.mjtObj.mjOBJ_GEOM,     "geom"),
        "body":     (mujoco.mjtObj.mjOBJ_BODY,     "body"),
        "joint":    (mujoco.mjtObj.mjOBJ_JOINT,    "jnt"),
        "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator"),
        "site":     (mujoco.mjtObj.mjOBJ_SITE,     "site"),
    }

    parts = param.split(".", 2)

    if parts[0] == "option":
        if len(parts) != 2:
            raise ValueError(f"option param must be 'option.field', got '{param}'")
        field = parts[1]
        attr = getattr(m.opt, field)
        if isinstance(attr, np.ndarray):
            attr[:] = np.asarray(value, dtype=attr.dtype)
        else:
            setattr(m.opt, field, type(attr)(value))
        return

    if len(parts) != 3:
        raise ValueError(
            f"Invalid param '{param}'. Use 'element.name.field' or 'option.field'."
        )
    elem, name, field = parts
    if elem not in _ELEM_MAP:
        raise ValueError(f"Unknown element '{elem}'. Use: {list(_ELEM_MAP)}")

    obj_type, prefix = _ELEM_MAP[elem]
    eid = mujoco.mj_name2id(m, obj_type, name)
    if eid < 0:
        raise ValueError(f"{elem} '{name}' not found in model")

    arr = getattr(m, f"{prefix}_{field}")
    row = arr[eid]
    if isinstance(row, np.ndarray):
        row[:] = np.asarray(value, dtype=arr.dtype)
    else:
        arr[eid] = type(row)(value)


def _run_single_experiment(
    xml_source: dict,
    param: str,
    value,
    n_steps: int,
    track: list[str],
) -> dict:
    """Run one sweep experiment in a child process. Never touches GL.

    Args:
        xml_source: {"xml_path": str} or {"xml_string": str}
        param:      Dot-notation parameter name (e.g. "geom.box.friction")
        value:      Parameter value for this experiment
        n_steps:    Number of mj_step calls
        track:      Quantities to record per step

    Returns summary dict (no images — child process has no GL).
    """
    import mujoco
    import numpy as np

    # Load model without renderer
    if "xml_path" in xml_source:
        m = mujoco.MjModel.from_xml_path(xml_source["xml_path"])
    else:
        m = mujoco.MjModel.from_xml_string(xml_source["xml_string"])
    d = mujoco.MjData(m)

    _apply_param(m, param, value)
    mujoco.mj_forward(m, d)

    trajectory = []
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
        frame: dict = {"t": float(d.time)}
        if "energy" in track:
            frame["kinetic"]      = float(d.energy[1])
            frame["potential"]    = float(d.energy[0])
            frame["total_energy"] = float(d.energy[0] + d.energy[1])
        if "contact_count" in track:
            frame["contact_count"] = int(d.ncon)
        if "qpos" in track:
            frame["qpos"] = d.qpos.tolist()
        if "qvel" in track:
            frame["qvel"] = d.qvel.tolist()
        trajectory.append(frame)

    # Build summary statistics
    summary: dict = {"param": param, "value": value, "n_steps": n_steps}
    if trajectory:
        if "energy" in track:
            energies = [f["total_energy"] for f in trajectory]
            summary["energy_mean"]  = float(np.mean(energies))
            summary["energy_std"]   = float(np.std(energies))
            summary["energy_final"] = float(energies[-1])
        if "contact_count" in track:
            counts = [f["contact_count"] for f in trajectory]
            summary["contact_mean"] = float(np.mean(counts))
            summary["contact_max"]  = int(max(counts))
    summary["trajectory"] = trajectory
    return summary


# ──────────────────────────────────────────────
# MCP tool
# ──────────────────────────────────────────────

@mcp.tool()
@safe_tool
async def run_sweep(
    ctx: Context,
    param: str,
    values: list,
    n_steps: int = 500,
    track: list[str] | None = None,
    output_csv: str | None = None,
    max_workers: int | None = None,
    sim_name: str | None = None,
) -> str:
    """Parallel parameter sweep: run one experiment per value, return statistics.

    Args:
        param:       Dot-notation parameter: "geom.box_geom.friction", "option.timestep"
        values:      List of values to sweep (e.g. [0.1, 0.3, 0.5, 0.8])
        n_steps:     Simulation steps per experiment (default 500)
        track:       Quantities to record: ["energy", "contact_count", "qpos", "qvel"]
        output_csv:  Optional path to write full trajectory CSV
        max_workers: Worker processes (default: MUJOCO_MCP_MAX_WORKERS env or 8)

    Returns summary JSON with per-value statistics.

    Child processes never touch GL — simulation only, safe on headless nodes.
    Per-task timeout: 300 s.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)

    # Determine XML source for child processes (GL-free)
    if slot.xml_string:
        xml_source = {"xml_string": slot.xml_string}
    elif slot.xml_path:
        xml_source = {"xml_path": slot.xml_path}
    else:
        return json.dumps({"error": "Slot has no XML source. Reload with sim_load first."})

    if not values:
        return json.dumps({"error": "values list is empty"})

    if track is None:
        track = ["energy", "contact_count"]

    workers = max_workers or int(
        os.environ.get("MUJOCO_MCP_MAX_WORKERS", str(BATCH_MAX_WORKERS_DEFAULT))
    )
    workers = min(workers, len(values))

    loop = asyncio.get_running_loop()
    results = []
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as pool:
        cf_futures = [
            pool.submit(_run_single_experiment, xml_source, param, v, n_steps, track)
            for v in values
        ]

        async def _collect(v, cf):
            try:
                return await loop.run_in_executor(
                    None, lambda: cf.result(timeout=BATCH_TASK_TIMEOUT)
                )
            except FuturesTimeoutError:
                return {"_error": True, "value": v,
                        "error": "timeout", "timeout_s": BATCH_TASK_TIMEOUT}
            except Exception as e:
                return {"_error": True, "value": v, "error": str(e)}

        raw = await asyncio.gather(
            *[_collect(v, cf) for v, cf in zip(values, cf_futures)]
        )

    # Separate results from errors; collect trajectory frames for CSV
    all_frames: list[dict] = []
    for item in raw:
        if item.get("_error"):
            errors.append({k: v for k, v in item.items() if k != "_error"})
        else:
            traj = item.pop("trajectory", [])
            item["n_trajectory_frames"] = len(traj)
            results.append(item)
            if output_csv:
                val = item["value"]
                for frame in traj:
                    frame["param_value"] = val
                all_frames.extend(traj)

    # Write CSV
    csv_path = None
    if output_csv and all_frames:
        fieldnames = list(all_frames[0].keys())
        try:
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for frame in all_frames:
                    row = {
                        k: json.dumps(v) if isinstance(v, list) else v
                        for k, v in frame.items()
                    }
                    writer.writerow(row)
            csv_path = output_csv
        except Exception as e:
            errors.append({"csv_error": str(e)})

    return json.dumps({
        "ok":        len(errors) == 0,
        "param":     param,
        "n_values":  len(values),
        "n_success": len(results),
        "n_errors":  len(errors),
        "results":   results,
        "errors":    errors,
        "csv_path":  csv_path,
    }, indent=2)
