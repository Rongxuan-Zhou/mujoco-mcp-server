"""Robustness tool group — perturbation analysis and domain randomization."""
from __future__ import annotations

import asyncio
import csv
import json

import numpy as np
import mujoco
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


# ── Module-level element map (shared by _get_param and _set_param) ─────────────

_ELEM_MAP: dict = {
    "geom":     (mujoco.mjtObj.mjOBJ_GEOM,     "geom"),
    "body":     (mujoco.mjtObj.mjOBJ_BODY,     "body"),
    "joint":    (mujoco.mjtObj.mjOBJ_JOINT,    "jnt"),
    "actuator": (mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator"),
    "site":     (mujoco.mjtObj.mjOBJ_SITE,     "site"),
}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _get_param(m: mujoco.MjModel, param: str):
    """Read a model parameter by dot-notation path (element.name.field or option.field).

    Special case: geom.NAME.mass → body_mass[geom_bodyid[geom_id]]
    (MuJoCo stores geom mass on the parent body, not on geom_mass).
    """
    parts = param.split(".", 2)
    if parts[0] == "option":
        val = getattr(m.opt, parts[1])
        return val.copy() if isinstance(val, np.ndarray) else val
    if len(parts) != 3:
        raise ValueError(f"Invalid param '{param}'. Use 'element.name.field' or 'option.field'.")
    elem, name, field = parts
    if elem not in _ELEM_MAP:
        raise ValueError(f"Unknown element '{elem}'. Use: {list(_ELEM_MAP)}")
    obj_type, prefix = _ELEM_MAP[elem]
    idx = mujoco.mj_name2id(m, obj_type, name)
    if idx < 0:
        raise ValueError(f"No {elem} named '{name}'")

    # Special handling: geom.NAME.mass → body_mass of the parent body
    if elem == "geom" and field == "mass":
        body_id = int(m.geom_bodyid[idx])
        return float(m.body_mass[body_id])

    arr = getattr(m, f"{prefix}_{field}")
    return arr[idx].copy() if isinstance(arr[idx], np.ndarray) else float(arr[idx])


def _set_param(m: mujoco.MjModel, param: str, value: float) -> None:
    """Write a model parameter by dot-notation path (element.name.field or option.field).

    Special case: geom.NAME.mass → body_mass[geom_bodyid[geom_id]]
    (MuJoCo stores geom mass on the parent body, not on geom_mass).
    """
    parts = param.split(".", 2)
    if parts[0] == "option":
        field = parts[1]
        attr = getattr(m.opt, field)
        if isinstance(attr, np.ndarray):
            attr[:] = np.asarray(value, dtype=attr.dtype)
        else:
            setattr(m.opt, field, type(attr)(value))
        return
    if len(parts) != 3:
        raise ValueError(f"Invalid param '{param}'. Use 'element.name.field' or 'option.field'.")
    elem, name, field = parts
    if elem not in _ELEM_MAP:
        raise ValueError(f"Unknown element '{elem}'. Use: {list(_ELEM_MAP)}")
    obj_type, prefix = _ELEM_MAP[elem]
    idx = mujoco.mj_name2id(m, obj_type, name)
    if idx < 0:
        raise ValueError(f"No {elem} named '{name}'")

    # Special handling: geom.NAME.mass → body_mass of the parent body
    if elem == "geom" and field == "mass":
        body_id = int(m.geom_bodyid[idx])
        m.body_mass[body_id] = float(value)
        return

    arr = getattr(m, f"{prefix}_{field}")
    if isinstance(arr[idx], np.ndarray):
        arr[idx] = np.asarray(value, dtype=arr.dtype)
    else:
        arr[idx] = value


def _sample_param(dist: dict, rng: np.random.Generator) -> float:
    """Sample a scalar value from a distribution spec dict."""
    t = dist["type"]
    if t == "uniform":
        return float(rng.uniform(dist["low"], dist["high"]))
    if t == "normal":
        return float(rng.normal(dist["mean"], dist["std"]))
    if t == "log_uniform":
        lo, hi = np.log(dist["low"]), np.log(dist["high"])
        return float(np.exp(rng.uniform(lo, hi)))
    raise ValueError(
        f"Unknown distribution type '{t}'. Use: uniform, normal, log_uniform"
    )


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniformly distributed unit vectors on the sphere.

    Uses the Fibonacci/golden-ratio lattice. Works for any gravity direction.
    Returns array of shape (n, 3).
    """
    golden = (1.0 + 5.0 ** 0.5) / 2.0
    indices = np.arange(n, dtype=float)
    theta = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    phi = 2.0 * np.pi * indices / golden
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


# ── apply_perturbation ─────────────────────────────────────────────────────────

def _apply_perturbation_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller,  # slot.controller or None
    body_name: str,
    force: list[float],
    torque: list[float],
    n_steps: int,
    recovery_steps: int,
    with_controller: bool,
    recovery_threshold: float,
) -> str:
    """Apply a force/torque pulse to a body and observe recovery.

    Saves and restores the slot data so the original simulation state is
    unchanged after the call. with_controller=True uses slot.controller
    but note stateful controllers (e.g. PID integrator) start cold.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in model")

    force_arr = np.array(force, dtype=float)
    torque_arr = np.array(torque, dtype=float)

    # Save original state (manual array copy — mj_copyData not available in all builds)
    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()
    act_save = data.act.copy()
    ctrl_save = data.ctrl.copy()
    xfrc_save = data.xfrc_applied.copy()
    time_save = float(data.time)

    trajectory: list = [data.qpos.copy().tolist()]
    qvel_norms: list = [float(np.linalg.norm(data.qvel))]

    try:
        # Apply perturbation force for n_steps
        data.xfrc_applied[body_id, :3] = force_arr
        data.xfrc_applied[body_id, 3:] = torque_arr
        for _ in range(n_steps):
            if with_controller and controller is not None:
                data.ctrl[:] = controller.compute(model, data)
            mujoco.mj_step(model, data)
            trajectory.append(data.qpos.copy().tolist())
            qvel_norms.append(float(np.linalg.norm(data.qvel)))

        # Remove perturbation; observe recovery
        data.xfrc_applied[body_id, :3] = 0.0
        data.xfrc_applied[body_id, 3:] = 0.0
        recovery_time: int | None = None
        for step in range(recovery_steps):
            if with_controller and controller is not None:
                data.ctrl[:] = controller.compute(model, data)
            mujoco.mj_step(model, data)
            speed = float(np.linalg.norm(data.qvel))
            trajectory.append(data.qpos.copy().tolist())
            qvel_norms.append(speed)
            if speed < recovery_threshold and recovery_time is None:
                recovery_time = step + 1

        max_deviation = float(np.max(qvel_norms))
        recovered = recovery_time is not None

        return json.dumps({
            "body": body_name,
            "applied_force": force,
            "applied_torque": torque,
            "max_qvel_deviation": round(max_deviation, 6),
            "recovery_time_steps": recovery_time,
            "recovered": recovered,
            "trajectory": trajectory,
        })

    finally:
        # Always restore original state
        data.qpos[:] = qpos_save
        data.qvel[:] = qvel_save
        data.act[:] = act_save
        data.ctrl[:] = ctrl_save
        data.xfrc_applied[:] = xfrc_save
        data.time = time_save
        mujoco.mj_forward(model, data)


@mcp.tool()
@safe_tool
async def apply_perturbation(
    ctx: Context,
    body_name: str,
    force: list[float] | None = None,
    torque: list[float] | None = None,
    n_steps: int = 50,
    recovery_steps: int = 100,
    with_controller: bool = False,
    recovery_threshold: float = 0.1,
    sim_name: str | None = None,
) -> str:
    """Apply an external force/torque pulse to a body and observe recovery.

    The slot state is preserved: the simulation is returned to its state
    before the perturbation after the call completes.

    Args:
        body_name: Name of the body to perturb.
        force: Force vector [fx, fy, fz] in world frame (N).
        torque: Torque vector [tx, ty, tz] in world frame (N·m).
        n_steps: Number of steps the force is applied.
        recovery_steps: Additional steps to observe recovery.
        with_controller: If True, call slot.controller each step (cold start).
        recovery_threshold: |qvel| threshold (rad/s or m/s) to declare recovery.
        sim_name: Simulation slot name (default slot if None).

    Returns:
        JSON with body, applied_force, applied_torque, max_qvel_deviation,
        recovery_time_steps, recovered, trajectory.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    await asyncio.sleep(0)
    if force is None:
        force = [0.0, 0.0, 0.0]
    if torque is None:
        torque = [0.0, 0.0, 0.0]
    return _apply_perturbation_impl(
        slot.model, slot.data, slot.controller,
        body_name=body_name, force=force, torque=torque,
        n_steps=n_steps, recovery_steps=recovery_steps,
        with_controller=with_controller,
        recovery_threshold=recovery_threshold,
    )


# ── stability_analysis ─────────────────────────────────────────────────────────

def _stability_analysis_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller,
    body_name: str,
    force_magnitudes: list[float],
    n_directions: int,
    n_steps: int,
    recovery_steps: int,
    with_controller: bool,
    recovery_threshold: float,
) -> str:
    """Sweep force magnitudes × sphere directions to characterise stability margin."""
    directions = _fibonacci_sphere(n_directions)  # (n_directions, 3)
    magnitudes_sorted = sorted(force_magnitudes)
    results: list[dict] = []
    n_failed = 0

    for magnitude in magnitudes_sorted:
        for direction in directions:
            force = (direction * magnitude).tolist()
            raw = json.loads(_apply_perturbation_impl(
                model, data, controller,
                body_name=body_name, force=force, torque=[0.0, 0.0, 0.0],
                n_steps=n_steps, recovery_steps=recovery_steps,
                with_controller=with_controller,
                recovery_threshold=recovery_threshold,
            ))
            recovered = raw["recovered"]
            if not recovered:
                n_failed += 1
            results.append({
                "magnitude": float(magnitude),
                "direction": direction.tolist(),
                "recovered": recovered,
                "recovery_steps": raw["recovery_time_steps"],
            })

    # Stability margin: largest magnitude where ALL directions recovered
    stability_margin = 0.0
    for mag in magnitudes_sorted:
        mag_results = [r for r in results if r["magnitude"] == mag]
        if all(r["recovered"] for r in mag_results):
            stability_margin = float(mag)
        else:
            break

    n_trials = len(results)
    return json.dumps({
        "stability_margin": stability_margin,
        "failure_ratio": round(n_failed / n_trials, 4) if n_trials else 0.0,
        "n_trials": n_trials,
        "results": results,
    })


@mcp.tool()
@safe_tool
async def stability_analysis(
    ctx: Context,
    body_name: str,
    force_magnitudes: list[float] | None = None,
    n_directions: int = 8,
    n_steps: int = 50,
    recovery_steps: int = 200,
    with_controller: bool = False,
    recovery_threshold: float = 0.1,
    sim_name: str | None = None,
) -> str:
    """Characterise stability margin by sweeping force magnitudes and directions.

    Uses Fibonacci sphere sampling for directions (works for any gravity orientation).
    Stability margin is the largest force magnitude at which ALL directions recover.

    Args:
        body_name: Body to perturb.
        force_magnitudes: List of force magnitudes to test (N).
        n_directions: Number of sphere directions to test per magnitude.
        n_steps: Steps of applied force.
        recovery_steps: Steps to observe recovery.
        with_controller: Whether to call slot.controller during recovery.
        recovery_threshold: |qvel| threshold to declare recovery.
        sim_name: Simulation slot name.

    Returns:
        JSON with stability_margin, failure_ratio, n_trials, results list.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    await asyncio.sleep(0)
    if force_magnitudes is None:
        force_magnitudes = [1.0, 5.0, 10.0, 20.0]
    return _stability_analysis_impl(
        slot.model, slot.data, slot.controller,
        body_name=body_name, force_magnitudes=force_magnitudes,
        n_directions=n_directions, n_steps=n_steps,
        recovery_steps=recovery_steps, with_controller=with_controller,
        recovery_threshold=recovery_threshold,
    )


# ── randomize_dynamics ─────────────────────────────────────────────────────────

def _randomize_dynamics_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    param_distributions: dict,
    n_samples: int,
    eval_steps: int,
    metric: str,
    goal_qpos: list[float] | None,
    export_csv: str | None,
    random_seed: int | None,
) -> str:
    """Sample N parameter sets, run simulations, and return robustness statistics."""
    # Validate inputs before any simulation
    valid_metrics = ("energy", "max_speed", "distance")
    if metric not in valid_metrics:
        raise ValueError(f"Unknown metric '{metric}'. Use: {valid_metrics}")
    if metric == "distance" and goal_qpos is None:
        raise ValueError("goal_qpos required for metric='distance'")

    goal = np.array(goal_qpos, dtype=float) if goal_qpos is not None else None
    rng = np.random.default_rng(random_seed)
    sample_results: list[dict] = []

    for _ in range(n_samples):
        sampled = {p: _sample_param(d, rng) for p, d in param_distributions.items()}
        originals = {p: _get_param(model, p) for p in sampled}
        orig_flags = int(model.opt.enableflags)

        try:
            # Apply sampled parameters in-place on shared model
            for p, v in sampled.items():
                _set_param(model, p, v)

            # Enable energy tracking if needed
            if metric == "energy":
                model.opt.enableflags = orig_flags | int(
                    mujoco.mjtEnableBit.mjENBL_ENERGY
                )

            # Run simulation on isolated MjData (slot.data unchanged)
            d_tmp = mujoco.MjData(model)
            d_tmp.qpos[:] = data.qpos
            d_tmp.qvel[:] = data.qvel
            mujoco.mj_forward(model, d_tmp)

            max_speed = 0.0
            for _step in range(eval_steps):
                mujoco.mj_step(model, d_tmp)
                speed = float(np.linalg.norm(d_tmp.qvel))
                if speed > max_speed:
                    max_speed = speed

            # Compute metric value
            if metric == "energy":
                val = float(d_tmp.energy[0] + d_tmp.energy[1])
            elif metric == "max_speed":
                val = max_speed
            else:  # distance
                dq = np.zeros(model.nv)
                mujoco.mj_differentiatePos(model, dq, 1.0, goal, d_tmp.qpos)
                val = float(np.linalg.norm(dq))

            sample_results.append({
                "params": {k: round(float(v), 6) for k, v in sampled.items()},
                "value": round(val, 6),
            })

        finally:
            # Always restore original model parameters and flags
            for p, v in originals.items():
                _set_param(model, p, v)
            model.opt.enableflags = orig_flags

    # Compute statistics
    values = [s["value"] for s in sample_results]
    arr = np.array(values)
    sorted_results = sorted(sample_results, key=lambda s: s["value"])
    worst_5 = sorted_results[-5:]
    best_5 = sorted_results[:5]

    # Optional CSV export
    csv_path: str | None = None
    if export_csv is not None:
        param_names = list(param_distributions.keys())
        with open(export_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id"] + param_names + [metric])
            for i, s in enumerate(sample_results):
                writer.writerow(
                    [i] + [s["params"][p] for p in param_names] + [s["value"]]
                )
        csv_path = export_csv

    return json.dumps({
        "n_samples": n_samples,
        "metric": metric,
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "max": round(float(arr.max()), 6),
        "best_params": best_5[0]["params"] if best_5 else {},
        "worst_params": worst_5[-1]["params"] if worst_5 else {},
        "samples": worst_5 + best_5,
        "csv_path": csv_path,
    })


@mcp.tool()
@safe_tool
async def randomize_dynamics(
    ctx: Context,
    param_distributions: dict,
    n_samples: int = 20,
    eval_steps: int = 200,
    metric: str = "max_speed",
    goal_qpos: list[float] | None = None,
    export_csv: str | None = None,
    random_seed: int | None = None,
    sim_name: str | None = None,
) -> str:
    """Sample N physics parameter sets from distributions and evaluate robustness.

    Parameters use dot-notation (same as run_sweep):
      - "geom.sphere.mass"     → body_mass of the geom's parent body
      - "body.torso.mass"      → model.body_mass[body_id]
      - "option.timestep"      → model.opt.timestep

    Distribution specs:
      - {"type": "uniform",     "low": 0.5, "high": 2.0}
      - {"type": "normal",      "mean": 1.0, "std": 0.2}
      - {"type": "log_uniform", "low": 0.1, "high": 10.0}

    The original model parameters are restored after each sample.
    slot.data is never modified.

    Args:
        param_distributions: Dict of param_path → distribution spec.
        n_samples: Number of sampled parameter sets.
        eval_steps: Simulation steps per sample.
        metric: "energy" | "max_speed" | "distance".
        goal_qpos: Target joint positions (required for metric="distance").
        export_csv: Path to write per-sample CSV (optional).
        random_seed: RNG seed for reproducibility.
        sim_name: Simulation slot name.

    Returns:
        JSON with n_samples, metric, mean, std, min, max,
        best_params, worst_params, samples, csv_path.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    await asyncio.sleep(0)
    return _randomize_dynamics_impl(
        slot.model, slot.data,
        param_distributions=param_distributions,
        n_samples=n_samples,
        eval_steps=eval_steps,
        metric=metric,
        goal_qpos=goal_qpos,
        export_csv=export_csv,
        random_seed=random_seed,
    )
