# Robustness Tool Group Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 3 MCP tools (`apply_perturbation`, `stability_analysis`, `randomize_dynamics`) for perturbation robustness testing and domain randomization.

**Architecture:** Single `src/mujoco_mcp/tools/robustness.py` module. Each tool has a `_XXX_impl()` sync function (testable without MCP server) wrapped by an async `@mcp.tool()` + `@safe_tool` function. State save/restore via `mujoco.mj_copyData`; parameter save/restore via inline `_get_param`/`_set_param`.

**Tech Stack:** Python 3.10+, MuJoCo ≥ 2.3, numpy, csv (stdlib), FastMCP

---

## Reference: Key project conventions

- Decorator order: `@mcp.tool()` outer, `@safe_tool` inner
- All errors via `raise`, never `return {"error": ...}`
- Every async tool: `await asyncio.sleep(0)` before calling `_impl`
- `_impl` functions are synchronous, testable directly (no MCP server needed)
- `ctx: Context` type annotation required (import from `mcp.server.fastmcp`)
- SimManager: `mgr = ctx.request_context.lifespan_context.sim_manager; slot = mgr.get(sim_name)`

---

## Task 1: Write 12 Failing Tests

**Files:**
- Create: `tests/test_robustness.py`

**Step 1: Write the test file**

```python
"""Tests for robustness tools (apply_perturbation, stability_analysis, randomize_dynamics)."""
import csv
import json
import tempfile

import numpy as np
import pytest
import mujoco

from mujoco_mcp.tools.robustness import (
    _apply_perturbation_impl,
    _stability_analysis_impl,
    _randomize_dynamics_impl,
)

# ── Test models ────────────────────────────────────────────────────────────────

# Damped pendulum: hangs at rest, oscillates under gravity, dissipates energy
PENDULUM_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pendulum">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="2.0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

# Slider with named geom: for randomize_dynamics parameter variation
SLIDER_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def _make_pendulum():
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _make_slider():
    model = mujoco.MjModel.from_xml_string(SLIDER_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── apply_perturbation tests ───────────────────────────────────────────────────

def test_apply_perturbation_increases_velocity():
    """A non-zero force pulse must produce a non-zero qvel deviation."""
    model, data = _make_pendulum()
    result = json.loads(_apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[10.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=20, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert result["max_qvel_deviation"] > 0.0


def test_apply_perturbation_small_force_recovers():
    """A small perturbation of a damped pendulum should recover within recovery_steps."""
    model, data = _make_pendulum()
    result = json.loads(_apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[0.1, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=10, recovery_steps=500,
        with_controller=False, recovery_threshold=0.01,
    ))
    assert result["recovered"] is True
    assert result["recovery_time_steps"] is not None


def test_apply_perturbation_unknown_body_raises():
    """Unknown body name must raise ValueError."""
    model, data = _make_pendulum()
    with pytest.raises(ValueError, match="not found"):
        _apply_perturbation_impl(
            model, data, controller=None,
            body_name="nonexistent", force=[1.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
            n_steps=10, recovery_steps=20,
            with_controller=False, recovery_threshold=0.1,
        )


def test_apply_perturbation_slot_state_unchanged():
    """Original slot data must be restored after apply_perturbation."""
    model, data = _make_pendulum()
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    _apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[5.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=30, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    )
    assert np.allclose(data.qpos, qpos_before), "qpos was modified"
    assert np.allclose(data.qvel, qvel_before), "qvel was modified"


# ── stability_analysis tests ───────────────────────────────────────────────────

def test_stability_analysis_result_schema():
    """Result must have all required keys."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0, 5.0],
        n_directions=4,
        n_steps=10, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    ))
    for key in ("stability_margin", "failure_ratio", "n_trials", "results"):
        assert key in result, f"Missing key: {key}"


def test_stability_analysis_directions_count():
    """Total trials must equal n_magnitudes × n_directions."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0, 5.0, 10.0],
        n_directions=6,
        n_steps=10, recovery_steps=30,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert result["n_trials"] == 3 * 6
    assert len(result["results"]) == 3 * 6


def test_stability_analysis_failure_ratio_range():
    """failure_ratio must be in [0, 1]."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0],
        n_directions=4,
        n_steps=5, recovery_steps=20,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert 0.0 <= result["failure_ratio"] <= 1.0


# ── randomize_dynamics tests ───────────────────────────────────────────────────

def test_randomize_dynamics_basic():
    """Basic uniform distribution: result must have correct schema and n_samples."""
    model, data = _make_slider()
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=10,
        eval_steps=20,
        metric="max_speed",
        goal_qpos=None,
        export_csv=None,
        random_seed=42,
    ))
    assert result["n_samples"] == 10
    for key in ("mean", "std", "min", "max", "best_params", "worst_params", "samples"):
        assert key in result, f"Missing key: {key}"
    assert result["min"] <= result["mean"] <= result["max"]


def test_randomize_dynamics_log_uniform_range():
    """log_uniform sampled values must stay within [low, high]."""
    model, data = _make_slider()
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "log_uniform", "low": 0.1, "high": 10.0}},
        n_samples=20,
        eval_steps=10,
        metric="max_speed",
        goal_qpos=None,
        export_csv=None,
        random_seed=7,
    ))
    for s in result["samples"]:
        mass = s["params"]["geom.box.mass"]
        assert 0.1 <= mass <= 10.0, f"log_uniform sample out of range: {mass}"


def test_randomize_dynamics_unknown_metric_raises():
    """Unknown metric must raise ValueError."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="metric"):
        _randomize_dynamics_impl(
            model, data,
            param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
            n_samples=2, eval_steps=5,
            metric="invalid_metric",
            goal_qpos=None, export_csv=None, random_seed=None,
        )


def test_randomize_dynamics_distance_without_goal_raises():
    """metric='distance' without goal_qpos must raise ValueError."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="goal_qpos"):
        _randomize_dynamics_impl(
            model, data,
            param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
            n_samples=2, eval_steps=5,
            metric="distance",
            goal_qpos=None,  # missing
            export_csv=None, random_seed=None,
        )


def test_randomize_dynamics_seeded_reproducible():
    """Same random_seed must produce identical mean across two calls."""
    model, data = _make_slider()
    kwargs = dict(
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=8, eval_steps=15,
        metric="max_speed", goal_qpos=None, export_csv=None, random_seed=123,
    )
    result1 = json.loads(_randomize_dynamics_impl(model, data, **kwargs))
    result2 = json.loads(_randomize_dynamics_impl(model, data, **kwargs))
    assert result1["mean"] == result2["mean"]


def test_randomize_dynamics_csv_export():
    """export_csv must create a readable CSV with correct columns."""
    model, data = _make_slider()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        csv_path = f.name
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=5, eval_steps=10,
        metric="max_speed", goal_qpos=None,
        export_csv=csv_path, random_seed=0,
    ))
    assert result["csv_path"] == csv_path
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 5
    assert "geom.box.mass" in rows[0]
    assert "max_speed" in rows[0]
```

**Step 2: Run tests to verify they all fail**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
python -m pytest tests/test_robustness.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name '_apply_perturbation_impl' from 'mujoco_mcp.tools.robustness'`

**Step 3: Commit failing tests**

```bash
git add tests/test_robustness.py
git commit -m "test(robustness): add 12 failing tests for apply_perturbation, stability_analysis, randomize_dynamics"
```

---

## Task 2: Implement `robustness.py`

**Files:**
- Create: `src/mujoco_mcp/tools/robustness.py`

**Step 1: Write the complete implementation**

```python
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
    """Read a model parameter by dot-notation path (element.name.field or option.field)."""
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
    arr = getattr(m, f"{prefix}_{field}")
    return arr[idx].copy() if isinstance(arr[idx], np.ndarray) else float(arr[idx])


def _set_param(m: mujoco.MjModel, param: str, value: float) -> None:
    """Write a model parameter by dot-notation path (element.name.field or option.field)."""
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

    # Save original state via deep copy
    backup = mujoco.MjData(model)
    mujoco.mj_copyData(backup, model, data)

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
        mujoco.mj_copyData(data, model, backup)
        mujoco.mj_forward(model, data)


@mcp.tool()
@safe_tool
async def apply_perturbation(
    ctx: Context,
    body_name: str,
    force: list[float] = [0.0, 0.0, 0.0],
    torque: list[float] = [0.0, 0.0, 0.0],
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
    force_magnitudes: list[float] = [1.0, 5.0, 10.0, 20.0],
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
            for _ in range(eval_steps):
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
      - "geom.sphere.mass"     → model.geom_mass[geom_id]
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
```

**Step 2: Run the 12 tests**

```bash
python -m pytest tests/test_robustness.py -v
```

Expected: 12 passed

**Step 3: Commit implementation**

```bash
git add src/mujoco_mcp/tools/robustness.py
git commit -m "feat(robustness): implement apply_perturbation, stability_analysis, randomize_dynamics"
```

---

## Task 3: Register Module + Full Test Suite

**Files:**
- Modify: `src/mujoco_mcp/server.py` (line with the tools import)

**Step 1: Add `robustness` to the import**

Find the line:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env, vision, diagnostics, kinematics, optimization  # noqa: E402,F401,E501
```

Change to:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env, vision, diagnostics, kinematics, optimization, robustness  # noqa: E402,F401,E501
```

**Step 2: Run full test suite**

```bash
python -m pytest tests/ -q
```

Expected: `123 passed` (111 existing + 12 new)

**Step 3: Commit**

```bash
git add src/mujoco_mcp/server.py
git commit -m "feat: register robustness tools in server.py"
```

---

## Task 4: Update README

**Files:**
- Modify: `README.md`

**Step 1: Update badge** — change `tests-111%20passed` → `tests-123%20passed`

**Step 2: Update tool count** — change `59 MCP tools` → `62 MCP tools`

**Step 3: Add table row** — after the Optimization row, add:
```
| **Robustness** | `apply_perturbation` `stability_analysis` `randomize_dynamics` | Perturbation recovery analysis and domain randomization |
```

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update tool count (59→62), test count (111→123), add Robustness row"
```

---

## Task 5: Final Verification + Push

**Step 1: Run full suite + lint**

```bash
python -m pytest tests/ -q && ruff check src/mujoco_mcp
```

Expected: `123 passed` + `All checks passed!`

**Step 2: Review commits**

```bash
git log --oneline -6
```

**Step 3: Push**

```bash
git push origin master
```

**Step 4: Update MEMORY.md**

Update project status line:
```
**Phase 1 ✅ ... Optimization ✅ Robustness ✅ — 62 工具，123 测试通过**
```

Add Robustness section in MEMORY.md with key implementation details.
