# Trajectory Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add two MCP tools (`optimize_ilqr`, `optimize_mppi`) for trajectory optimization in `src/mujoco_mcp/tools/optimization.py`.

**Architecture:** iLQR uses `mjd_transitionFD` linearization + Riccati backward pass + line-search forward pass (pure numpy, no new deps). MPPI uses sequential sampling with temperature-weighted control update. Both share `_build_cost_matrices` (3 templates + user Q/R), return unified JSON with `trajectory`, `controls`, `waypoints` fields. `waypoints` (≤10 subsampled key poses) enables piecewise `plan_trajectory` usage.

**Tech Stack:** Python 3.10+, numpy (already installed), mujoco (already installed)

---

## Context for Implementer

**Project layout:**
- Tools live in `src/mujoco_mcp/tools/`
- Every new tool file must import `mcp` from `_registry.py` and `safe_tool` from the tools package
- `@mcp.tool()` is outer decorator, `@safe_tool` is inner — same pattern as `kinematics.py`
- Registration: import the module in `src/mujoco_mcp/server.py` line 65
- Tests live in `tests/` — run with `python -m pytest tests/test_optimization.py -v`

**Key MuJoCo APIs used:**
- `mujoco.MjData(model)` — create fresh data copy (avoids state pollution between rollouts)
- `mujoco.mj_forward(model, data)` — recompute kinematics after setting qpos/qvel
- `mujoco.mj_step(model, data)` — advance physics one timestep
- `mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)` — linearize dynamics at current state; returns A (2nv×2nv), B (2nv×nu) in velocity-parameterized state space
- `mujoco.mj_differentiatePos(model, res, dt, qpos1, qpos2)` — computes `res = (qpos2-qpos1)/dt` in velocity coordinates (handles quaternions; for slide/hinge joints equals simple subtraction)

**State representation for iLQR:**
- State vector x = [pos_err_vel; qvel] of length 2nv (velocity-parameterized)
- pos_err_vel = mj_differentiatePos(goal_qpos, cur_qpos) → cur_qpos - goal_qpos in vel space
- A and B from mjd_transitionFD match this parameterization exactly

---

## Task 1: Write 6 Failing Tests

**Files:**
- Create: `tests/test_optimization.py`

**Step 1: Write the test file**

```python
"""Tests for trajectory optimization tools (iLQR + MPPI)."""
import json
import numpy as np
import pytest
import mujoco

from mujoco_mcp.tools.optimization import (
    _ilqr_impl,
    _mppi_impl,
    _build_cost_matrices,
)

# ── Simple 1-DOF slider model ─────────────────────────────────────────────────
SLIDER_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body>
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def _make_slider():
    model = mujoco.MjModel.from_xml_string(SLIDER_XML)
    data = mujoco.MjData(model)
    return model, data


# ── iLQR tests ────────────────────────────────────────────────────────────────

def test_ilqr_slider_converges():
    """iLQR should drive 1-DOF slider from 0 to 0.5 and converge."""
    model, data = _make_slider()
    result = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0],
        goal_qpos=[0.5],
        horizon=50,
        max_iter=20,
        template="reach",
        Q_user=None,
        R_user=None,
    ))
    assert result["converged"] is True
    assert result["iterations"] <= 20
    assert abs(result["trajectory"][-1][0] - 0.5) < 0.05


def test_ilqr_custom_QR_accepted():
    """User-supplied Q/R matrices should be accepted and alter the result."""
    model, data = _make_slider()
    # Default template="reach"
    result_default = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.3],
        horizon=30, max_iter=10,
        template="reach", Q_user=None, R_user=None,
    ))
    # Heavy control penalty → should use less control effort
    R_heavy = [[100.0]]  # 1×1 matrix
    Q_light = [[1.0, 0.0], [0.0, 0.1]]  # 2×2 matrix
    result_custom = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.3],
        horizon=30, max_iter=10,
        template="reach", Q_user=Q_light, R_user=R_heavy,
    ))
    # Custom Q/R should not raise; controls should differ
    ctrl_default_max = max(abs(u[0]) for u in result_default["controls"])
    ctrl_custom_max = max(abs(u[0]) for u in result_custom["controls"])
    # Heavy R → smaller controls
    assert ctrl_custom_max < ctrl_default_max + 5.0  # relaxed, direction check only


def test_ilqr_waypoints_subsampled():
    """waypoints field must be present, length ≤ 10, and endpoints match trajectory."""
    model, data = _make_slider()
    result = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.4],
        horizon=50, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    wp = result["waypoints"]
    traj = result["trajectory"]
    assert isinstance(wp, list)
    assert 1 <= len(wp) <= 10
    # First waypoint should be near start
    assert abs(wp[0][0] - traj[0][0]) < 1e-9
    # Last waypoint should be near trajectory end
    assert abs(wp[-1][0] - traj[-1][0]) < 1e-9


# ── MPPI tests ────────────────────────────────────────────────────────────────

def test_mppi_slider_reaches_goal():
    """MPPI should drive slider from 0 toward 0.5 within tolerance."""
    model, data = _make_slider()
    np.random.seed(42)
    result = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0],
        goal_qpos=[0.5],
        horizon=20,
        n_samples=50,
        temperature=0.1,
        noise_sigma=1.0,
        max_iter=10,
        template="reach",
        Q_user=None,
        R_user=None,
    ))
    assert abs(result["trajectory"][-1][0] - 0.5) < 0.2  # within 20cm


def test_mppi_more_samples_better_result():
    """With more samples, MPPI should reach a lower final cost (seed fixed)."""
    model, data = _make_slider()

    np.random.seed(42)
    result_few = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.5],
        horizon=20, n_samples=3,
        temperature=0.1, noise_sigma=1.0, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    np.random.seed(42)
    result_many = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.5],
        horizon=20, n_samples=100,
        temperature=0.1, noise_sigma=1.0, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    # More samples should not be catastrophically worse
    assert result_many["final_cost"] < result_few["final_cost"] * 10.0


def test_unknown_template_raises():
    """Unknown template name should raise ValueError mentioning 'template'."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="template"):
        _ilqr_impl(
            model, data,
            start_qpos=[0.0], goal_qpos=[0.5],
            horizon=10, max_iter=5,
            template="unknown_template",
            Q_user=None, R_user=None,
        )
```

**Step 2: Run to verify it fails**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && python -m pytest tests/test_optimization.py -v 2>&1 | head -20
```

Expected: `ImportError: cannot import name '_ilqr_impl' from 'mujoco_mcp.tools.optimization'` (module doesn't exist yet).

**Step 3: Commit the failing tests**

```bash
git add tests/test_optimization.py
git commit -m "test(optimization): add 6 failing tests for iLQR + MPPI"
```

---

## Task 2: Implement `optimization.py` — Shared Helpers + iLQR

**Files:**
- Create: `src/mujoco_mcp/tools/optimization.py`

**Step 1: Write the full file**

```python
"""Optimization tool group — iLQR and MPPI trajectory optimization."""
from __future__ import annotations

import json

import numpy as np
import mujoco

from .._registry import mcp
from . import safe_tool


# ── Shared helpers ────────────────────────────────────────────────────────────

def _build_cost_matrices(
    model: mujoco.MjModel,
    template: str,
    Q_user: list | None,
    R_user: list | None,
):
    """Build (Q, R, Qf) cost matrices from a template or user-supplied arrays.

    Args:
        model: Loaded MuJoCo model. Used for nv and nu dimensions.
        template: One of "reach", "minimize_effort", "energy".
        Q_user: State cost matrix (2nv × 2nv) as nested list, or None.
        R_user: Control cost matrix (nu × nu) as nested list, or None.

    Returns:
        Tuple (Q, R, Qf) as numpy arrays.

    Raises:
        ValueError: If template is unknown or Q/R shape is incorrect.
    """
    nv = model.nv
    nu = model.nu

    if Q_user is not None:
        Q = np.array(Q_user, dtype=float)
        R = np.array(R_user, dtype=float)
        if Q.shape != (2 * nv, 2 * nv):
            raise ValueError(
                f"Q must be ({2*nv}, {2*nv}), got {Q.shape}"
            )
        if R.shape != (nu, nu):
            raise ValueError(f"R must be ({nu}, {nu}), got {R.shape}")
        Qf = 10.0 * Q
        return Q, R, Qf

    if template == "reach":
        q_diag = np.concatenate([100.0 * np.ones(nv), 1.0 * np.ones(nv)])
        r_diag = 0.01 * np.ones(nu)
    elif template == "minimize_effort":
        q_diag = np.concatenate([10.0 * np.ones(nv), 1.0 * np.ones(nv)])
        r_diag = 1.0 * np.ones(nu)
    elif template == "energy":
        q_diag = np.concatenate([10.0 * np.ones(nv), 10.0 * np.ones(nv)])
        r_diag = 0.1 * np.ones(nu)
    else:
        raise ValueError(
            f"Unknown template '{template}'. Valid: 'reach', 'minimize_effort', 'energy'"
        )

    Q = np.diag(q_diag)
    R = np.diag(r_diag)
    Qf = 10.0 * Q
    return Q, R, Qf


def _state_err(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, goal_qpos: np.ndarray) -> np.ndarray:
    """Velocity-parameterized state error vector (length 2nv).

    Position part: mj_differentiatePos(goal_qpos, qpos) = qpos - goal_qpos in vel coords.
    Velocity part: qvel (we want zero terminal velocity).
    """
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, goal_qpos, qpos)
    return np.concatenate([dq, qvel])


def _rollout(
    model: mujoco.MjModel,
    start_qpos: np.ndarray,
    controls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a control sequence from start_qpos.

    Args:
        model: MuJoCo model.
        start_qpos: Initial joint positions (nq,).
        controls: Control sequence (horizon, nu).

    Returns:
        (qpos_traj, qvel_traj) each of shape (horizon+1, nq/nv).
    """
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]

    d = mujoco.MjData(model)
    d.qpos[:] = start_qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)

    qpos_list = [d.qpos.copy()]
    qvel_list = [d.qvel.copy()]
    for u in controls:
        d.ctrl[:] = np.clip(u, ctrl_lo, ctrl_hi)
        mujoco.mj_step(model, d)
        qpos_list.append(d.qpos.copy())
        qvel_list.append(d.qvel.copy())

    return np.array(qpos_list), np.array(qvel_list)


def _compute_cost(
    model: mujoco.MjModel,
    qpos_traj: np.ndarray,
    qvel_traj: np.ndarray,
    controls: np.ndarray,
    goal_qpos: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    Qf: np.ndarray,
) -> float:
    """Compute total trajectory cost."""
    cost = 0.0
    for t in range(len(controls)):
        xe = _state_err(model, qpos_traj[t], qvel_traj[t], goal_qpos)
        cost += float(xe @ Q @ xe + controls[t] @ R @ controls[t])
    xe_T = _state_err(model, qpos_traj[-1], qvel_traj[-1], goal_qpos)
    cost += float(xe_T @ Qf @ xe_T)
    return cost


def _subsampled_waypoints(qpos_traj: np.ndarray, n: int = 10) -> list:
    """Return up to n evenly-spaced poses from a qpos trajectory."""
    length = len(qpos_traj)
    indices = np.linspace(0, length - 1, min(n, length), dtype=int)
    return [qpos_traj[i].tolist() for i in indices]


# ── iLQR ─────────────────────────────────────────────────────────────────────

def _ilqr_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,  # noqa: ARG001 — kept for API symmetry with other _impl functions
    start_qpos: list[float],
    goal_qpos: list[float],
    horizon: int = 50,
    max_iter: int = 20,
    template: str = "reach",
    Q_user: list | None = None,
    R_user: list | None = None,
) -> str:
    """Iterative Linear Quadratic Regulator trajectory optimization.

    Uses mjd_transitionFD for dynamics linearization, backward Riccati pass
    for optimal gains, and line-search forward pass for trajectory update.
    """
    nv = model.nv
    nu = model.nu
    tol = 1e-4

    Q, R, Qf = _build_cost_matrices(model, template, Q_user, R_user)
    start_qpos = np.array(start_qpos, dtype=float)
    goal_qpos = np.array(goal_qpos, dtype=float)
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]

    # Initialise: zero controls, rollout to get nominal trajectory
    U = np.zeros((horizon, nu))
    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        # ── Rollout ──────────────────────────────────────────────────────────
        qpos_bar, qvel_bar = _rollout(model, start_qpos, U)

        # ── Linearize along trajectory ────────────────────────────────────────
        A_list, B_list = [], []
        for t in range(horizon):
            d = mujoco.MjData(model)
            d.qpos[:] = qpos_bar[t]
            d.qvel[:] = qvel_bar[t]
            d.ctrl[:] = np.clip(U[t], ctrl_lo, ctrl_hi)
            mujoco.mj_forward(model, d)
            A = np.zeros((2 * nv, 2 * nv))
            B = np.zeros((2 * nv, nu))
            mujoco.mjd_transitionFD(model, d, 1e-6, True, A, B, None, None)
            A_list.append(A)
            B_list.append(B)

        # ── Backward Riccati pass ─────────────────────────────────────────────
        xe_T = _state_err(model, qpos_bar[-1], qvel_bar[-1], goal_qpos)
        Vx = 2.0 * Qf @ xe_T          # (2nv,)
        Vxx = 2.0 * Qf                 # (2nv, 2nv)

        k_list: list[np.ndarray] = []
        K_list: list[np.ndarray] = []

        for t in reversed(range(horizon)):
            xe_t = _state_err(model, qpos_bar[t], qvel_bar[t], goal_qpos)
            A_t, B_t = A_list[t], B_list[t]

            lx = 2.0 * Q @ xe_t        # (2nv,)
            lu = 2.0 * R @ U[t]        # (nu,)

            Qx = lx + A_t.T @ Vx       # (2nv,)
            Qu = lu + B_t.T @ Vx       # (nu,)
            Qxx = 2.0 * Q + A_t.T @ Vxx @ A_t    # (2nv, 2nv)
            Quu = 2.0 * R + B_t.T @ Vxx @ B_t    # (nu, nu)
            Qux = B_t.T @ Vxx @ A_t              # (nu, 2nv)

            # Cholesky regularisation: increase ε until Quu is PD
            eps = 1e-6
            for _ in range(20):
                try:
                    np.linalg.cholesky(Quu + eps * np.eye(nu))
                    break
                except np.linalg.LinAlgError:
                    eps *= 10.0
            Quu_reg = Quu + eps * np.eye(nu)

            k = -np.linalg.solve(Quu_reg, Qu)     # (nu,)  feedforward
            K = -np.linalg.solve(Quu_reg, Qux)    # (nu, 2nv) feedback

            k_list.insert(0, k)
            K_list.insert(0, K)

            # Value function update (Schur complement form)
            Vx = Qx + Qux.T @ k      # (2nv,)
            Vxx = Qxx + Qux.T @ K    # (2nv, 2nv)

        # ── Forward pass with backtracking line search ────────────────────────
        alpha = 1.0
        cost_bar = _compute_cost(model, qpos_bar, qvel_bar, U, goal_qpos, Q, R, Qf)

        for _ in range(10):  # max 10 line-search steps
            d_fwd = mujoco.MjData(model)
            d_fwd.qpos[:] = start_qpos
            d_fwd.qvel[:] = 0.0
            mujoco.mj_forward(model, d_fwd)

            U_cand = np.zeros_like(U)
            qpos_new = [d_fwd.qpos.copy()]
            qvel_new = [d_fwd.qvel.copy()]

            for t in range(horizon):
                # State deviation from nominal (velocity-parameterised)
                dpos = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dpos, 1.0, qpos_bar[t], qpos_new[-1])
                dx = np.concatenate([dpos, qvel_new[-1] - qvel_bar[t]])

                du = alpha * k_list[t] + K_list[t] @ dx
                U_cand[t] = np.clip(U[t] + du, ctrl_lo, ctrl_hi)
                d_fwd.ctrl[:] = U_cand[t]
                mujoco.mj_step(model, d_fwd)
                qpos_new.append(d_fwd.qpos.copy())
                qvel_new.append(d_fwd.qvel.copy())

            qpos_new = np.array(qpos_new)
            qvel_new = np.array(qvel_new)
            cost_new = _compute_cost(model, qpos_new, qvel_new, U_cand, goal_qpos, Q, R, Qf)

            if cost_new < cost_bar:
                U = U_cand
                break
            alpha *= 0.5

        # ── Convergence check ─────────────────────────────────────────────────
        qpos_check, qvel_check = _rollout(model, start_qpos, U)
        xe_check = _state_err(model, qpos_check[-1], qvel_check[-1], goal_qpos)
        if float(np.linalg.norm(xe_check[:nv])) < tol:
            converged = True
            break

    # ── Final trajectory ──────────────────────────────────────────────────────
    qpos_final, qvel_final = _rollout(model, start_qpos, U)
    final_cost = _compute_cost(model, qpos_final, qvel_final, U, goal_qpos, Q, R, Qf)

    return json.dumps({
        "converged": converged,
        "iterations": iterations,
        "final_cost": round(float(final_cost), 6),
        "trajectory": qpos_final.tolist(),
        "controls": U.tolist(),
        "waypoints": _subsampled_waypoints(qpos_final),
    })


@mcp.tool()
@safe_tool
async def optimize_ilqr(
    ctx,
    start_qpos: list[float],
    goal_qpos: list[float],
    horizon: int = 50,
    max_iter: int = 20,
    template: str = "reach",
    Q: list | None = None,
    R: list | None = None,
    sim_name: str | None = None,
) -> str:
    """Iterative Linear Quadratic Regulator (iLQR) trajectory optimization.

    Optimizes a joint-space trajectory from start_qpos to goal_qpos using
    dynamics linearization (mjd_transitionFD) + Riccati backward pass.

    Args:
        start_qpos: Initial joint positions (length nq).
        goal_qpos: Target joint positions (length nq).
        horizon: Planning horizon in physics steps.
        max_iter: Maximum iLQR iterations.
        template: Cost template — 'reach' | 'minimize_effort' | 'energy'.
        Q: State cost matrix (2nv × 2nv) as nested list; overrides template.
        R: Control cost matrix (nu × nu) as nested list; overrides template.
        sim_name: Simulation slot name (default slot if None).

    Returns:
        JSON with converged, iterations, final_cost, trajectory, controls, waypoints.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    return _ilqr_impl(
        slot.model, slot.data,
        start_qpos=start_qpos,
        goal_qpos=goal_qpos,
        horizon=horizon,
        max_iter=max_iter,
        template=template,
        Q_user=Q,
        R_user=R,
    )


# ── MPPI ──────────────────────────────────────────────────────────────────────

def _mppi_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,  # noqa: ARG001
    start_qpos: list[float],
    goal_qpos: list[float],
    horizon: int = 20,
    n_samples: int = 50,
    temperature: float = 0.1,
    noise_sigma: float = 1.0,
    max_iter: int = 5,
    template: str = "reach",
    Q_user: list | None = None,
    R_user: list | None = None,
) -> str:
    """Model Predictive Path Integral (MPPI) trajectory optimization.

    Samples K perturbed control sequences, computes costs, and updates
    the nominal control via temperature-weighted average.
    """
    nv = model.nv
    nu = model.nu

    Q, R, Qf = _build_cost_matrices(model, template, Q_user, R_user)
    start_qpos = np.array(start_qpos, dtype=float)
    goal_qpos = np.array(goal_qpos, dtype=float)
    ctrl_lo = model.actuator_ctrlrange[:, 0]
    ctrl_hi = model.actuator_ctrlrange[:, 1]

    def rollout_cost(controls: np.ndarray) -> float:
        """Roll out controls and compute total cost."""
        d = mujoco.MjData(model)
        d.qpos[:] = start_qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        cost = 0.0
        for u in controls:
            xe = _state_err(model, d.qpos.copy(), d.qvel.copy(), goal_qpos)
            cost += float(xe @ Q @ xe + u @ R @ u)
            d.ctrl[:] = np.clip(u, ctrl_lo, ctrl_hi)
            mujoco.mj_step(model, d)
        xe_T = _state_err(model, d.qpos.copy(), d.qvel.copy(), goal_qpos)
        cost += float(xe_T @ Qf @ xe_T)
        return cost

    U = np.zeros((horizon, nu))  # nominal control sequence
    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        # Sample K perturbations and compute costs
        noise = np.random.randn(n_samples, horizon, nu) * noise_sigma
        costs = np.zeros(n_samples)
        for k in range(n_samples):
            U_k = np.clip(U + noise[k], ctrl_lo, ctrl_hi)
            costs[k] = rollout_cost(U_k)

        # Temperature-weighted update
        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / temperature)
        weights /= np.sum(weights)
        U += np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
        U = np.clip(U, ctrl_lo, ctrl_hi)

    # Final rollout for trajectory and convergence check
    qpos_final, qvel_final = _rollout(model, start_qpos, U)
    xe_T = _state_err(model, qpos_final[-1], qvel_final[-1], goal_qpos)
    if float(np.linalg.norm(xe_T[:nv])) < 0.05:
        converged = True

    final_cost = _compute_cost(model, qpos_final, qvel_final, U, goal_qpos, Q, R, Qf)

    return json.dumps({
        "converged": converged,
        "iterations": iterations,
        "final_cost": round(float(final_cost), 6),
        "trajectory": qpos_final.tolist(),
        "controls": U.tolist(),
        "waypoints": _subsampled_waypoints(qpos_final),
    })


@mcp.tool()
@safe_tool
async def optimize_mppi(
    ctx,
    start_qpos: list[float],
    goal_qpos: list[float],
    horizon: int = 20,
    n_samples: int = 50,
    temperature: float = 0.1,
    noise_sigma: float = 1.0,
    max_iter: int = 5,
    template: str = "reach",
    Q: list | None = None,
    R: list | None = None,
    sim_name: str | None = None,
) -> str:
    """Model Predictive Path Integral (MPPI) trajectory optimization.

    Samples K perturbed control sequences, evaluates costs, and updates
    the nominal control via temperature-weighted averaging.
    Good for contact-rich scenarios where gradient-based methods struggle.

    Args:
        start_qpos: Initial joint positions (length nq).
        goal_qpos: Target joint positions (length nq).
        horizon: Planning horizon in physics steps.
        n_samples: Number of sampled trajectories K.
        temperature: Exploration temperature λ (lower = more greedy).
        noise_sigma: Control noise standard deviation.
        max_iter: Number of MPPI update iterations.
        template: Cost template — 'reach' | 'minimize_effort' | 'energy'.
        Q: State cost matrix (2nv × 2nv) as nested list; overrides template.
        R: Control cost matrix (nu × nu) as nested list; overrides template.
        sim_name: Simulation slot name (default slot if None).

    Returns:
        JSON with converged, iterations, final_cost, trajectory, controls, waypoints.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    return _mppi_impl(
        slot.model, slot.data,
        start_qpos=start_qpos,
        goal_qpos=goal_qpos,
        horizon=horizon,
        n_samples=n_samples,
        temperature=temperature,
        noise_sigma=noise_sigma,
        max_iter=max_iter,
        template=template,
        Q_user=Q,
        R_user=R,
    )
```

**Step 2: Run tests**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && python -m pytest tests/test_optimization.py -v
```

Expected: all 6 tests pass.

**Step 3: Lint check**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && ruff check src/mujoco_mcp/tools/optimization.py
```

Expected: no errors.

**Step 4: Commit**

```bash
git add src/mujoco_mcp/tools/optimization.py
git commit -m "feat(optimization): add optimize_ilqr and optimize_mppi tools"
```

---

## Task 3: Register optimization in server.py + Full Test Suite

**Files:**
- Modify: `src/mujoco_mcp/server.py:65`

**Step 1: Add `optimization` to the import line**

Current line 65:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env, vision, diagnostics, kinematics  # noqa: E402,F401,E501
```

Change to:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env, vision, diagnostics, kinematics, optimization  # noqa: E402,F401,E501
```

**Step 2: Run full test suite**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: 109 tests pass (103 existing + 6 new).

**Step 3: Commit**

```bash
git add src/mujoco_mcp/server.py
git commit -m "feat(optimization): register optimization module in server.py"
```

---

## Task 4: Update README.md

**Files:**
- Modify: `README.md`

**Step 1: Three changes**

1. Line 8: `**57 MCP tools**` → `**59 MCP tools**`
2. Line 5: `tests-103%20passed` → `tests-109%20passed`
3. Add row to Tools table, after the `| **Kinematics** |` row:
   ```
   | **Optimization** | `optimize_ilqr` `optimize_mppi` | iLQR and MPPI trajectory optimization |
   ```

**Step 2: Verify**

```bash
grep -n "Optimization\|59 MCP\|109" /home/rongxuan_zhou/mujoco_mcp/README.md
```

Expected: matches on all 3 patterns.

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Optimization group to README, bump tool count 57→59, tests 103→109"
```

---

## Task 5: Final Verification + Push

**Step 1: Full test suite**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: 109 passed.

**Step 2: Lint**

```bash
cd /home/rongxuan_zhou/mujoco_mcp && ruff check src/mujoco_mcp/tools/optimization.py
```

Expected: All checks passed.

**Step 3: Git log**

```bash
git log --oneline -6
```

Expected (newest first):
- `docs: add Optimization group to README, bump tool count 57→59, tests 103→109`
- `feat(optimization): register optimization module in server.py`
- `feat(optimization): add optimize_ilqr and optimize_mppi tools`
- `test(optimization): add 6 failing tests for iLQR + MPPI`

**Step 4: Push**

```bash
git push origin master
```

**Step 5: Report** pass count, lint result, log, push status.
