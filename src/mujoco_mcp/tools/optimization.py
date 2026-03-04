"""Optimization tool group — iLQR and MPPI trajectory optimization."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import mujoco
from mcp.server.fastmcp import Context

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
        if R_user is None:
            raise ValueError("R must be provided when Q is provided")
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


def _state_err(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    goal_qpos: np.ndarray,
) -> np.ndarray:
    """Velocity-parameterized state error vector (length 2nv).

    Position part: mj_differentiatePos(goal_qpos, qpos) = qpos - goal_qpos in vel coords.
    Velocity part: qvel (we want zero terminal velocity).
    """
    dq = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, dq, 1.0, goal_qpos, qpos)
    return np.concatenate([dq, qvel])


def _apply_ctrl(u: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Clip control to actuator range, respecting ctrllimited flag.

    Handles both 1D (nu,) single-step and 2D (horizon, nu) batch arrays.
    Actuators without ctrllimited=True have ctrlrange=[0,0] by default;
    clipping them unconditionally would silently zero all controls.
    """
    ctrl = u.copy()
    limited = model.actuator_ctrllimited.astype(bool)
    if not np.any(limited):
        return ctrl
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    if ctrl.ndim == 1:
        ctrl[limited] = np.clip(ctrl[limited], lo[limited], hi[limited])
    else:
        ctrl[:, limited] = np.clip(ctrl[:, limited], lo[limited], hi[limited])
    return ctrl


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
    d = mujoco.MjData(model)
    d.qpos[:] = start_qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)

    qpos_list = [d.qpos.copy()]
    qvel_list = [d.qvel.copy()]
    for u in controls:
        d.ctrl[:] = _apply_ctrl(u, model)
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
    data: mujoco.MjData,  # noqa: ARG001
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
    # tol=0.01 m: 1 cm is achievable with discrete dt=0.01 s; 1e-4 m is unreachable
    # due to control-limit discretisation in the slider test model.
    tol = 0.01

    Q, R, Qf = _build_cost_matrices(model, template, Q_user, R_user)
    start_qpos = np.array(start_qpos, dtype=float)
    goal_qpos = np.array(goal_qpos, dtype=float)

    U = np.zeros((horizon, nu))
    converged = False
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        # Rollout
        qpos_bar, qvel_bar = _rollout(model, start_qpos, U)

        # Linearize along trajectory
        A_list, B_list = [], []
        for t in range(horizon):
            d = mujoco.MjData(model)
            d.qpos[:] = qpos_bar[t]
            d.qvel[:] = qvel_bar[t]
            d.ctrl[:] = _apply_ctrl(U[t], model)
            mujoco.mj_forward(model, d)
            A = np.zeros((2 * nv, 2 * nv))
            B = np.zeros((2 * nv, nu))
            mujoco.mjd_transitionFD(model, d, 1e-6, True, A, B, None, None)
            A_list.append(A)
            B_list.append(B)

        # Backward Riccati pass
        xe_T = _state_err(model, qpos_bar[-1], qvel_bar[-1], goal_qpos)
        Vx = 2.0 * Qf @ xe_T
        Vxx = 2.0 * Qf

        k_list: list[np.ndarray] = []
        K_list: list[np.ndarray] = []

        for t in reversed(range(horizon)):
            xe_t = _state_err(model, qpos_bar[t], qvel_bar[t], goal_qpos)
            A_t, B_t = A_list[t], B_list[t]

            lx = 2.0 * Q @ xe_t
            lu = 2.0 * R @ U[t]

            Qx = lx + A_t.T @ Vx
            Qu = lu + B_t.T @ Vx
            Qxx = 2.0 * Q + A_t.T @ Vxx @ A_t
            Quu = 2.0 * R + B_t.T @ Vxx @ B_t
            Qux = B_t.T @ Vxx @ A_t

            # Cholesky regularisation
            eps = 1e-6
            for _ in range(20):
                try:
                    np.linalg.cholesky(Quu + eps * np.eye(nu))
                    break
                except np.linalg.LinAlgError:
                    eps *= 10.0
            Quu_reg = Quu + eps * np.eye(nu)

            k = -np.linalg.solve(Quu_reg, Qu)
            K = -np.linalg.solve(Quu_reg, Qux)

            k_list.insert(0, k)
            K_list.insert(0, K)

            # Value function update (Schur complement form)
            Vx = Qx + Qux.T @ k
            Vxx = Qxx + Qux.T @ K

        # Forward pass with backtracking line search
        cost_bar = _compute_cost(model, qpos_bar, qvel_bar, U, goal_qpos, Q, R, Qf)
        alpha = 1.0

        for _ in range(10):
            d_fwd = mujoco.MjData(model)
            d_fwd.qpos[:] = start_qpos
            d_fwd.qvel[:] = 0.0
            mujoco.mj_forward(model, d_fwd)

            U_cand = np.zeros_like(U)
            qpos_new = [d_fwd.qpos.copy()]
            qvel_new = [d_fwd.qvel.copy()]

            for t in range(horizon):
                dpos = np.zeros(nv)
                mujoco.mj_differentiatePos(model, dpos, 1.0, qpos_bar[t], qpos_new[-1])
                dx = np.concatenate([dpos, qvel_new[-1] - qvel_bar[t]])

                du = alpha * k_list[t] + K_list[t] @ dx
                U_cand[t] = _apply_ctrl(U[t] + du, model)
                d_fwd.ctrl[:] = U_cand[t]
                mujoco.mj_step(model, d_fwd)
                qpos_new.append(d_fwd.qpos.copy())
                qvel_new.append(d_fwd.qvel.copy())

            qpos_new_arr = np.array(qpos_new)
            qvel_new_arr = np.array(qvel_new)
            cost_new = _compute_cost(
                model, qpos_new_arr, qvel_new_arr, U_cand, goal_qpos, Q, R, Qf
            )

            if cost_new < cost_bar:
                U = U_cand
                break
            alpha *= 0.5

        # Convergence check
        qpos_check, qvel_check = _rollout(model, start_qpos, U)
        xe_check = _state_err(model, qpos_check[-1], qvel_check[-1], goal_qpos)
        if float(np.linalg.norm(xe_check[:nv])) < tol:
            converged = True
            break

    # Final trajectory
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
    ctx: Context,
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
    await asyncio.sleep(0)
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
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    nv = model.nv
    nu = model.nu

    Q, R, Qf = _build_cost_matrices(model, template, Q_user, R_user)
    start_qpos = np.array(start_qpos, dtype=float)
    goal_qpos = np.array(goal_qpos, dtype=float)

    def rollout_cost(controls: np.ndarray) -> float:
        d = mujoco.MjData(model)
        d.qpos[:] = start_qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(model, d)
        cost = 0.0
        for u in controls:
            xe = _state_err(model, d.qpos.copy(), d.qvel.copy(), goal_qpos)
            cost += float(xe @ Q @ xe + u @ R @ u)
            d.ctrl[:] = _apply_ctrl(u, model)
            mujoco.mj_step(model, d)
        xe_T = _state_err(model, d.qpos.copy(), d.qvel.copy(), goal_qpos)
        cost += float(xe_T @ Qf @ xe_T)
        return cost

    U = np.zeros((horizon, nu))
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1

        noise = np.random.randn(n_samples, horizon, nu) * noise_sigma
        costs = np.zeros(n_samples)
        for k in range(n_samples):
            U_k = _apply_ctrl(U + noise[k], model)
            costs[k] = rollout_cost(U_k)

        beta = float(np.min(costs))
        weights = np.exp(-(costs - beta) / temperature)
        w_sum = float(np.sum(weights))
        if w_sum < 1e-300 or not np.isfinite(w_sum):
            weights = np.ones(n_samples) / n_samples
        else:
            weights /= w_sum
        U += np.sum(weights[:, np.newaxis, np.newaxis] * noise, axis=0)
        U = _apply_ctrl(U, model)

    # Final rollout
    qpos_final, qvel_final = _rollout(model, start_qpos, U)
    xe_T = _state_err(model, qpos_final[-1], qvel_final[-1], goal_qpos)
    converged = bool(float(np.linalg.norm(xe_T[:nv])) < 0.05)
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
    ctx: Context,
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
    await asyncio.sleep(0)
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
