# Trajectory Optimization Tool Group — Design

**Date:** 2026-03-04

## Background

Trajectory optimization is a core primitive for robot control research. The current codebase
exposes `compute_derivatives` (which wraps `mjd_transitionFD` to produce linearized A, B
matrices) and `plan_trajectory` (min-jerk interpolation), but has no optimizer. This design
adds two complementary algorithms: iLQR for gradient-based smooth-system optimization, and
MPPI for sampling-based contact-rich optimization.

## Scope

Two new MCP tools in a new `Optimization` tool group, one new file
`src/mujoco_mcp/tools/optimization.py`.

## Tool Specifications

### `optimize_ilqr`

```
optimize_ilqr(
    start_qpos: list[float],        # initial joint positions (required)
    goal_qpos: list[float],         # target joint positions (required)
    horizon: int = 50,              # planning horizon (steps)
    max_iter: int = 20,             # outer iLQR iterations
    template: str = "reach",        # "reach" | "minimize_effort" | "energy"
    Q: list[list[float]] | None,    # state cost matrix (2nv × 2nv); overrides template
    R: list[list[float]] | None,    # control cost matrix (nu × nu); overrides template
    sim_name: str | None = None,
) -> str
```

### `optimize_mppi`

```
optimize_mppi(
    start_qpos: list[float],
    goal_qpos: list[float],
    horizon: int = 20,              # planning horizon (steps) — smaller default for speed
    n_samples: int = 50,            # K sampled trajectories — smaller default for speed
    temperature: float = 0.01,      # λ — exploration temperature
    noise_sigma: float = 0.1,       # control noise std dev
    max_iter: int = 5,              # outer MPPI iterations — smaller default for speed
    template: str = "reach",
    Q: list[list[float]] | None,
    R: list[list[float]] | None,
    sim_name: str | None = None,
) -> str
```

### Unified Return Format

```json
{
  "converged": true,
  "iterations": 12,
  "final_cost": 0.023,
  "trajectory": [[q0, q1, ...], ...],
  "controls": [[u0, u1, ...], ...],
  "waypoints": [[q0, q1, ...], ...]
}
```

- `trajectory`: full optimized qpos sequence (horizon × nq)
- `controls`: optimized control sequence (horizon × nu)
- `waypoints`: subsampled key poses (≤10 evenly spaced), ready to feed into `plan_trajectory`
  sequentially for piecewise execution. Does NOT discard optimization results.

## Cost Function Templates

| Template | Q diagonal | R diagonal | Use case |
|---|---|---|---|
| `"reach"` | pos error ×100, vel ×1 | ctrl ×0.01 | End-effector reaching |
| `"minimize_effort"` | pos ×10, vel ×1 | ctrl ×1.0 | Energy-efficient motion |
| `"energy"` | pos ×10, vel ×10 | ctrl ×0.1 | Kinetic+potential minimization |

User-supplied Q/R fully overrides the template. Q shape must be (2nv × 2nv),
R shape must be (nu × nu).

## Algorithm Details

### iLQR — State Representation

**Key:** use velocity-parameterized state `x = [δqpos_vel; qvel]` of dimension 2nv, not
qpos-space. This avoids the nq ≠ nv problem caused by quaternion free joints.

- `A, B = mjd_transitionFD(model, data, eps)` — already returns (2nv × 2nv) and (2nv × nu)
- Position error: `mj_differentiatePos(model, qpos_cur, qpos_goal)` → nv-vector (handles
  quaternion subtraction correctly)

### iLQR — Algorithm

```
Initialize: U = zeros(horizon, nu), X = rollout(start_qpos, U)

for iter in range(max_iter):
    # Linearize along trajectory
    A[t], B[t] = mjd_transitionFD at each (X[t], U[t])

    # Backward Riccati pass
    Vx, Vxx = terminal cost gradients (Qf)
    for t = horizon-1 down to 0:
        Qx = lx + A[t]ᵀ Vx
        Qu = lu + B[t]ᵀ Vx
        Qxx = lxx + A[t]ᵀ Vxx A[t]
        Quu = luu + B[t]ᵀ Vxx B[t]
        Qux = B[t]ᵀ Vxx A[t]

        # Regularization for numerical stability
        Quu_reg = Quu + ε·I  (increase ε until Cholesky succeeds)

        k[t] = -Quu_reg⁻¹ Qu      # feedforward
        K[t] = -Quu_reg⁻¹ Qux     # feedback

    # Forward pass with line search (α backtracking)
    X_new, U_new, cost_new = forward_pass(X, U, k, K, α=1.0)
    if cost_new < cost: accept, update X/U; if Δcost < tol: converged=True; break
```

### MPPI — Algorithm (sequential, no ProcessPoolExecutor)

```
Initialize: U = zeros(horizon, nu)   # nominal control sequence

for iter in range(max_iter):
    noise = np.random.randn(n_samples, horizon, nu) * noise_sigma
    costs = zeros(n_samples)

    for k in range(n_samples):
        U_k = U + noise[k]
        costs[k] = rollout_cost(start_qpos, U_k, goal_qpos)

    # Temperature-weighted update
    beta = min(costs)
    weights = exp(-(costs - beta) / temperature)
    weights /= sum(weights)
    U += sum(weights[:, None, None] * noise, axis=0)

    if rollout_cost(start_qpos, U, goal_qpos) < tol: converged=True; break
```

Sequential (no multiprocessing) avoids MuJoCo model/data pickling issues and is fast enough
for the default parameters (50 samples × 20 steps × 5 iters = 5000 mj_step calls ≈ 0.5s).

## Architecture

- **New file:** `src/mujoco_mcp/tools/optimization.py`
- **Registration:** Add `optimization` to the import line in `server.py`
- **Internal helpers:**
  - `_build_cost_matrices(model, template, Q_user, R_user)` → (Q, R, Qf)
  - `_rollout(model, data_copy, start_qpos, controls)` → (X, total_cost)
  - `_ilqr_impl(model, data, ...)` → JSON string (sync, directly testable)
  - `_mppi_impl(model, data, ...)` → JSON string (sync, directly testable)
- **No new dependencies:** numpy only (already available)
- **Error handling:** `@mcp.tool()` outer, `@safe_tool` inner; raises `ValueError` for
  unknown template, mismatched Q/R shape, or empty model

## Testing (6 tests)

1. **iLQR: 1-DOF slider converges** — start=0, goal=0.5 → `converged=True`,
   `trajectory[-1][0]` close to goal within 0.05
2. **iLQR: custom Q/R matrices accepted** — pass explicit Q=100·I, R=I; result differs
   from `template="reach"` default; no exception raised
3. **iLQR: `waypoints` field present and subsampled** — len(waypoints) ≤ 10,
   waypoints[0] ≈ start_qpos, waypoints[-1] ≈ trajectory[-1]
4. **MPPI: 1-DOF slider reaches goal** — `np.random.seed(42)`; final qpos within 0.1 of goal
5. **MPPI: more samples improve cost** — `np.random.seed(42)`, same model; n_samples=100
   gives lower `final_cost` than n_samples=5
6. **Invalid template raises ValueError** — `template="unknown"` → ValueError with message
   containing "template"

## README Update

Add to Tools table:
```
| **Optimization** | `optimize_ilqr` `optimize_mppi` | iLQR and MPPI trajectory optimization |
```

Tool count: 57 → 59.

## Future Work (out of scope)

- Online MPC rolling horizon (`run_mpc` tool)
- Null-space optimization for redundant arms
- Collision-aware iLQR (augmented Lagrangian contact constraints)
- GPU-parallel MPPI via CuPy/JAX
