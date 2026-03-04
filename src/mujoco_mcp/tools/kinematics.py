"""Kinematics tool group — Damped Least Squares inverse kinematics."""
from __future__ import annotations

import json
import numpy as np
import mujoco

from .._registry import mcp
from . import safe_tool


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),   2*(x*z + w*y)],
        [  2*(x*y + w*z), 1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [  2*(x*z - w*y),   2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def _rotation_error(R_cur: np.ndarray, R_tgt: np.ndarray) -> np.ndarray:
    """3D orientation error (axis-angle) from current to target rotation matrix."""
    R_err = R_tgt @ R_cur.T
    return 0.5 * np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])


def _build_dof_mask(model: mujoco.MjModel, joint_names: list[str] | None) -> np.ndarray:
    """Boolean mask over model.nv — True for DOFs that participate in IK."""
    if joint_names is None:
        return np.ones(model.nv, dtype=bool)
    mask = np.zeros(model.nv, dtype=bool)
    ndof_for_type = {
        mujoco.mjtJoint.mjJNT_FREE: 6,
        mujoco.mjtJoint.mjJNT_BALL: 3,
        mujoco.mjtJoint.mjJNT_SLIDE: 1,
        mujoco.mjtJoint.mjJNT_HINGE: 1,
    }
    for jname in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            raise ValueError(f"Joint '{jname}' not found in model")
        dof_adr = int(model.jnt_dofadr[jid])
        ndof = ndof_for_type.get(int(model.jnt_type[jid]), 1)
        mask[dof_adr:dof_adr + ndof] = True
    return mask


def _clamp_joint_limits(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Clamp qpos to joint ranges for all limited hinge/slide joints."""
    for jid in range(model.njnt):
        if not model.jnt_limited[jid]:
            continue
        jtype = int(model.jnt_type[jid])
        if jtype == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if jtype == mujoco.mjtJoint.mjJNT_BALL:
            continue
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.jnt_range[jid]
        data.qpos[qadr] = float(np.clip(data.qpos[qadr], lo, hi))


def solve_ik_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_name: str,
    target_pos: list[float],
    target_quat: list[float] | None = None,
    joint_names: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 1e-3,
) -> str:
    """Damped Least Squares IK. Modifies data.qpos in place. Returns JSON result."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model")

    target_pos_arr = np.array(target_pos, dtype=float)
    target_mat = _quat_to_mat(np.array(target_quat, dtype=float)) if target_quat is not None else None
    use_orientation = target_mat is not None
    err_dim = 6 if use_orientation else 3

    dof_mask = _build_dof_mask(model, joint_names)
    dof_indices = np.where(dof_mask)[0]

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    converged = False
    iterations = 0

    for step in range(max_iter):
        iterations = step + 1
        mujoco.mj_forward(model, data)

        site_pos = data.site_xpos[site_id].copy()
        err_pos = target_pos_arr - site_pos

        if use_orientation:
            site_mat = data.site_xmat[site_id].reshape(3, 3).copy()
            err_rot = _rotation_error(site_mat, target_mat)
            err = np.concatenate([err_pos, err_rot])
        else:
            err = err_pos

        if float(np.linalg.norm(err_pos)) < tol:
            converged = True
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        J_full = np.vstack([jacp, jacr]) if use_orientation else jacp.copy()
        J = J_full[:, dof_mask]

        lam2 = damping ** 2
        A = J @ J.T + lam2 * np.eye(err_dim)
        dq = J.T @ np.linalg.solve(A, err)

        data.qpos[dof_indices] += dq
        _clamp_joint_limits(model, data)

    mujoco.mj_forward(model, data)
    final_site_pos = data.site_xpos[site_id].copy()
    final_pos_error = float(np.linalg.norm(target_pos_arr - final_site_pos))
    final_ori_error = None
    if use_orientation:
        final_site_mat = data.site_xmat[site_id].reshape(3, 3).copy()
        final_ori_error = float(np.linalg.norm(_rotation_error(final_site_mat, target_mat)))

    return json.dumps({
        "converged": converged,
        "iterations": iterations,
        "pos_error": final_pos_error,
        "ori_error": final_ori_error,
        "site_pos": final_site_pos.tolist(),
        "qpos": data.qpos.tolist(),
    })


@mcp.tool()
@safe_tool
async def solve_ik(
    ctx,
    site_name: str,
    target_pos: list[float],
    target_quat: list[float] | None = None,
    joint_names: list[str] | None = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    damping: float = 1e-3,
    sim_name: str | None = None,
) -> str:
    """Solve inverse kinematics using Damped Least Squares (DLS).

    Iteratively moves the named site to target_pos (and optionally target_quat)
    by adjusting qpos. Result is written into the sim slot's data.qpos.

    Args:
        site_name: Name of the end-effector site in the model.
        target_pos: Target position [x, y, z] in world frame.
        target_quat: Target orientation [w, x, y, z] (optional).
        joint_names: List of joint names to move. None = all joints participate.
        max_iter: Maximum DLS iterations (default 100).
        tol: Convergence threshold in metres (default 1e-4).
        damping: DLS damping coefficient lambda (default 1e-3).
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"converged": bool, "iterations": int, "pos_error": float,
               "ori_error": float|null, "site_pos": [x,y,z], "qpos": [...]}
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    return solve_ik_impl(
        slot.model, slot.data,
        site_name=site_name,
        target_pos=target_pos,
        target_quat=target_quat,
        joint_names=joint_names,
        max_iter=max_iter,
        tol=tol,
        damping=damping,
    )
