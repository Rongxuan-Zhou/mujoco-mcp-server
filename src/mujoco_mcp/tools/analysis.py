"""Analysis tools (tools 11–16): contacts, jacobian, derivatives, sensors, energy, forces."""

import json
import mujoco
import numpy as np
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_name, list_named, ensure_energy, restore_energy, contact_geoms
from ..constants import JACOBIAN_NV_THRESHOLD
from . import safe_tool


@mcp.tool()
@safe_tool
async def analyze_contacts(
    ctx: Context,
    max_contacts: int = 20,
    sim_name: str | None = None,
) -> str:
    """Active contact pairs: geom names, positions, forces, penetration depth.

    Returns up to max_contacts entries sorted by contact index.
    Call sim_forward first if state was recently changed.

    Args:
        max_contacts: Maximum number of contacts to return (default 20).
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"n_contacts": int,
               "contacts": [{"geom1": str, "geom2": str,
                              "pos": [x,y,z], "dist": float,
                              "normal_force": float, ...}]}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    contacts = []
    for i in range(min(d.ncon, max_contacts)):
        c = d.contact[i]
        gid1, gid2 = contact_geoms(c)
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid1) or f"geom_{gid1}"
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid2) or f"geom_{gid2}"
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        contacts.append({
            "geom1":        g1,
            "geom2":        g2,
            "pos":          c.pos.tolist(),
            "dist":         float(c.dist),
            "normal_force": float(force[0]),
            "friction_force": force[1:3].tolist(),
            "condim":       int(c.dim),
        })
    return json.dumps({"n_contacts": d.ncon, "contacts": contacts}, indent=2)


@mcp.tool()
@safe_tool
async def compute_jacobian(
    ctx: Context,
    target: str,
    target_type: str = "site",
    full_matrix: bool = False,
    sim_name: str | None = None,
) -> str:
    """End-effector Jacobian (6×nv), SVD, manipulability, condition number.

    Args:
        target: Name of the target site, body, or geom.
        target_type: 'site' | 'body' | 'geom'.
        full_matrix: Return full jacp/jacr matrices even for large models (nv > 50).
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data

    type_map = {
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
    }
    fn_map = {
        "site": mujoco.mj_jacSite,
        "body": mujoco.mj_jacBody,
        "geom": mujoco.mj_jacGeom,
    }
    if target_type not in type_map:
        return json.dumps({"error": "target_type must be 'site', 'body', or 'geom'"})

    oid = resolve_name(m, type_map[target_type], target, target_type)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    fn_map[target_type](m, d, jacp, jacr, oid)

    J = np.vstack([jacp, jacr])
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-10))
    active_sv = sv[:rank] if rank > 0 else sv[:1]
    manip = float(np.prod(active_sv)) if rank > 0 else 0.0
    cond = float(sv[0] / active_sv[-1]) if rank > 0 else float("inf")

    pos_map = {
        "site": lambda: d.site(target).xpos,
        "body": lambda: d.body(target).xpos,
        "geom": lambda: d.geom(target).xpos,
    }

    result = {
        "target":          target,
        "target_type":     target_type,
        "nv":              m.nv,
        "position":        pos_map[target_type]().tolist(),
        "rank":            rank,
        "singular_values": sv.tolist(),
        "manipulability":  manip,
        "condition_number": cond,
    }
    if m.nv <= JACOBIAN_NV_THRESHOLD or full_matrix:
        result["jacp"] = jacp.tolist()
        result["jacr"] = jacr.tolist()
    else:
        result["note"] = (
            f"Matrices omitted (nv={m.nv} > {JACOBIAN_NV_THRESHOLD}). "
            "Set full_matrix=True to include them."
        )
    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def compute_derivatives(
    ctx: Context,
    eps: float = 1e-6,
    sim_name: str | None = None,
) -> str:
    """Linearized discrete-time dynamics: x_{t+1} ≈ A x_t + B u_t.

    Uses mjd_transitionFD (finite-difference Jacobians).
    Also returns C and D matrices if sensors are present.
    Reports eigenvalue magnitudes of A for stability analysis.
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    nv, nu, ns = m.nv, m.nu, m.nsensordata

    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    C = np.zeros((ns, 2 * nv)) if ns else None
    D = np.zeros((ns, nu)) if ns else None

    mujoco.mjd_transitionFD(m, d, eps, True, A, B, C, D)

    eigs = np.abs(np.linalg.eigvals(A))
    result = {
        "A_shape":            [2 * nv, 2 * nv],
        "B_shape":            [2 * nv, nu],
        "max_eig_magnitude":  float(eigs.max()),
        "discrete_stable":    bool(eigs.max() < 1.0),
    }
    if C is not None:
        result["C_shape"] = [ns, 2 * nv]
        result["D_shape"] = [ns, nu]

    # Include matrices only for small models
    if 2 * nv <= 40:
        result["A"] = A.tolist()
        result["B"] = B.tolist()
        if C is not None:
            result["C"] = C.tolist()
            result["D"] = D.tolist()
    else:
        result["note"] = f"Matrices omitted (state_dim={2*nv} > 40). Access via export_csv."

    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def read_sensors(
    ctx: Context,
    sensor_names: list[str] | None = None,
    sim_name: str | None = None,
) -> str:
    """Read current sensor values by name, or all sensors if no names given.

    Returns a dict mapping sensor name → list of values (dim ≥ 1).

    Args:
        sensor_names: List of sensor names to read. ``None`` reads all sensors.
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {sensor_name: [float, ...], ...}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    result = {}

    if sensor_names:
        for name in sensor_names:
            sid = resolve_name(m, mujoco.mjtObj.mjOBJ_SENSOR, name, "sensor")
            adr = m.sensor_adr[sid]
            dim = m.sensor_dim[sid]
            result[name] = d.sensordata[adr: adr + dim].tolist()
    else:
        for sid in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if name:
                adr = m.sensor_adr[sid]
                dim = m.sensor_dim[sid]
                result[name] = d.sensordata[adr: adr + dim].tolist()

    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def analyze_energy(ctx: Context, sim_name: str | None = None) -> str:
    """Current potential energy, kinetic energy, and total mechanical energy.

    Automatically enables the mjENBL_ENERGY flag for the duration of the call
    if not already set (restores original state afterwards). This ensures correct
    non-zero values even when the model XML does not set the energy enable flag.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"potential": float, "kinetic": float, "total": float,
               "time": float, "energy_enabled": true}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    was_enabled = ensure_energy(m)
    mujoco.mj_forward(m, d)
    result = {
        "potential":      float(d.energy[0]),
        "kinetic":        float(d.energy[1]),
        "total":          float(d.energy[0] + d.energy[1]),
        "time":           d.time,
        "energy_enabled": True,
    }
    restore_energy(m, was_enabled)
    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def analyze_forces(ctx: Context, sim_name: str | None = None) -> str:
    """Joint-space force decomposition: applied, constraint, passive, bias, actuator.

    All vectors are in generalized (joint) coordinates, length nv.
    qacc is the resulting joint acceleration.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"qfrc_applied": [...], "qfrc_constraint": [...],
               "qfrc_passive": [...], "qfrc_bias": [...],
               "qfrc_actuator": [...], "qacc": [...]}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    result = {
        "qfrc_applied":    d.qfrc_applied.tolist(),
        "qfrc_constraint": d.qfrc_constraint.tolist(),
        "qfrc_passive":    d.qfrc_passive.tolist(),
        "qfrc_bias":       d.qfrc_bias.tolist(),
        "qfrc_actuator":   d.qfrc_actuator.tolist(),
        "qacc":            d.qacc.tolist(),
    }
    if m.nu > 0:
        result["actuator_force"] = d.actuator_force.tolist()
    return json.dumps(result, indent=2)
