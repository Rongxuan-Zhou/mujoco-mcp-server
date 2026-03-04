"""Diagnostics tool group — XML validation, model summary, contact params, instability detection."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
import mujoco
import numpy as np

from .._registry import mcp
from . import safe_tool


def validate_mjcf_impl(*, xml_path: str | None = None, xml_string: str | None = None) -> str:
    """Core logic for validate_mjcf — testable without MCP context."""
    if xml_path is not None and xml_string is not None:
        return json.dumps({"valid": False, "errors": [{"rule": "invalid_input", "element": "args", "message": "Provide xml_path or xml_string, not both"}], "warnings": []})

    if xml_path is None and xml_string is None:
        return json.dumps({"valid": False, "errors": [{"rule": "invalid_input", "element": "args", "message": "Provide xml_path or xml_string"}], "warnings": []})

    if xml_path is not None:
        with open(xml_path) as f:
            xml_string = f.read()

    errors: list[dict] = []
    warnings: list[dict] = []

    # --- Static checks via ElementTree ---
    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError as e:
        return json.dumps({"valid": False, "errors": [{"rule": "xml_parse", "element": "root", "message": str(e)}], "warnings": []})

    # Check for worldbody
    worldbody = root.find("worldbody")
    if worldbody is None:
        errors.append({"rule": "missing_worldbody", "element": "mujoco", "message": "No <worldbody> element found"})

    # Check for duplicate names within the same element type (MuJoCo uses per-type namespaces)
    names_seen: dict[tuple, str] = {}
    for elem in root.iter():
        name = elem.get("name")
        if name:
            key = (elem.tag, name)
            if key in names_seen:
                errors.append({
                    "rule": "duplicate_name",
                    "element": f"<{elem.tag} name='{name}'>",
                    "message": f"Duplicate name '{name}' on element <{elem.tag}>",
                })
            else:
                names_seen[key] = elem.tag

    # Check geom missing size (warn only — MuJoCo may infer from mesh)
    for geom in root.iter("geom"):
        gtype = geom.get("type", "sphere")
        if gtype in ("box", "cylinder", "capsule", "ellipsoid") and geom.get("size") is None and geom.get("fromto") is None:
            warnings.append({
                "rule": "geom_missing_size",
                "element": f"<geom name='{geom.get('name', '?')}' type='{gtype}'>",
                "message": f"Geom type '{gtype}' typically requires 'size' attribute",
            })

    # Check dangling actuator joint/tendon references
    joint_names = {j.get("name") for j in root.iter("joint") if j.get("name")}
    tendon_names = {
        t.get("name")
        for t in root.iter()
        if t.tag in ("spatial", "fixed") and t.get("name")
    }
    for act in root.iter():
        if act.tag in ("motor", "position", "velocity", "intvelocity", "damper"):
            ref_joint = act.get("joint")
            ref_tendon = act.get("tendon")
            if ref_joint and ref_joint not in joint_names:
                errors.append({
                    "rule": "dangling_actuator_joint",
                    "element": f"<{act.tag} name='{act.get('name', '?')}'>",
                    "message": f"Actuator references joint '{ref_joint}' which does not exist",
                })
            if ref_tendon and ref_tendon not in tendon_names:
                errors.append({
                    "rule": "dangling_actuator_tendon",
                    "element": f"<{act.tag} name='{act.get('name', '?')}'>",
                    "message": f"Actuator references tendon '{ref_tendon}' which does not exist",
                })

    # Only attempt MuJoCo compile if static checks passed
    if not errors:
        try:
            mujoco.MjModel.from_xml_string(xml_string)
        except Exception as e:
            errors.append({
                "rule": "mujoco_compile",
                "element": "model",
                "message": str(e),
            })

    return json.dumps({"valid": len(errors) == 0, "errors": errors, "warnings": warnings})


def model_summary_impl(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    """Core logic for model_summary — testable without MCP context."""
    summary: dict = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "njnt": model.njnt,
        "nsensor": model.nsensor,
        "ntendon": model.ntendon,
        "timestep": float(model.opt.timestep),
        "integrator": int(model.opt.integrator),
        "solver": int(model.opt.solver),
    }

    # Per-joint info (first 20)
    joints = []
    n_joints = min(model.njnt, 20)
    for i in range(n_joints):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or f"joint_{i}"
        jtype = int(model.jnt_type[i])
        jrange = model.jnt_range[i].tolist()
        damping_idx = int(model.jnt_dofadr[i])
        damping = float(model.dof_damping[damping_idx]) if damping_idx < model.nv else 0.0
        joints.append({"name": name, "type": jtype, "range": jrange, "damping": damping})
    summary["joints"] = joints

    # Per-actuator info (first 20)
    actuators = []
    n_act = min(model.nu, 20)
    for i in range(n_act):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"act_{i}"
        ctrl_range = model.actuator_ctrlrange[i].tolist()
        gear = float(model.actuator_gear[i, 0])
        actuators.append({"name": name, "ctrl_range": ctrl_range, "gear": gear})
    summary["actuators"] = actuators

    # Mass extremes (skip worldbody at index 0)
    if model.nbody > 1:
        masses = model.body_mass[1:]  # skip world
        body_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i + 1) or f"body_{i+1}"
            for i in range(len(masses))
        ]
        heaviest_idx = int(np.argmax(masses))
        lightest_idx = int(np.argmin(masses))
        if masses[heaviest_idx] > 0:
            summary["heaviest_body"] = {"name": body_names[heaviest_idx], "mass": float(masses[heaviest_idx])}
            summary["lightest_body"] = {"name": body_names[lightest_idx], "mass": float(masses[lightest_idx])}
            if masses[lightest_idx] > 0:
                summary["mass_ratio"] = float(masses[heaviest_idx] / masses[lightest_idx])
        else:
            summary["heaviest_body"] = None
            summary["lightest_body"] = None
    else:
        summary["heaviest_body"] = None
        summary["lightest_body"] = None

    return json.dumps(summary)


def suggest_contact_params_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1: str | None = None,
    geom2: str | None = None,
) -> str:
    """Core logic for suggest_contact_params — testable without MCP context."""
    timestep = float(model.opt.timestep)
    issues: list[dict] = []

    # If specific geoms are requested, use their params; otherwise use global average
    if geom1 is not None:
        gid1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        if gid1 == -1:
            raise ValueError(f"Geom '{geom1}' not found in model")
        avg_solref = model.geom_solref[gid1].tolist()
        avg_solimp = model.geom_solimp[gid1].tolist()
        avg_friction = model.geom_friction[gid1].tolist()
    elif model.ngeom > 0:
        avg_solref = model.geom_solref.mean(axis=0).tolist()
        avg_solimp = model.geom_solimp.mean(axis=0).tolist()
        avg_friction = model.geom_friction.mean(axis=0).tolist()
    else:
        avg_solref = [0.02, 1.0]
        avg_solimp = [0.9, 0.95, 0.001, 0.5, 2.0]
        avg_friction = [1.0, 0.005, 0.0001]

    current = {
        "solref": avg_solref,
        "solimp": avg_solimp,
        "friction": avg_friction,
        "timestep": timestep,
    }

    # Rule 1: solref[0] (time constant) must be >= 2 * timestep
    if avg_solref[0] < 2 * timestep:
        issues.append({
            "type": "solref_too_tight",
            "message": (
                f"solref[0]={avg_solref[0]:.4f} < 2×timestep={2*timestep:.4f}. "
                "Contact stiffness exceeds solver capability → oscillation or explosion."
            ),
            "fix": f"Set solref[0] >= {2*timestep:.4f}",
        })

    # Rule 2: solref[1] (damping ratio) < 1.0 → underdamped
    if avg_solref[1] < 1.0:
        issues.append({
            "type": "underdamped_contact",
            "message": f"solref[1]={avg_solref[1]:.3f} < 1.0 → underdamped contact → bouncing artifacts.",
            "fix": "Set solref[1] >= 1.0",
        })

    # Rule 3: friction = 0 likely missing
    if avg_friction[0] < 1e-6:
        issues.append({
            "type": "zero_friction",
            "message": "friction[0]=0.0 → objects will slide freely. Likely missing friction definition.",
            "fix": "Set friction to [1.0, 0.005, 0.0001] or similar",
        })

    # Rule 4: friction > 10 likely erroneous
    if avg_friction[0] > 10.0:
        issues.append({
            "type": "extreme_friction",
            "message": f"friction[0]={avg_friction[0]:.2f} > 10.0 → unusually high, likely erroneous.",
            "fix": "Typical friction is 0.5–2.0 for rigid bodies",
        })

    # Conservative preset: stable-first, soft contact
    conservative_solref0 = max(10 * timestep, 0.02)
    recommended = {
        "conservative": {
            "solref": [conservative_solref0, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0],
            "friction": [1.0, 0.005, 0.0001],
            "note": "Stable-first. Good for RL training.",
        },
        "stiff": {
            "solref": [max(2 * timestep, 0.004), 1.0],
            "solimp": [0.95, 0.99, 0.0005, 0.5, 2.0],
            "friction": [1.0, 0.005, 0.0001],
            "note": "Precision-first, tighter tolerances. Good for MPC/manipulation.",
        },
    }

    return json.dumps({"current": current, "issues": issues, "recommended": recommended})


@mcp.tool()
@safe_tool
async def validate_mjcf(
    xml_path: str | None = None,
    xml_string: str | None = None,
) -> str:
    """Pre-load static validation of MJCF XML. Does NOT require a loaded sim slot.

    Checks for: duplicate names, dangling actuator references, missing worldbody,
    geom size warnings, then falls through to full MuJoCo compile validation.

    Args:
        xml_path: Absolute path to .xml file (alternative to xml_string).
        xml_string: Raw MJCF XML string (alternative to xml_path).

    Returns:
        JSON: {"valid": bool, "errors": [{"rule": str, "element": str, "message": str}],
               "warnings": [...]}
    """
    return validate_mjcf_impl(xml_path=xml_path, xml_string=xml_string)


@mcp.tool()
@safe_tool
async def model_summary(ctx, sim_name: str | None = None) -> str:
    """Compact structural overview of a loaded model.

    Returns nq/nv/nu, body/geom/joint/sensor counts, timestep, solver type,
    per-joint details (first 20), per-actuator details (first 20), and mass extremes.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: Structured model summary with ~40 fields.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    return model_summary_impl(slot.model, slot.data)


@mcp.tool()
@safe_tool
async def suggest_contact_params(
    ctx,
    sim_name: str | None = None,
    geom1: str | None = None,
    geom2: str | None = None,
) -> str:
    """Analyse contact solver configuration and recommend adjustments.

    Checks solref time constant vs timestep, damping ratio, friction values.
    Returns current values, flagged issues, and two preset recommendations:
    'conservative' (RL training) and 'stiff' (MPC/manipulation).

    Args:
        sim_name: Slot name (default slot if None).
        geom1: Optional geom name to analyse that geom's specific contact params instead of the global average.
        geom2: Reserved for future pair-specific analysis (currently ignored if geom1 is set).

    Returns:
        JSON: {"current": {...}, "issues": [{...}], "recommended": {"conservative": {...}, "stiff": {...}}}
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    return suggest_contact_params_impl(slot.model, slot.data, geom1, geom2)
