"""Diagnostics tool group — XML validation, model summary, contact params, instability detection."""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from typing import Optional

import mujoco

from .._registry import mcp
from . import safe_tool


def validate_mjcf_impl(*, xml_path: Optional[str] = None, xml_string: Optional[str] = None) -> str:
    """Core logic for validate_mjcf — testable without MCP context."""
    if xml_path is None and xml_string is None:
        return json.dumps({"error": "invalid_input", "message": "Provide xml_path or xml_string"})

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

    # Check for duplicate names across all named elements
    names_seen: dict[str, str] = {}
    for elem in root.iter():
        name = elem.get("name")
        if name:
            key = name
            if key in names_seen:
                errors.append({
                    "rule": "duplicate_name",
                    "element": f"<{elem.tag} name='{name}'>",
                    "message": f"Duplicate name '{name}' (also used by <{names_seen[key]}>)",
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
    tendon_names = {t.get("name") for t in root.iter("tendon") if t.get("name")}
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


@mcp.tool()
@safe_tool
async def validate_mjcf(
    xml_path: Optional[str] = None,
    xml_string: Optional[str] = None,
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
