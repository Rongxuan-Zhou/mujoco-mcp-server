"""Tests for Diagnostics tool group."""
import json
import mujoco
import pytest

VALID_XML = """
<mujoco>
  <worldbody>
    <body name="box">
      <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

MISSING_GEOM_SIZE_XML = """
<mujoco>
  <worldbody>
    <body name="box">
      <geom name="bad_geom" type="box"/>
    </body>
  </worldbody>
</mujoco>
"""

DANGLING_ACTUATOR_XML = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="shoulder" type="hinge"/>
      <geom type="capsule" size="0.05" fromto="0 0 0 0 0 0.3"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="nonexistent_joint"/>
  </actuator>
</mujoco>
"""

DUPLICATE_NAME_XML = """
<mujoco>
  <worldbody>
    <body name="box">
      <geom name="geom1" type="sphere" size="0.1"/>
    </body>
    <body name="box">
      <geom name="geom2" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_validate_mjcf_valid_xml_passes():
    from mujoco_mcp.tools.diagnostics import validate_mjcf_impl
    result = json.loads(validate_mjcf_impl(xml_string=VALID_XML))
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_mjcf_missing_geom_size_caught():
    from mujoco_mcp.tools.diagnostics import validate_mjcf_impl
    result = json.loads(validate_mjcf_impl(xml_string=MISSING_GEOM_SIZE_XML))
    # MuJoCo will fail to compile (box without size), so valid=False
    assert result["valid"] is False
    rules = [e["rule"] for e in result["errors"]]
    assert any("size" in r or "mujoco" in r or "geom" in r for r in rules)


def test_validate_mjcf_dangling_actuator_caught():
    from mujoco_mcp.tools.diagnostics import validate_mjcf_impl
    result = json.loads(validate_mjcf_impl(xml_string=DANGLING_ACTUATOR_XML))
    assert result["valid"] is False
    rules = [e["rule"] for e in result["errors"]]
    assert any("dangling" in r or "mujoco" in r for r in rules)


def test_validate_mjcf_duplicate_name_caught():
    from mujoco_mcp.tools.diagnostics import validate_mjcf_impl
    result = json.loads(validate_mjcf_impl(xml_string=DUPLICATE_NAME_XML))
    assert result["valid"] is False
    rules = [e["rule"] for e in result["errors"]]
    assert any("duplicate" in r for r in rules)
