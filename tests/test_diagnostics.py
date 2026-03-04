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
    # Also verify the static warning was emitted
    assert any("geom_missing_size" in w["rule"] for w in result["warnings"])


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


def test_validate_mjcf_both_args_rejected():
    from mujoco_mcp.tools.diagnostics import validate_mjcf_impl
    result = json.loads(validate_mjcf_impl(xml_path="/some/path.xml", xml_string="<mujoco/>"))
    assert result["valid"] is False
    assert result["errors"][0]["rule"] == "invalid_input"
    assert "not both" in result["errors"][0]["message"]
    assert result["warnings"] == []


# ---- model_summary tests ----

BOX_DROP_XML = """
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint name="box_joint"/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _load_box_model():
    model = mujoco.MjModel.from_xml_string(BOX_DROP_XML)
    data = mujoco.MjData(model)
    return model, data


def test_model_summary_correct_nq_nv_nu():
    from mujoco_mcp.tools.diagnostics import model_summary_impl
    model, data = _load_box_model()
    result = json.loads(model_summary_impl(model, data))
    assert result["nq"] == 7   # freejoint = 7 qpos (quaternion)
    assert result["nv"] == 6   # freejoint = 6 DOF
    assert result["nu"] == 0   # no actuators


def test_model_summary_joint_names_present():
    from mujoco_mcp.tools.diagnostics import model_summary_impl
    model, data = _load_box_model()
    result = json.loads(model_summary_impl(model, data))
    joint_names = [j["name"] for j in result["joints"]]
    assert "box_joint" in joint_names


def test_model_summary_mass_extremes_reported():
    from mujoco_mcp.tools.diagnostics import model_summary_impl
    model, data = _load_box_model()
    result = json.loads(model_summary_impl(model, data))
    assert "heaviest_body" in result
    assert result["heaviest_body"]["mass"] >= 1.0


def test_model_summary_timestep_present():
    from mujoco_mcp.tools.diagnostics import model_summary_impl
    model, data = _load_box_model()
    result = json.loads(model_summary_impl(model, data))
    assert abs(result["timestep"] - 0.002) < 1e-9
