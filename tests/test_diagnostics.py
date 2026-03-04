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


# ---- suggest_contact_params tests ----

TIGHT_SOLREF_XML = """
<mujoco>
  <option timestep="0.002"/>
  <default>
    <geom solref="0.001 1.0" solimp="0.9 0.95 0.001 0.5 2.0" friction="1.0 0.005 0.0001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

ZERO_FRICTION_XML = """
<mujoco>
  <option timestep="0.002"/>
  <default>
    <geom solref="0.02 1.0" solimp="0.9 0.95 0.001 0.5 2.0" friction="0.0 0.0 0.0"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 1">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_suggest_contact_params_tight_solref_flagged():
    from mujoco_mcp.tools.diagnostics import suggest_contact_params_impl
    model = mujoco.MjModel.from_xml_string(TIGHT_SOLREF_XML)
    data = mujoco.MjData(model)
    result = json.loads(suggest_contact_params_impl(model, data))
    issue_types = [i["type"] for i in result["issues"]]
    assert "solref_too_tight" in issue_types


def test_suggest_contact_params_zero_friction_flagged():
    from mujoco_mcp.tools.diagnostics import suggest_contact_params_impl
    model = mujoco.MjModel.from_xml_string(ZERO_FRICTION_XML)
    data = mujoco.MjData(model)
    result = json.loads(suggest_contact_params_impl(model, data))
    issue_types = [i["type"] for i in result["issues"]]
    assert "zero_friction" in issue_types


def test_suggest_contact_params_conservative_solref_valid():
    from mujoco_mcp.tools.diagnostics import suggest_contact_params_impl
    model = mujoco.MjModel.from_xml_string(TIGHT_SOLREF_XML)
    data = mujoco.MjData(model)
    result = json.loads(suggest_contact_params_impl(model, data))
    timestep = float(model.opt.timestep)
    conservative_solref0 = result["recommended"]["conservative"]["solref"][0]
    assert conservative_solref0 >= 2 * timestep


def test_suggest_contact_params_stiff_not_looser_than_conservative():
    from mujoco_mcp.tools.diagnostics import suggest_contact_params_impl
    model = mujoco.MjModel.from_xml_string(TIGHT_SOLREF_XML)
    data = mujoco.MjData(model)
    result = json.loads(suggest_contact_params_impl(model, data))
    timestep = float(model.opt.timestep)
    stiff_solref0 = result["recommended"]["stiff"]["solref"][0]
    conservative_solref0 = result["recommended"]["conservative"]["solref"][0]
    # Both must be >= 2*timestep; stiff has shorter time constant (tighter) = smaller or equal
    assert stiff_solref0 >= 2 * timestep
    assert stiff_solref0 <= conservative_solref0


def test_suggest_contact_params_invalid_geom_raises():
    from mujoco_mcp.tools.diagnostics import suggest_contact_params_impl
    model = mujoco.MjModel.from_xml_string(TIGHT_SOLREF_XML)
    data = mujoco.MjData(model)
    # Invalid geom name should raise ValueError (caught by @safe_tool in MCP context)
    try:
        suggest_contact_params_impl(model, data, geom1="nonexistent_geom")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_geom" in str(e)


# ---- diagnose_instability tests ----

STABLE_BOX_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="box" pos="0 0 0.2">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_diagnose_instability_stable_model():
    from mujoco_mcp.tools.diagnostics import diagnose_instability_impl
    model = mujoco.MjModel.from_xml_string(STABLE_BOX_XML)
    data = mujoco.MjData(model)
    result = json.loads(diagnose_instability_impl(model, data, n_steps=50))
    assert result["stable"] is True
    assert result["first_unstable_step"] is None


def test_diagnose_instability_velocity_explosion_caught():
    from mujoco_mcp.tools.diagnostics import diagnose_instability_impl
    model = mujoco.MjModel.from_xml_string(STABLE_BOX_XML)
    data = mujoco.MjData(model)
    # Manually inject large velocity to trigger check before stepping
    data.qvel[0] = 2000.0
    result = json.loads(diagnose_instability_impl(model, data, n_steps=5))
    assert result["stable"] is False
    assert result["first_unstable_step"] == 0  # detected at step 0 (initial state)


def test_diagnose_instability_first_unstable_step_reported():
    from mujoco_mcp.tools.diagnostics import diagnose_instability_impl
    model = mujoco.MjModel.from_xml_string(STABLE_BOX_XML)
    data = mujoco.MjData(model)
    # Inject instability
    data.qvel[0] = 5000.0
    result = json.loads(diagnose_instability_impl(model, data, n_steps=10))
    assert result["stable"] is False
    assert isinstance(result["first_unstable_step"], int)
    assert result["first_unstable_step"] >= 0


def test_diagnose_instability_result_schema():
    from mujoco_mcp.tools.diagnostics import diagnose_instability_impl
    model = mujoco.MjModel.from_xml_string(STABLE_BOX_XML)
    data = mujoco.MjData(model)
    result = json.loads(diagnose_instability_impl(model, data, n_steps=10))
    # All required keys present
    assert "stable" in result
    assert "steps_run" in result
    assert "first_unstable_step" in result
    assert "issues" in result
    assert "suggestions" in result
    assert isinstance(result["issues"], list)
    assert isinstance(result["suggestions"], list)
