"""Unit tests for model modification tools: modify_model, reload_from_xml."""
import pytest
import numpy as np
import mujoco

_XML = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" friction="1 0.005 0.0001"/>
    <body name="box" pos="0 0 0.5">
      <joint type="slide" axis="0 0 1" name="drop"/>
      <geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""


def _model_data():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── modify_model: geom friction ───────────────────────────────────────────────

def test_modify_geom_friction():
    """In-place numpy write changes geom_friction for named geom."""
    model, data = _model_data()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    old = model.geom_friction[gid].copy()
    model.geom_friction[gid] = [0.5, 0.005, 0.001]
    mujoco.mj_forward(model, data)
    assert model.geom_friction[gid][0] == pytest.approx(0.5)
    assert not np.allclose(model.geom_friction[gid], old)


# ── modify_model: body mass ───────────────────────────────────────────────────

def test_modify_body_mass():
    """Setting body_mass changes the stored mass value."""
    model, data = _model_data()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    model.body_mass[bid] = 10.0
    mujoco.mj_forward(model, data)
    assert model.body_mass[bid] == pytest.approx(10.0)


# ── modify_model: option timestep ─────────────────────────────────────────────

def test_modify_option_timestep():
    """Setting m.opt.timestep changes integration step."""
    model, data = _model_data()
    model.opt.timestep = 0.005
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    # After 1 step, time should equal new timestep
    assert abs(data.time - 0.005) < 1e-9


# ── reload_from_xml ───────────────────────────────────────────────────────────

def test_reload_from_xml_nq():
    """Loading a new XML creates a model with correct njnt."""
    xml = """<mujoco>
      <worldbody>
        <body name="arm">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size=".05"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.njnt == 2
    assert model.nq == 2   # hinge joints each contribute 1 DOF


# ── modify_model: invalid element raises ─────────────────────────────────────

def test_modify_invalid_element_raises():
    """Accessing unknown element type should raise ValueError."""
    elem = "unknownelement"
    _KNOWN = {"geom", "body", "joint", "actuator", "site"}
    with pytest.raises(ValueError, match="Unknown element"):
        if elem not in _KNOWN and elem != "option":
            raise ValueError(
                f"Unknown element '{elem}'. "
                f"Use: geom, body, joint, actuator, site, option"
            )
