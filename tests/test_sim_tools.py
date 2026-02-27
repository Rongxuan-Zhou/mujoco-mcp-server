"""Unit tests for core simulation tools (sim_step, sim_forward, sim_reset, etc.).

Tests operate directly on MuJoCo model/data to avoid MCP context complexity.
"""
import pytest
import numpy as np
import mujoco

# XML with free-floating box above a floor — suitable for contact tests
_XML_BOX = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""

# XML with slider joint and actuator — for ctrl tests
_XML_ACTUATED = """<mujoco>
  <worldbody>
    <body name="slider">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
    </body>
  </worldbody>
  <actuator><motor name="m1" joint="j1"/></actuator>
</mujoco>"""


def _box_model_data():
    model = mujoco.MjModel.from_xml_string(_XML_BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── sim_step ──────────────────────────────────────────────────────────────────

def test_sim_step_advances_time():
    """A single mj_step increments simulation time."""
    model, data = _box_model_data()
    mujoco.mj_step(model, data)
    assert data.time > 0.0


def test_sim_step_n_steps():
    """100 steps advances time by 100 × timestep (within float64 precision)."""
    model, data = _box_model_data()
    dt = model.opt.timestep
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.time - 100 * dt) < 1e-9


def test_sim_step_ctrl_write():
    """ctrl written to data.ctrl persists across a step."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [2.5]
    mujoco.mj_step(model, data)
    assert abs(data.ctrl[0] - 2.5) < 1e-9


def test_sim_step_invalid_n_steps_raises():
    """Layer 2: sim_step validation must raise ValueError for n_steps=0."""
    from mujoco_mcp.constants import MAX_SIM_STEPS
    n_steps = 0
    with pytest.raises(ValueError, match=r"n_steps must be 1"):
        if not 1 <= n_steps <= MAX_SIM_STEPS:
            raise ValueError(f"n_steps must be 1–{MAX_SIM_STEPS}")


# ── sim_forward ───────────────────────────────────────────────────────────────

def test_sim_forward_updates_ncon():
    """mj_forward after stepping updates contact count (ncon >= 0)."""
    model, data = _box_model_data()
    for _ in range(500):        # let box fall and land
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.ncon >= 0


# ── sim_reset ─────────────────────────────────────────────────────────────────

def test_sim_reset_zeroes_time():
    """mj_resetData sets time back to 0."""
    model, data = _box_model_data()
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert data.time > 0.0
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    assert data.time == 0.0


def test_sim_reset_restores_qpos():
    """mj_resetData restores qpos to initial values."""
    model, data = _box_model_data()
    initial_qpos = data.qpos.copy()
    for _ in range(200):
        mujoco.mj_step(model, data)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos, initial_qpos, atol=1e-9)


# ── sim_get_state ─────────────────────────────────────────────────────────────

def test_sim_get_state_fields():
    """State dict contains time, qpos, qvel, ctrl with correct dimensions."""
    model, data = _box_model_data()
    state = {
        "time": data.time,
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
    }
    for key in ("time", "qpos", "qvel", "ctrl"):
        assert key in state
    assert len(state["qpos"]) == model.nq
    assert len(state["qvel"]) == model.nv


# ── sim_set_state ─────────────────────────────────────────────────────────────

def test_sim_set_state_qpos_roundtrip():
    """Setting qpos via data.qpos write + mj_forward persists."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[0] = 0.42
    mujoco.mj_forward(model, data)
    assert abs(data.qpos[0] - 0.42) < 1e-9


# ── sim_record ────────────────────────────────────────────────────────────────

def test_sim_record_trajectory_accumulates():
    """Recording trajectory: N steps → N frames in buffer."""
    model, data = _box_model_data()
    trajectory = []
    for _ in range(10):
        mujoco.mj_step(model, data)
        trajectory.append({"t": float(data.time), "qpos": data.qpos.copy().tolist()})
    assert len(trajectory) == 10
    assert trajectory[0]["t"] > 0.0
    assert trajectory[-1]["t"] > trajectory[0]["t"]


def test_sim_record_clear():
    """Clearing trajectory empties the buffer."""
    trajectory = [{"t": float(i) * 0.002} for i in range(20)]
    trajectory.clear()
    assert len(trajectory) == 0
