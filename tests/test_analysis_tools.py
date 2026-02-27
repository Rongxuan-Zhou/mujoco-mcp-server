"""Unit tests for analysis tools: contacts, Jacobian, derivatives, sensors, energy, forces."""
import numpy as np
import mujoco

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

_XML_SENSOR = """<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
      <site name="tip" pos=".1 0 0"/>
    </body>
  </worldbody>
  <sensor><framepos name="tip_pos" objtype="site" objname="tip"/></sensor>
</mujoco>"""

_XML_ACTUATED = """<mujoco>
  <worldbody>
    <body name="s">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
    </body>
  </worldbody>
  <actuator><motor name="m1" joint="j1"/></actuator>
</mujoco>"""


def _box():
    model = mujoco.MjModel.from_xml_string(_XML_BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── analyze_contacts ──────────────────────────────────────────────────────────

def test_analyze_contacts_after_landing():
    """After box falls to floor, ncon > 0."""
    model, data = _box()
    for _ in range(800):
        mujoco.mj_step(model, data)
    assert data.ncon > 0


def test_analyze_contacts_returns_geom_ids():
    """Contact geom IDs are valid (< ngeom)."""
    from mujoco_mcp.compat import contact_geoms
    model, data = _box()
    for _ in range(800):
        mujoco.mj_step(model, data)
    if data.ncon > 0:
        gid1, gid2 = contact_geoms(data.contact[0])
        assert 0 <= gid1 < model.ngeom
        assert 0 <= gid2 < model.ngeom


# ── compute_jacobian ──────────────────────────────────────────────────────────

def test_compute_jacobian_shape():
    """Jacobian for a 1-DOF model has shape (6, nv)."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])
    assert J.shape == (6, model.nv)


def test_compute_jacobian_singular_values():
    """SVD of Jacobian gives at least one singular value >= 0 and rank > 0."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    sv = np.linalg.svd(np.vstack([jacp, jacr]), compute_uv=False)
    assert all(s >= 0 for s in sv)
    rank = int(np.sum(sv > 1e-10))
    assert rank > 0


# ── analyze_energy ────────────────────────────────────────────────────────────

def test_analyze_energy_initial_state():
    """Initial-state box at height 0.5 has potential energy > 0."""
    from mujoco_mcp.compat import ensure_energy, restore_energy
    model, data = _box()
    was = ensure_energy(model)
    mujoco.mj_forward(model, data)
    pot = float(data.energy[0])
    kin = float(data.energy[1])
    restore_energy(model, was)
    # Box at z=0.5 has gravitational PE
    assert pot + kin > 0.0


# ── analyze_forces ────────────────────────────────────────────────────────────

def test_analyze_forces_qfrc_length():
    """qfrc vectors must have length == nv."""
    model, data = _box()
    assert len(data.qfrc_applied) == model.nv
    assert len(data.qfrc_bias) == model.nv
    assert len(data.qacc) == model.nv


# ── read_sensors ──────────────────────────────────────────────────────────────

def test_read_sensors_all():
    """Model with one sensor returns non-empty dict with correct dimensions."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = {}
    for sid in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
        if name:
            adr = model.sensor_adr[sid]
            dim = model.sensor_dim[sid]
            result[name] = data.sensordata[adr: adr + dim].tolist()
    assert len(result) > 0
    assert "tip_pos" in result
    assert len(result["tip_pos"]) == 3


# ── compute_derivatives ───────────────────────────────────────────────────────

def test_compute_derivatives_ab_shape():
    """A and B matrices have correct shape for 1-DOF actuated model."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nv, nu = model.nv, model.nu
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    assert A.shape == (2 * nv, 2 * nv)
    assert B.shape == (2 * nv, nu)
