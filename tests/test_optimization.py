"""Tests for trajectory optimization tools (iLQR + MPPI)."""
import json
import numpy as np
import pytest
import mujoco

from mujoco_mcp.tools.optimization import (
    _ilqr_impl,
    _mppi_impl,
)

# ── Simple 1-DOF slider model ─────────────────────────────────────────────────
SLIDER_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body>
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def _make_slider():
    model = mujoco.MjModel.from_xml_string(SLIDER_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── iLQR tests ────────────────────────────────────────────────────────────────

def test_ilqr_slider_converges():
    """iLQR should drive 1-DOF slider from 0 to 0.5 and converge."""
    model, data = _make_slider()
    result = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0],
        goal_qpos=[0.5],
        horizon=50,
        max_iter=20,
        template="reach",
        Q_user=None,
        R_user=None,
    ))
    assert result["converged"] is True
    assert result["iterations"] <= 20
    assert abs(result["trajectory"][-1][0] - 0.5) < 0.05


def test_ilqr_custom_QR_accepted():
    """User-supplied Q/R matrices should be accepted and alter the result."""
    model, data = _make_slider()
    # Default template="reach"
    result_default = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.3],
        horizon=30, max_iter=10,
        template="reach", Q_user=None, R_user=None,
    ))
    # Heavy control penalty → should use less control effort
    model, data = _make_slider()
    R_heavy = [[100.0]]  # 1×1 matrix
    Q_light = [[1.0, 0.0], [0.0, 0.1]]  # 2×2 matrix
    result_custom = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.3],
        horizon=30, max_iter=10,
        template="reach", Q_user=Q_light, R_user=R_heavy,
    ))
    # Custom Q/R should not raise; controls should differ
    ctrl_default_max = max(abs(u[0]) for u in result_default["controls"])
    ctrl_custom_max = max(abs(u[0]) for u in result_custom["controls"])
    # Heavy R penalty should produce smaller control magnitude
    assert ctrl_custom_max < ctrl_default_max, (
        f"Heavy R should reduce control magnitude: "
        f"default={ctrl_default_max:.4f}, custom={ctrl_custom_max:.4f}"
    )


def test_ilqr_waypoints_subsampled():
    """waypoints field must be present, length ≤ 10, and endpoints match trajectory."""
    model, data = _make_slider()
    result = json.loads(_ilqr_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.4],
        horizon=50, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    wp = result["waypoints"]
    traj = result["trajectory"]
    assert isinstance(wp, list)
    assert 1 <= len(wp) <= 10
    # First waypoint should be near start
    assert abs(wp[0][0] - traj[0][0]) < 1e-9
    # Last waypoint should be near trajectory end
    assert abs(wp[-1][0] - traj[-1][0]) < 1e-9


# ── MPPI tests ────────────────────────────────────────────────────────────────

def test_mppi_slider_reaches_goal():
    """MPPI should drive slider from 0 toward 0.5 within tolerance."""
    model, data = _make_slider()
    np.random.seed(42)
    result = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0],
        goal_qpos=[0.5],
        horizon=50,
        n_samples=50,
        temperature=0.1,
        noise_sigma=1.0,
        max_iter=10,
        template="reach",
        Q_user=None,
        R_user=None,
    ))
    assert abs(result["trajectory"][-1][0] - 0.5) < 0.2  # within 20cm


def test_mppi_more_samples_better_result():
    """With more samples, MPPI should reach a lower final cost (seed fixed)."""
    model, data = _make_slider()

    np.random.seed(42)
    result_few = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.5],
        horizon=30, n_samples=3,
        temperature=0.1, noise_sigma=1.0, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    model, data = _make_slider()
    np.random.seed(42)
    result_many = json.loads(_mppi_impl(
        model, data,
        start_qpos=[0.0], goal_qpos=[0.5],
        horizon=30, n_samples=100,
        temperature=0.1, noise_sigma=1.0, max_iter=5,
        template="reach", Q_user=None, R_user=None,
    ))
    # More samples should not be catastrophically worse
    assert result_many["final_cost"] < result_few["final_cost"] * 10.0


def test_unknown_template_raises():
    """Unknown template name should raise ValueError mentioning 'template'."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="template"):
        _ilqr_impl(
            model, data,
            start_qpos=[0.0], goal_qpos=[0.5],
            horizon=10, max_iter=5,
            template="unknown_template",
            Q_user=None, R_user=None,
        )
