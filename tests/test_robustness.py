"""Tests for robustness tools (apply_perturbation, stability_analysis, randomize_dynamics)."""
import csv
import json

import numpy as np
import pytest
import mujoco

from mujoco_mcp.tools.robustness import (
    _apply_perturbation_impl,
    _stability_analysis_impl,
    _randomize_dynamics_impl,
)

# ── Test models ────────────────────────────────────────────────────────────────

PENDULUM_XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pendulum">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="2.0"/>
      <geom name="rod" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

SLIDER_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""


def _make_pendulum():
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _make_slider():
    model = mujoco.MjModel.from_xml_string(SLIDER_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_apply_perturbation_increases_velocity():
    """A non-zero force pulse must produce a non-zero qvel deviation."""
    model, data = _make_pendulum()
    result = json.loads(_apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[10.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=20, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert result["max_qvel_deviation"] > 0.0


def test_apply_perturbation_small_force_recovers():
    """A small perturbation of a damped pendulum should recover within recovery_steps."""
    model, data = _make_pendulum()
    result = json.loads(_apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[0.1, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=10, recovery_steps=500,
        with_controller=False, recovery_threshold=0.01,
    ))
    assert result["recovered"] is True
    assert result["recovery_time_steps"] is not None


def test_apply_perturbation_unknown_body_raises():
    """Unknown body name must raise ValueError."""
    model, data = _make_pendulum()
    with pytest.raises(ValueError, match="not found"):
        _apply_perturbation_impl(
            model, data, controller=None,
            body_name="nonexistent", force=[1.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
            n_steps=10, recovery_steps=20,
            with_controller=False, recovery_threshold=0.1,
        )


def test_apply_perturbation_slot_state_unchanged():
    """Original slot data must be restored after apply_perturbation."""
    model, data = _make_pendulum()
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    result = json.loads(_apply_perturbation_impl(
        model, data, controller=None,
        body_name="pendulum", force=[5.0, 0.0, 0.0], torque=[0.0, 0.0, 0.0],
        n_steps=30, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert result["max_qvel_deviation"] > 0.0, "Perturbation had no effect"
    assert np.allclose(data.qpos, qpos_before), "qpos was modified"
    assert np.allclose(data.qvel, qvel_before), "qvel was modified"


def test_stability_analysis_result_schema():
    """Result must have all required keys."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0, 5.0],
        n_directions=4,
        n_steps=10, recovery_steps=50,
        with_controller=False, recovery_threshold=0.1,
    ))
    for key in ("stability_margin", "failure_ratio", "n_trials", "results"):
        assert key in result, f"Missing key: {key}"


def test_stability_analysis_directions_count():
    """Total trials must equal n_magnitudes × n_directions."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0, 5.0, 10.0],
        n_directions=6,
        n_steps=10, recovery_steps=30,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert result["n_trials"] == 3 * 6
    assert len(result["results"]) == 3 * 6


def test_stability_analysis_failure_ratio_range():
    """failure_ratio must be in [0, 1]."""
    model, data = _make_pendulum()
    result = json.loads(_stability_analysis_impl(
        model, data, controller=None,
        body_name="pendulum",
        force_magnitudes=[1.0],
        n_directions=4,
        n_steps=5, recovery_steps=20,
        with_controller=False, recovery_threshold=0.1,
    ))
    assert 0.0 <= result["failure_ratio"] <= 1.0


def test_randomize_dynamics_basic():
    """Basic uniform distribution: result must have correct schema and n_samples."""
    model, data = _make_slider()
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=10,
        eval_steps=20,
        metric="max_speed",
        goal_qpos=None,
        export_csv=None,
        random_seed=42,
    ))
    assert result["n_samples"] == 10
    for key in ("mean", "std", "min", "max", "best_params", "worst_params", "samples"):
        assert key in result, f"Missing key: {key}"
    assert result["min"] <= result["mean"] <= result["max"]


def test_randomize_dynamics_log_uniform_range():
    """log_uniform sampled values must stay within [low, high]."""
    model, data = _make_slider()
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "log_uniform", "low": 0.1, "high": 10.0}},
        n_samples=20,
        eval_steps=10,
        metric="max_speed",
        goal_qpos=None,
        export_csv=None,
        random_seed=7,
    ))
    for s in result["samples"]:
        mass = s["params"]["geom.box.mass"]
        assert 0.1 <= mass <= 10.0, f"log_uniform sample out of range: {mass}"


def test_randomize_dynamics_unknown_metric_raises():
    """Unknown metric must raise ValueError."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="metric"):
        _randomize_dynamics_impl(
            model, data,
            param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
            n_samples=2, eval_steps=5,
            metric="invalid_metric",
            goal_qpos=None, export_csv=None, random_seed=None,
        )


def test_randomize_dynamics_distance_without_goal_raises():
    """metric='distance' without goal_qpos must raise ValueError."""
    model, data = _make_slider()
    with pytest.raises(ValueError, match="goal_qpos"):
        _randomize_dynamics_impl(
            model, data,
            param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
            n_samples=2, eval_steps=5,
            metric="distance",
            goal_qpos=None,
            export_csv=None, random_seed=None,
        )


def test_randomize_dynamics_seeded_reproducible():
    """Same random_seed must produce identical mean across two calls."""
    kwargs = dict(
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=8, eval_steps=15,
        metric="max_speed", goal_qpos=None, export_csv=None, random_seed=123,
    )
    model1, data1 = _make_slider()
    result1 = json.loads(_randomize_dynamics_impl(model1, data1, **kwargs))
    model2, data2 = _make_slider()
    result2 = json.loads(_randomize_dynamics_impl(model2, data2, **kwargs))
    assert result1["mean"] == result2["mean"]


def test_randomize_dynamics_csv_export(tmp_path):
    """export_csv must create a readable CSV with correct columns."""
    model, data = _make_slider()
    csv_path = str(tmp_path / "out.csv")
    result = json.loads(_randomize_dynamics_impl(
        model, data,
        param_distributions={"geom.box.mass": {"type": "uniform", "low": 0.5, "high": 2.0}},
        n_samples=5, eval_steps=10,
        metric="max_speed", goal_qpos=None,
        export_csv=csv_path, random_seed=0,
    ))
    assert result["csv_path"] == csv_path
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 5
    assert "geom.box.mass" in rows[0]
    assert "max_speed" in rows[0]
