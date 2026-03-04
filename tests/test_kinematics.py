"""Tests for Kinematics tool group — solve_ik."""
import json
import pytest
import numpy as np
import mujoco

TWO_LINK_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14159 3.14159"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0.3 0 0"/>
        <site name="tip" pos="0.3 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _make_arm():
    model = mujoco.MjModel.from_xml_string(TWO_LINK_ARM_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_solve_ik_reachable_target_converges():
    from mujoco_mcp.tools.kinematics import solve_ik_impl
    model, data = _make_arm()
    result = json.loads(solve_ik_impl(
        model, data,
        site_name="tip",
        target_pos=[0.4, 0.2, 0.0],
        target_quat=None,
        joint_names=None,
        max_iter=200,
        tol=1e-4,
        damping=1e-3,
    ))
    assert result["converged"] is True
    assert result["pos_error"] < 1e-4
    assert result["ori_error"] is None
    assert len(result["qpos"]) == model.nq


def test_solve_ik_unreachable_returns_best_approximation():
    from mujoco_mcp.tools.kinematics import solve_ik_impl
    model, data = _make_arm()
    result = json.loads(solve_ik_impl(
        model, data,
        site_name="tip",
        target_pos=[2.0, 0.0, 0.0],
        target_quat=None,
        joint_names=None,
        max_iter=100,
        tol=1e-4,
        damping=1e-3,
    ))
    assert result["converged"] is False
    assert result["pos_error"] > 1e-4
    assert len(result["qpos"]) == model.nq


def test_solve_ik_joint_names_subset_restricts_motion():
    from mujoco_mcp.tools.kinematics import solve_ik_impl
    model, data = _make_arm()
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    result = json.loads(solve_ik_impl(
        model, data,
        site_name="tip",
        target_pos=[0.0, 0.3, 0.0],
        target_quat=None,
        joint_names=["joint1"],
        max_iter=200,
        tol=1e-3,
        damping=1e-3,
    ))
    # joint2 is at qpos index 1 — must remain 0
    assert abs(result["qpos"][1]) < 1e-10


def test_solve_ik_invalid_site_raises():
    from mujoco_mcp.tools.kinematics import solve_ik_impl
    model, data = _make_arm()
    with pytest.raises(ValueError, match="nonexistent_site"):
        solve_ik_impl(model, data, site_name="nonexistent_site", target_pos=[0.3, 0.0, 0.0])
