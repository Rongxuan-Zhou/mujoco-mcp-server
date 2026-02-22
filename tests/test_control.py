import pytest
import json
import numpy as np
from mujoco_mcp.advanced_controllers import (
    PIDConfig, PIDController, TrajectoryPlanner, create_arm_controller
)

def test_pid_controller_convergence():
    """PID controller must reduce error over time."""
    # Use gains stable for simplified plant: pos += output * dt (dt=0.02)
    config = PIDConfig(kp=2.0, ki=0.0, kd=0.1)
    ctrl = PIDController(config)
    error_history = []
    pos = 0.0
    target = 1.0
    dt = 0.02
    for _ in range(50):
        output = ctrl.update(target, pos, dt)
        pos += output * dt
        error_history.append(abs(target - pos))
    assert error_history[-1] < error_history[0]

def test_min_jerk_trajectory_shape():
    """Minimum jerk trajectory must have correct shape."""
    start = np.zeros(7)
    end = np.ones(7)
    positions, velocities, accels = TrajectoryPlanner.minimum_jerk_trajectory(
        start, end, duration=2.0, frequency=100.0
    )
    assert positions.shape == (200, 7)
    assert velocities.shape == (200, 7)

def test_min_jerk_boundary_conditions():
    """Trajectory must start at start_pos and end at end_pos."""
    start = np.array([0.0, 0.5, -1.0])
    end = np.array([1.0, -0.5, 0.5])
    positions, _, _ = TrajectoryPlanner.minimum_jerk_trajectory(start, end, 1.0)
    np.testing.assert_allclose(positions[0], start, atol=1e-6)
    np.testing.assert_allclose(positions[-1], end, atol=1e-4)

def test_create_arm_controller():
    ctrl = create_arm_controller("franka_panda")
    assert ctrl.n_joints == 7
    assert len(ctrl.pid_controllers) == 7
