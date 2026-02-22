import pytest
from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator

def test_coordinator_headless_init():
    """Coordinator must init without viewer_client."""
    coord = MultiRobotCoordinator()
    assert coord is not None

def test_add_robot():
    coord = MultiRobotCoordinator()
    ok = coord.add_robot("r1", "franka_panda", {"manipulation": True})
    assert ok is True or ok is None  # accept bool or None return

def test_get_system_status():
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    status = coord.get_system_status()
    assert "num_robots" in status
    assert status["num_robots"] == 1

def test_collision_check_no_collision():
    import numpy as np
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    coord.add_robot("r2", "ur5e", {"manipulation": True})
    robot_ids = list(coord.robot_states.keys())
    collisions = []
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r1, r2 = robot_ids[i], robot_ids[j]
            if coord.collision_checker.check_collision(
                coord.robot_states[r1], coord.robot_states[r2]
            ):
                collisions.append([r1, r2])
    assert len(collisions) == 0
