"""Phase 4e: Multi-robot coordination MCP tools.

A global MultiRobotCoordinator singleton manages robot registration,
task allocation, and collision detection across all sim slots.
"""

import json
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool
from ..multi_robot_coordinator import MultiRobotCoordinator

_coordinator = None


def _get_coordinator() -> MultiRobotCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiRobotCoordinator()
    return _coordinator


@mcp.tool()
@safe_tool
async def coordinator_add_robot(
    ctx: Context,
    robot_id: str,
    robot_type: str,
    capabilities: dict | None = None,
) -> str:
    """Register a robot in the multi-robot coordinator.

    Args:
        robot_id: Unique identifier for this robot instance, e.g. "arm_left".
        robot_type: Type -- "franka_panda"|"ur5e"|"anymal_c"|"go2".
        capabilities: Dict of capabilities, e.g. {"manipulation": true, "mobility": false}.

    Returns:
        JSON: {"added": bool, "robot_id": str, "robot_type": str}
    """
    coord = _get_coordinator()
    caps = capabilities or {"manipulation": True, "mobility": False}
    ok = coord.add_robot(robot_id, robot_type, caps)
    return json.dumps({"added": bool(ok) if ok is not None else True,
                       "robot_id": robot_id, "robot_type": robot_type}, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_get_status(ctx: Context) -> str:
    """Get multi-robot coordinator system status.

    Returns:
        JSON: {"running": bool, "num_robots": int, "pending_tasks": int,
               "active_tasks": int, "completed_tasks": int,
               "robots": {robot_id: status_string}}
    """
    coord = _get_coordinator()
    status = coord.get_system_status()
    # Convert RobotStatus enum values to strings
    if "robots" in status:
        robots = {}
        for k, v in status["robots"].items():
            robots[k] = v.value if hasattr(v, 'value') else str(v)
        status["robots"] = robots
    return json.dumps(status, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_check_collisions(ctx: Context) -> str:
    """Run pairwise collision detection across all registered robots.

    Returns:
        JSON: {"collisions": [[robot1_id, robot2_id], ...], "count": int}
    """
    coord = _get_coordinator()
    collisions = []
    robot_ids = list(coord.robot_states.keys())
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r1, r2 = robot_ids[i], robot_ids[j]
            if coord.collision_checker.check_collision(
                coord.robot_states[r1], coord.robot_states[r2]
            ):
                collisions.append([r1, r2])
    return json.dumps({"collisions": collisions, "count": len(collisions)}, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_assign_task(
    ctx: Context,
    task_type: str,
    robot_ids: list[str],
    parameters: dict | None = None,
) -> str:
    """Assign a coordinated task to a set of robots.

    Args:
        task_type: "cooperative_manipulation"|"formation_control".
        robot_ids: List of robot IDs (must be registered via coordinator_add_robot).
        parameters:
            For "formation_control": {"formation": "line"|"circle", "spacing": float}
            For "cooperative_manipulation": {"target_object": str}

    Returns:
        JSON: {"task_id": str, "status": "pending"}
    """
    import numpy as np
    coord = _get_coordinator()
    params = parameters or {}

    valid_types = ["cooperative_manipulation", "formation_control"]
    if task_type not in valid_types:
        raise ValueError(f"task_type must be one of {valid_types}, got {task_type!r}")

    if task_type == "formation_control":
        task_id = coord.formation_control(
            robot_ids,
            params.get("formation", "line"),
            params.get("spacing", 1.0),
        )
    else:
        approaches = {
            rid: np.array(params.get(f"{rid}_approach", [0.0, 0.0, 0.0]))
            for rid in robot_ids
        }
        task_id = coord.cooperative_manipulation(
            robot_ids, params.get("target_object", "object"), approaches
        )

    return json.dumps({"task_id": str(task_id) if task_id else "task_0",
                       "status": "pending"}, indent=2)
