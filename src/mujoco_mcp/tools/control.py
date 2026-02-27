"""Phase 4c: Advanced robot controller MCP tools.

Provides PID + trajectory control accessible via MCP.
Controllers are stored per sim slot and persist across tool calls.
"""

import asyncio
import json
import numpy as np
import mujoco
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..constants import ASYNC_YIELD_INTERVAL
from . import safe_tool, _viewer_sync
from ..advanced_controllers import (
    TrajectoryPlanner,
    create_arm_controller, create_quadruped_controller, create_humanoid_controller,
)

_FACTORY = {
    "arm": create_arm_controller,
    "quadruped": create_quadruped_controller,
    "humanoid": create_humanoid_controller,
}


def _get_controller(slot):
    ctrl = getattr(slot, "controller", None)
    if ctrl is None:
        raise ValueError(
            "No controller for this slot. Call create_controller() first."
        )
    return ctrl


@mcp.tool()
@safe_tool
async def create_controller(
    ctx: Context,
    robot_type: str = "franka_panda",
    controller_kind: str = "arm",
    sim_name: str | None = None,
) -> str:
    """Create a PID+trajectory controller for a robot in a sim slot.

    Args:
        robot_type: Preset — "franka_panda"|"ur5e"|"anymal_c"|"go2"|"g1"|"h1".
        controller_kind: Category — "arm"|"quadruped"|"humanoid".
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"created": true, "robot_type": str, "controller_kind": str, "n_joints": int}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    factory = _FACTORY.get(controller_kind, create_arm_controller)
    ctrl = factory(robot_type)
    slot.controller = ctrl
    return json.dumps({
        "created": True,
        "robot_type": robot_type,
        "controller_kind": controller_kind,
        "n_joints": ctrl.n_joints,
    }, indent=2)


@mcp.tool()
@safe_tool
async def plan_trajectory(
    ctx: Context,
    start_qpos: list[float],
    end_qpos: list[float],
    duration: float = 2.0,
    trajectory_type: str = "min_jerk",
    frequency: float = 100.0,
    sim_name: str | None = None,
) -> str:
    """Plan a smooth joint-space trajectory and store it on the slot's controller.

    Args:
        start_qpos: Start joint positions (radians).
        end_qpos: End joint positions (radians).
        duration: Trajectory duration in seconds.
        trajectory_type: "min_jerk" (5th-order polynomial) or "spline" (cubic).
        frequency: Sampling frequency in Hz (default 100).
        sim_name: Slot name. Controller must already be created via create_controller().

    Returns:
        JSON: {"trajectory_type": str, "n_waypoints": int, "duration": float,
               "preview_positions": [[...], [...], [...]]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    ctrl = _get_controller(slot)

    start = np.array(start_qpos)
    end = np.array(end_qpos)

    if trajectory_type == "min_jerk":
        positions, _, _ = TrajectoryPlanner.minimum_jerk_trajectory(
            start, end, duration, frequency=frequency
        )
    else:
        waypoints = np.array([start, end])
        times = np.array([0.0, duration])
        positions, _, _ = TrajectoryPlanner.spline_trajectory(waypoints, times, frequency)

    ctrl.set_trajectory(np.array([start, end]), np.array([0.0, duration]))

    return json.dumps({
        "trajectory_type": trajectory_type,
        "n_waypoints": positions.shape[0],
        "duration": duration,
        "frequency": frequency,
        "preview_positions": positions[:3].tolist(),
    }, indent=2)


@mcp.tool()
@safe_tool
async def step_controller(
    ctx: Context,
    n_steps: int = 1,
    sim_name: str | None = None,
) -> str:
    """Execute N physics steps with PID+trajectory-tracking control.

    Each step: get target from trajectory -> compute PID -> apply ctrl -> mj_step.

    Args:
        n_steps: Number of physics steps (each = model.opt.timestep seconds).
        sim_name: Slot name.

    Returns:
        JSON: {"steps_executed": int, "final_qpos": [...], "trajectory_done": bool, "time": float}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data
    ctrl = _get_controller(slot)

    trajectory_done = False
    for step in range(n_steps):
        target = ctrl.get_trajectory_command()
        if target is None:
            trajectory_done = True
            break
        current_qpos = data.qpos[:ctrl.n_joints]
        commands = ctrl.pid_control(target, current_qpos)
        data.ctrl[:min(len(commands), model.nu)] = commands[:model.nu]
        mujoco.mj_step(model, data)
        if slot.recording:
            slot.trajectory.append({
                "t": float(data.time),
                "qpos": data.qpos.tolist(),
                "qvel": data.qvel.tolist(),
            })
        if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop every 1000 steps
    _viewer_sync(slot)

    return json.dumps({
        "steps_executed": n_steps,
        "final_qpos": data.qpos[:ctrl.n_joints].tolist(),
        "trajectory_done": trajectory_done,
        "time": float(data.time),
    }, indent=2)


@mcp.tool()
@safe_tool
async def get_controller_state(ctx: Context, sim_name: str | None = None) -> str:
    """Get controller state: current qpos, target, error, trajectory status.

    Args:
        sim_name: Slot name.

    Returns:
        JSON: {"current_qpos": [...], "time": float, "trajectory_active": bool,
               "target_qpos": [...], "error": [...]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data
    ctrl = _get_controller(slot)

    target = ctrl.get_trajectory_command()
    current_qpos = data.qpos[:ctrl.n_joints].tolist()

    result = {
        "current_qpos": current_qpos,
        "time": float(data.time),
        "trajectory_active": target is not None,
        "n_joints": ctrl.n_joints,
    }
    if target is not None:
        result["target_qpos"] = target.tolist()
        result["error"] = (np.array(target) - np.array(current_qpos)).tolist()

    return json.dumps(result, indent=2)
