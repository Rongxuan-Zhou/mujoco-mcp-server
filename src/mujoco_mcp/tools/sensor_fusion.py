"""Phase 4d: Sensor fusion MCP tools.

Wraps SensorFusion and SensorManager from sensor_feedback.py into MCP tools.
"""

import json
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool
from ..sensor_feedback import (
    SensorFusion, SensorType, SensorReading, create_robot_sensor_suite
)
import time


def _get_sensor_manager(slot):
    sm = getattr(slot, "sensor_manager", None)
    if sm is None:
        raise ValueError(
            "No sensor manager for this slot. Call configure_sensor_fusion() first."
        )
    return sm


@mcp.tool()
@safe_tool
async def configure_sensor_fusion(
    ctx: Context,
    robot_type: str = "generic",
    n_joints: int = 7,
    sim_name: str | None = None,
) -> str:
    """Configure sensor fusion for a robot in the simulation.

    Creates a SensorManager with joint position/velocity/torque sensors.

    Args:
        robot_type: Robot category for sensor suite ("generic"|robot names).
        n_joints: Number of robot joints.
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"configured": true, "robot_type": str, "n_joints": int, "sensor_ids": [...]}
    """
    sim_manager = ctx.request_context.lifespan_context.sim_manager
    slot = sim_manager.get(sim_name)

    sensor_suite = create_robot_sensor_suite(robot_type, n_joints)
    # Attach real MuJoCo model/data so _collect_sensor_data reads from d.sensordata
    with sensor_suite._data_lock:
        sensor_suite._mj_model = slot.model
        sensor_suite._mj_data = slot.data
    slot.sensor_manager = sensor_suite
    slot.sensor_manager_n_joints = n_joints

    sensor_ids = list(sensor_suite.sensors.keys())

    return json.dumps({
        "configured": True,
        "robot_type": robot_type,
        "n_joints": n_joints,
        "sensor_ids": sensor_ids,
    }, indent=2)


@mcp.tool()
@safe_tool
async def get_fused_state(
    ctx: Context,
    sim_name: str | None = None,
) -> str:
    """Get fused sensor state from the simulation.

    Reads joint position, velocity from MuJoCo data and runs sensor fusion.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"joint_positions": [...], "joint_velocities": [...],
               "timestamp": float, "n_joints": int}
    """
    sim_manager = ctx.request_context.lifespan_context.sim_manager
    slot = sim_manager.get(sim_name)
    model, data = slot.model, slot.data
    _get_sensor_manager(slot)  # validate sensor manager exists

    n_joints = getattr(slot, "sensor_manager_n_joints", min(model.nv, 7))
    ts = float(data.time) if data.time > 0 else time.time()

    # Build fused state using SensorFusion
    fusion = SensorFusion()
    fusion.add_sensor("joint_positions", SensorType.JOINT_POSITION, weight=1.0)
    fusion.add_sensor("joint_velocities", SensorType.JOINT_VELOCITY, weight=1.0)

    qpos = data.qpos[:n_joints].copy()
    qvel = data.qvel[:n_joints].copy()

    pos_reading = SensorReading(
        sensor_id="joint_positions",
        sensor_type=SensorType.JOINT_POSITION,
        data=qpos,
        timestamp=ts if ts > 0 else time.time(),
        quality=1.0,
    )
    vel_reading = SensorReading(
        sensor_id="joint_velocities",
        sensor_type=SensorType.JOINT_VELOCITY,
        data=qvel,
        timestamp=ts if ts > 0 else time.time(),
        quality=1.0,
    )

    fused = fusion.fuse_sensor_data([pos_reading, vel_reading])

    return json.dumps({
        "joint_positions": fused.get(SensorType.JOINT_POSITION.value, qpos).tolist(),
        "joint_velocities": fused.get(SensorType.JOINT_VELOCITY.value, qvel).tolist(),
        "timestamp": ts,
        "n_joints": n_joints,
    }, indent=2)
