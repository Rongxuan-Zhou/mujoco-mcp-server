"""Phase 4f: Reinforcement Learning environment MCP tools.

Wraps MuJoCoRLEnvironment (Gymnasium-compatible) as MCP tools.
Requires: pip install gymnasium
"""

import json
import numpy as np
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool

try:
    import gymnasium  # noqa: F401
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

try:
    from ..rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType
    _RL_AVAILABLE = True
except ImportError:
    _RL_AVAILABLE = False


def _require_rl():
    if not _GYM_AVAILABLE:
        raise RuntimeError("gymnasium not installed. Run: pip install gymnasium")
    if not _RL_AVAILABLE:
        raise RuntimeError("RL integration module unavailable.")


@mcp.tool()
@safe_tool
async def create_rl_env(
    ctx: Context,
    robot_type: str = "franka_panda",
    task_type: str = "reaching",
    max_episode_steps: int = 1000,
    sim_name: str | None = None,
) -> str:
    """Create a Gymnasium RL environment wrapping a sim slot's model.

    Requires gymnasium: pip install gymnasium

    Args:
        robot_type: Robot type for action/observation space -- "franka_panda"|"ur5e"|etc.
        task_type: RL task -- "reaching"|"balancing"|"walking".
        max_episode_steps: Episode horizon (number of steps before truncation).
        sim_name: Slot name (used to auto-detect observation/action dimensions).

    Returns:
        JSON: {"created": true, "obs_shape": [...], "action_shape": [...],
               "task_type": str, "max_episode_steps": int}
    """
    _require_rl()
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)

    task_map = {
        "reaching": TaskType.REACHING,
        "balancing": TaskType.BALANCING,
        "walking": TaskType.WALKING,
    }
    if task_type not in task_map:
        raise ValueError(f"task_type must be one of {list(task_map)}, got {task_type!r}")

    config = RLConfig(
        robot_type=robot_type,
        task_type=task_map[task_type],
        max_episode_steps=max_episode_steps,
        observation_space_size=slot.model.nq + slot.model.nv,
        action_space_size=slot.model.nu,
    )
    env = MuJoCoRLEnvironment(config)
    slot.rl_env = env

    obs, _ = env.reset()
    return json.dumps({
        "created": True,
        "robot_type": robot_type,
        "task_type": task_type,
        "obs_shape": list(obs.shape),
        "action_shape": [slot.model.nu],
        "max_episode_steps": max_episode_steps,
    }, indent=2)


@mcp.tool()
@safe_tool
async def rl_step(
    ctx: Context,
    action: list[float],
    sim_name: str | None = None,
) -> str:
    """Execute one RL environment step.

    Args:
        action: Control vector (length must match model.nu).
        sim_name: Slot name.

    Returns:
        JSON: {"obs": [...], "reward": float, "terminated": bool,
               "truncated": bool, "info": {...}}
    """
    _require_rl()
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    env = getattr(slot, "rl_env", None)
    if env is None:
        raise ValueError("No RL environment. Call create_rl_env() first.")

    obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
    return json.dumps({
        "obs": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()},
    }, indent=2)
