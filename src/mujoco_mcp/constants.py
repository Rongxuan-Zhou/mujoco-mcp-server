# src/mujoco_mcp/constants.py
"""Centralised constants shared across mujoco_mcp modules.

Import from here rather than defining magic numbers locally.
"""

# Maximum physics steps per single MCP tool call
# (simulation.py sim_step + workflows.py run_and_analyze / evaluate_trajectory)
MAX_SIM_STEPS: int = 100_000

# asyncio.sleep(0) yield interval — prevents event-loop starvation in long loops
# (simulation.py, workflows.py, control.py, vision.py)
ASYNC_YIELD_INTERVAL: int = 1_000

# Jacobian matrix display threshold: omit full matrices when nv > this value
JACOBIAN_NV_THRESHOLD: int = 50

# SensorManager internal queue capacity
MAX_SENSOR_QUEUE: int = 1_000

# ProcessPoolExecutor child-process timeout (seconds)
BATCH_TASK_TIMEOUT: int = 300

# Default worker count for run_sweep (overridden by MUJOCO_MCP_MAX_WORKERS env var)
BATCH_MAX_WORKERS_DEFAULT: int = 8
