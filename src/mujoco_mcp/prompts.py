"""MCP Prompts — guided prompt templates for common simulation workflows."""

from mcp.server.fastmcp import Context
from ._registry import mcp


@mcp.prompt()
async def run_simulation(
    model_path: str,
    n_steps: int = 1000,
    track_quantities: str = "energy,contact_count",
) -> str:
    """Guided prompt: load a model and run a basic simulation."""
    track = [t.strip() for t in track_quantities.split(",")]
    return f"""Load and run a MuJoCo simulation.

Steps to follow:
1. Load the model:
   sim_load(xml_path="{model_path}", name="default")

2. Check what was loaded (bodies, joints, actuators):
   Use the model summary returned by sim_load.

3. Start recording and run the simulation:
   sim_record(action="start")
   run_and_analyze(n_steps={n_steps}, track={track}, capture_every_n=200)

4. Analyze the results:
   - Review the time-series data for {', '.join(track)}
   - Check the keyframe images
   - Note any unusual energy jumps or contact events

5. If you see unexpected behavior:
   sim_reset()
   Then adjust parameters with modify_model() before re-running.
"""


@mcp.prompt()
async def debug_contact_scenario(
    model_path: str,
    n_steps: int = 500,
) -> str:
    """Guided prompt: diagnose contact and collision issues."""
    return f"""Debug contacts in a MuJoCo simulation.

Steps to follow:
1. Load the model:
   sim_load(xml_path="{model_path}", name="contact_debug")

2. Run contact debugging (captures snapshots on contact events):
   debug_contacts(
       n_steps={n_steps},
       penetration_threshold=0.005,
       capture_on_change=True
   )

3. Analyze the event log:
   - Look for "contact_made" / "contact_lost" events — when do they occur?
   - Check "penetration" events — which geom pairs are penetrating?
   - Review "force_exceeded" events if force_threshold was set

4. If penetration is a problem, try:
   modify_model(modifications=[
       {{"element": "geom", "name": "<geom_name>", "field": "margin", "value": 0.001}}
   ])

5. Re-run to verify the fix.
"""


@mcp.prompt()
async def validate_trajectory(
    trajectory_csv_path: str,
    model_path: str,
) -> str:
    """Guided prompt: validate a trajectory from an external planner."""
    return f"""Validate a planned trajectory against MuJoCo physics.

Steps to follow:
1. Load the model:
   sim_load(xml_path="{model_path}", name="validator")

2. Evaluate the trajectory:
   evaluate_trajectory(
       trajectory_csv="{trajectory_csv_path}",
       n_keyframes=6,
       energy_threshold=1.0,
       penetration_limit=0.01
   )

3. Interpret the plausibility score:
   - 100: No violations detected
   - 80–99: Minor issues (small energy jumps or brief limit violations)
   - < 80: Significant violations — review before using the trajectory

4. For each violation:
   - "energy_jump": Check for discontinuities in the planned qpos/qvel
   - "penetration": Check collision model or trajectory resolution
   - "joint_limit": Clip or re-plan the offending segment

5. If score is acceptable, use sim_record + sim_step to execute the trajectory
   and render a figure strip for verification:
   render_figure_strip(timestamps=[0.0, 0.5, 1.0, 1.5, 2.0])
"""


@mcp.prompt()
async def parameter_sensitivity(
    model_path: str,
    element: str = "geom",
    element_name: str = "box_geom",
    field: str = "friction",
) -> str:
    """Guided prompt: study how a model parameter affects simulation outcome."""
    return f"""Study parameter sensitivity: {element}/{element_name}/{field}.

Steps to follow:
1. Load the model:
   sim_load(xml_path="{model_path}", name="baseline")

2. Run baseline and record:
   sim_record(action="start")
   sim_step(n_steps=500)
   sim_record(action="stop")

3. Try a modified value:
   sim_reset()
   modify_model(modifications=[
       {{"element": "{element}", "name": "{element_name}", "field": "{field}", "value": <new_value>}}
   ])
   sim_record(action="clear")
   sim_record(action="start")
   # Load the modified result into a second slot for comparison:
   sim_load(xml_path="{model_path}", name="modified")
   modify_model(modifications=[...], sim_name="modified")
   sim_step(n_steps=500, sim_name="modified")
   sim_record(action="stop", sim_name="modified")

4. Compare trajectories:
   compare_trajectories(slot_a="baseline", slot_b="modified")

5. For a systematic sweep, use run_sweep (Phase 3):
   run_sweep(param="{element}.{element_name}.{field}", values=[...], n_steps=500)
"""
