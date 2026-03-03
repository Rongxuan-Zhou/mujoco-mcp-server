"""MCP Prompts — guided prompt templates for common simulation workflows."""

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


@mcp.prompt()
async def place_object(
    scene_slot: str = "default",
    object_description: str = "robot base",
    target_description: str = "table top, center of +X short edge",
) -> str:
    """Guided prompt: place an object using spatial reasoning tools (no manual coordinate math)."""
    return f"""Place {object_description} at {target_description} using spatial tools.

Standard placement workflow:
1. scene_map(sim_name="{scene_slot}")
   -> Identify all body names in the scene

2. body_aabb(body_name="<target_body>", sim_name="{scene_slot}")
   -> Get bounding box: min/max/center/size/top_z

3. surface_anchor(body_name="<target>", surface="top", anchor="+x", sim_name="{scene_slot}")
   -> Get exact world coordinates of the placement anchor point

4. compute_placement(target_body="<target>", surface="top", anchor="+x",
                     object_half_height=<half_height>, sim_name="{scene_slot}")
   -> Returns placement_pos accounting for object size

5. modify_model(modifications=[{{"element":"body","name":"<obj>","field":"pos","value":<placement_pos>}}])

6. render_snapshot() -> Verify visually
"""


@mcp.prompt()
async def load_and_control_robot(
    model_name: str = "franka_emika_panda",
    task: str = "move joints to home position",
) -> str:
    """Guided prompt: load a Menagerie robot and set up trajectory control."""
    return f"""Load {model_name} and control it to {task}.

Step 1 - Load model:
  load_menagerie_model(model_name="{model_name}", sim_name="robot")

Step 2 - Create controller:
  create_controller(robot_type="{model_name}", controller_kind="arm", sim_name="robot")

Step 3 - Get current joint positions:
  sim_get_state(sim_name="robot")

Step 4 - Plan trajectory:
  plan_trajectory(start_qpos=[...], end_qpos=[0,-0.785,0,-2.356,0,1.571,0.785],
                  duration=3.0, trajectory_type="min_jerk", sim_name="robot")

Step 5 - Execute:
  step_controller(n_steps=300, sim_name="robot")

Step 6 - Verify:
  render_snapshot(sim_name="robot")
"""


@mcp.prompt()
async def multi_robot_workflow(
    robot_count: int = 2,
    task: str = "formation_control",
) -> str:
    """Guided prompt: set up and coordinate multiple robots."""
    return f"""Coordinate {robot_count} robots in a {task} task.

Step 1 - Load robots into separate slots:
  load_menagerie_model(model_name="franka_emika_panda", sim_name="robot_1")
  load_menagerie_model(model_name="universal_robots_ur5e", sim_name="robot_2")

Step 2 - Register with coordinator:
  coordinator_add_robot(robot_id="r1", robot_type="franka_panda", capabilities={{"manipulation": true}})
  coordinator_add_robot(robot_id="r2", robot_type="ur5e", capabilities={{"manipulation": true}})

Step 3 - Check status:
  coordinator_get_status()

Step 4 - Assign task:
  coordinator_assign_task(task_type="formation_control", robot_ids=["r1", "r2"],
                          parameters={{"formation": "line", "spacing": 1.0}})

Step 5 - Check for collisions:
  coordinator_check_collisions()

Step 6 - Create controllers and move:
  create_controller(robot_type="franka_panda", controller_kind="arm", sim_name="robot_1")
  create_controller(robot_type="ur5e", controller_kind="arm", sim_name="robot_2")
"""
