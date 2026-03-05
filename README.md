# MuJoCo MCP Server

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/mujoco-%E2%89%A52.3-green.svg)](https://mujoco.org/)
[![Tests](https://img.shields.io/badge/tests-124%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MuJoCo physics simulation as **62 MCP tools** for [Claude Code](https://claude.ai/code) and other [Model Context Protocol](https://modelcontextprotocol.io) clients — run simulations, render frames, analyze physics, control robots, and sweep parameters directly from your AI assistant.

## Requirements

- Python ≥ 3.10, MuJoCo ≥ 2.3
- Linux with EGL (GPU) or OSMesa (CPU headless) for rendering
- [uv](https://docs.astral.sh/uv/) recommended

## Quick Start

```bash
git clone https://github.com/Rongxuan-Zhou/mujoco-mcp-server.git
cd mujoco-mcp-server
pip install -e .
```

### Add to Claude Code

```bash
claude mcp add mujoco-sim -- uv run --directory /path/to/mujoco-mcp-server mujoco-mcp
```

Or manually in `~/.claude.json`:

```json
{
  "mcpServers": {
    "mujoco-sim": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mujoco-mcp-server", "mujoco-mcp"],
      "env": { "MUJOCO_GL": "egl" }
    }
  }
}
```

### HTTP Transport (remote)

```bash
python -m mujoco_mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

## Tools

| Group | Tools | Description |
|---|---|---|
| **Simulation** | `sim_load` `sim_step` `sim_forward` `sim_reset` `sim_get_state` `sim_set_state` `sim_record` `sim_list` | Load MJCF/XML models and step physics |
| **Rendering** | `render_snapshot` `render_depth` | PNG snapshots and depth maps |
| **Analysis** | `analyze_contacts` `analyze_energy` `analyze_forces` `compute_jacobian` `compute_derivatives` `read_sensors` | Contacts, energy, forces, Jacobians, linearized dynamics |
| **Model** | `modify_model` `reload_from_xml` | In-place parameter edits (no recompile) or full XML reload |
| **Batch** | `run_sweep` | Parallel parameter sweeps via `ProcessPoolExecutor` |
| **Export** | `export_csv` `plot_data` | Save trajectories to CSV and plot |
| **Spatial** | `scene_map` `body_aabb` `surface_anchor` `compute_placement` | AABB, surface anchors, placement computation |
| **Menagerie** | `list_menagerie_models` `validate_menagerie_model` `load_menagerie_model` | Download and load 50+ robots from MuJoCo Menagerie |
| **Control** | `create_controller` `plan_trajectory` `step_controller` `get_controller_state` | PID + min-jerk trajectory for arms/quadrupeds/humanoids |
| **Sensor Fusion** | `configure_sensor_fusion` `get_fused_state` | Low-pass filtered joint state estimation |
| **Coordination** | `coordinator_add_robot` `coordinator_get_status` `coordinator_check_collisions` `coordinator_assign_task` | Multi-robot fleet management |
| **RL** | `create_rl_env` `rl_step` | Gymnasium-compatible RL environment wrapper |
| **Viewer** | `viewer_open` `viewer_sync` `viewer_close` | Live interactive viewer (requires display) |
| **Vision** *(optional)* | `analyze_scene` `compare_scenes` `track_object` `render_figure_strip` | Gemini 2.5 Pro scene analysis and trajectory tracking |
| **Meta** | `server_diagnostics` | Server health, GL backend, and loaded slots |
| **Kinematics** | `solve_ik` | Damped Least Squares IK for end-effector sites |
| **Optimization** | `optimize_ilqr` `optimize_mppi` | iLQR and MPPI trajectory optimization |
| **Robustness** | `apply_perturbation` `stability_analysis` `randomize_dynamics` | Perturbation robustness analysis and domain randomization |
| **Diagnostics** | `validate_mjcf` `model_summary` `suggest_contact_params` `diagnose_instability` | Pre-load XML validation, model overview, contact parameter tuning, instability detection |
| **Workflow** | `run_and_analyze` `debug_contacts` `evaluate_trajectory` `compare_trajectories` | Composite research workflows |

## Optional: Vision Analysis

```bash
pip install -e ".[vision]"
export GEMINI_API_KEY=your_key
```

Enables `analyze_scene`, `compare_scenes`, `track_object`, and `render_figure_strip` via Gemini 2.5 Pro.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MUJOCO_GL` | auto-probed | GL backend: `egl` (GPU) or `osmesa` (CPU) |
| `MUJOCO_MCP_RENDER_WIDTH` | `640` | Render width in pixels |
| `MUJOCO_MCP_RENDER_HEIGHT` | `480` | Render height in pixels |
| `MUJOCO_MCP_NO_RENDER` | `0` | Set `1` to skip GL init entirely |
| `MUJOCO_MCP_MAX_WORKERS` | `8` | Worker processes for `run_sweep` |
| `GEMINI_API_KEY` | — | Required for Vision tools |
| `GEMINI_VISION_MODEL` | `gemini-2.5-pro` | Override Gemini model |

## Development

```bash
pytest tests/                    # run tests
ruff check src/mujoco_mcp        # lint
pytest tests/test_sim_tools.py -v  # single file
```

## License

[MIT](LICENSE)
