# MuJoCo MCP Server

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/mujoco-%E2%89%A52.3-green.svg)](https://mujoco.org/)
[![Tests](https://img.shields.io/badge/tests-81%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MuJoCo physics simulation exposed as **MCP tools** for Claude Code and other MCP clients.

**53 tools** · multi-slot simulation · Menagerie robot library · optional Gemini vision analysis

---

## Features

- **Physics simulation** — load MJCF/XML, step, reset, record trajectories
- **Rendering** — PNG snapshots, depth maps, figure strips
- **Analysis** — contacts, energy, forces, Jacobians, linearized dynamics, sensors
- **Model modification** — in-place parameter edits (no recompile) and full XML reload
- **Batch sweeps** — parallel parameter sweeps via `ProcessPoolExecutor`
- **Spatial reasoning** — AABB, surface anchors, placement computation
- **Robot menagerie** — download and load 50+ robots from MuJoCo Menagerie
- **Control** — PID + min-jerk trajectory controller for arms/quadrupeds/humanoids
- **Sensor fusion** — low-pass filtered joint state estimation
- **Multi-robot coordination** — register fleets, assign cooperative tasks
- **RL environments** — Gymnasium-compatible environment wrapper
- **Vision analysis** — Gemini 2.5 Pro scene analysis and trajectory tracking *(optional)*
- **Workflow tools** — composite tools: `run_and_analyze`, `debug_contacts`, `evaluate_trajectory`, `compare_trajectories`

---

## Quick Start

### Install

```bash
pip install mujoco-mcp-server
```

Or in development mode:

```bash
git clone https://github.com/<your-org>/mujoco-mcp-server.git
cd mujoco-mcp-server
pip install -e .
```

### Run

```bash
# stdio (local, for Claude Code)
mujoco-mcp

# HTTP (remote)
python -m mujoco_mcp --transport streamable-http --host 0.0.0.0 --port 8080
```

### Claude Code Integration

Add to `~/.claude/claude_desktop_config.json`:

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

### Optional: Vision Analysis (Gemini)

```bash
pip install "mujoco-mcp-server[vision]"
export GEMINI_API_KEY=your_key
```

---

## Tools Reference

| Group | Tools | Description |
|---|---|---|
| **Simulation** | `sim_load` `sim_step` `sim_forward` `sim_reset` `sim_get_state` `sim_set_state` `sim_record` `sim_list` | Load models and step physics |
| **Rendering** | `render_snapshot` `render_depth` `render_figure_strip` | PNG images and depth maps |
| **Analysis** | `analyze_contacts` `analyze_energy` `analyze_forces` `compute_jacobian` `compute_derivatives` `read_sensors` | Physics diagnostics |
| **Model** | `modify_model` `reload_from_xml` | Parameter edits and structural reload |
| **Batch** | `run_sweep` | Parallel parameter sweeps |
| **Export** | `export_csv` `plot_data` | Save trajectories and plot data |
| **Spatial** | `scene_map` `body_aabb` `surface_anchor` `compute_placement` | Spatial queries and placement |
| **Menagerie** | `list_menagerie_models` `validate_menagerie_model` `load_menagerie_model` | MuJoCo Menagerie robot library |
| **Control** | `create_controller` `plan_trajectory` `step_controller` `get_controller_state` | PID + trajectory control |
| **Sensor Fusion** | `configure_sensor_fusion` `get_fused_state` | Filtered joint state estimation |
| **Coordination** | `coordinator_add_robot` `coordinator_get_status` `coordinator_check_collisions` `coordinator_assign_task` | Multi-robot fleet management |
| **RL** | `create_rl_env` `rl_step` | Gymnasium RL environment wrapper |
| **Vision** *(optional)* | `analyze_scene` `compare_scenes` `track_object` | Gemini 2.5 Pro visual analysis |
| **Workflow** | `run_and_analyze` `debug_contacts` `evaluate_trajectory` `compare_trajectories` | Composite research workflows |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MUJOCO_GL` | auto-probed | GL backend: `egl` (GPU) or `osmesa` (CPU) |
| `MUJOCO_MCP_RENDER_WIDTH` | `640` | Default render width in pixels |
| `MUJOCO_MCP_RENDER_HEIGHT` | `480` | Default render height in pixels |
| `MUJOCO_MCP_NO_RENDER` | `0` | Set `1` to skip GL initialisation entirely |
| `MUJOCO_MCP_MAX_WORKERS` | `8` | Worker processes for `run_sweep` |
| `GEMINI_API_KEY` | — | Required for Vision tools |
| `GEMINI_VISION_MODEL` | `gemini-2.5-pro` | Override Gemini model |

---

## Development

```bash
# Run tests
pytest tests/

# Lint
ruff check src/mujoco_mcp

# Single file
pytest tests/test_sim_tools.py -v
```

---

## License

[MIT](LICENSE)
