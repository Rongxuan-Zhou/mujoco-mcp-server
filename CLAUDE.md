# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MuJoCo MCP Server — a Model Context Protocol (MCP) server exposing MuJoCo physics simulation capabilities to Claude Code and other MCP clients. The complete specification lives in `mujoco_mcp_architecture_v2.0.md`.

- **Platform:** Linux x86_64 / aarch64
- **Python:** ≥ 3.10, **MuJoCo:** ≥ 2.3.0 (native bindings, not `mujoco-py`), **MCP SDK:** `mcp` ≥ 1.26 (FastMCP)
- **Transport:** stdio (local) or Streamable HTTP (remote)

## Commands

```bash
# Install in development mode
pip install -e .

# Run (stdio, local)
python -m mujoco_mcp --transport stdio

# Run (HTTP, remote)
python -m mujoco_mcp --transport streamable-http --host 0.0.0.0 --port 8080

# Tests
pytest tests/
pytest tests/test_sim_tools.py   # single file

# Type check / lint
mypy src/mujoco_mcp
ruff check src/mujoco_mcp
```

## Architecture

### Two-Tier Tool Design
- **Atomic tools (18):** 1-to-1 wrappers around MuJoCo primitives — in `tools/simulation.py`, `tools/rendering.py`, `tools/analysis.py`, `tools/model.py`, `tools/batch.py`, `tools/export.py`, `tools/meta.py`
- **Workflow tools (5):** Compose atomic tools for research tasks — in `tools/workflows.py`

### Planned Source Layout
```
src/mujoco_mcp/
├── server.py          # FastMCP app + lifespan context
├── compat.py          # MuJoCo version detection + API shims (set MUJOCO_GL BEFORE import)
├── sim_manager.py     # Multi-slot simulation manager (thread-safe)
├── tools/             # All @mcp.tool() registrations
├── resources.py       # MCP Resources
├── prompts.py         # MCP Prompts
└── utils/
    ├── gl_setup.py    # Probe EGL→OSMesa; sets MUJOCO_GL before mujoco is imported
    ├── rendering.py   # Rendering helpers
    ├── serialization.py  # numpy → JSON
    └── parallel.py    # ProcessPoolExecutor wrapper for run_sweep
```

### Key Patterns

**SimManager:** Named slots (`"default"`, `"exp1"`, …), each holding `model`, `data`, optional renderer, `xml_path`, and recorded `trajectory`. All operations are thread-safe.

**Tool registration:** All tools use `@mcp.tool()`. Context accessed via `ctx.request_context.lifespan_context.sim_manager`. Return `str` (JSON) or `list[TextContent | ImageContent]`.

**Error handling:** Wrap every tool body with a `@safe_tool` decorator catching `mujoco.FatalError`, `FileNotFoundError`, `ValueError`, `RuntimeError`, and generic `Exception` (last 400 chars of traceback). Always return a JSON error object, never raise.

**Version compat:** Feature flags set once at import in `compat.py`. `MUJOCO_GL` **must** be set before `import mujoco` executes — handled in `gl_setup.py` via subprocess probing (EGL first, then OSMesa).

**Model modification:** Use `modify_model()` for in-place numpy writes (no recompile); use `reload_from_xml()` for structural changes. Always call `mj_forward()` after state/parameter changes.

**Batch processing:** `run_sweep()` uses `ProcessPoolExecutor`; child processes never touch GL. Parameters addressed via dot notation (`"geom.box.friction"`, `"option.timestep"`).

### Environment Variables
| Variable | Default | Purpose |
|---|---|---|
| `MUJOCO_GL` | auto-probed | `egl` / `osmesa` — set before mujoco import |
| `MUJOCO_MCP_RENDER_WIDTH` | 640 | Render width |
| `MUJOCO_MCP_RENDER_HEIGHT` | 480 | Render height |
| `MUJOCO_MCP_NO_RENDER` | 0 | Set `1` to skip GL init entirely |
| `MUJOCO_MCP_MAX_WORKERS` | 8 | Workers for `run_sweep` |

### Claude Code Integration
```json
{
  "mcpServers": {
    "mujoco-sim": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mujoco-mcp-server", "mujoco-mcp"],
      "env": {"MUJOCO_GL": "egl"}
    }
  }
}
```

## Spec Reference

`mujoco_mcp_architecture_v2.0.md` is the authoritative source of truth. It supersedes v1.0–v1.2. Sections to note:
- **Sections 3–9:** Implementation-ready pseudocode for all modules
- **Section 13:** Phase 1 (MVP) implementation checklist — 12 files
- **Section 14:** Audit checklist — verify before testing
