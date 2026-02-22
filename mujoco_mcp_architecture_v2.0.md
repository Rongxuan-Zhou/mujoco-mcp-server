# MuJoCo MCP Server — Architecture v2.0 (Final)

**Platform**: Linux x86_64 / aarch64  
**MuJoCo**: ≥ 2.3.0 native Python bindings (`import mujoco`, NOT mujoco-py)  
**MCP SDK**: Python `mcp` ≥ 1.26 (FastMCP)  
**Python**: ≥ 3.10  
**Transport**: stdio (local) / Streamable HTTP (remote)

This is the single source of truth. Supersedes v1.0, v1.1, and v1.2 patch.

---

## 1. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    MCP CLIENT  (Claude Code)                     │
│            Natural language → tool calls → results               │
└───────────────────────────┬──────────────────────────────────────┘
                            │ MCP (stdio / streamable-http)
┌───────────────────────────▼──────────────────────────────────────┐
│                    MuJoCo MCP Server                             │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Version Compat Layer (compat.py)            │  │
│  │   runtime feature flags → route to correct MuJoCo calls    │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────── WORKFLOW TOOLS (mid-level) ──────────────────┐  │
│  │ run_and_analyze · debug_contacts · evaluate_trajectory     │  │
│  │ compare_trajectories · render_figure_strip                 │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │ composed from                     │
│  ┌───────────── ATOMIC TOOLS (low-level) ─────────────────────┐  │
│  │ Sim: load step forward reset get/set_state record list     │  │
│  │ Render: snapshot depth                                     │  │
│  │ Analysis: contacts jacobian derivatives sensors energy      │  │
│  │           forces                                           │  │
│  │ Model: modify_model reload_from_xml                        │  │
│  │ Batch: run_sweep                                           │  │
│  │ Export: export_csv plot_data                                │  │
│  │ Meta: server_diagnostics                                   │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │  SimManager (multi-slot) + GL Context + Renderer per slot  │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼────────────────────────────────┐  │
│  │        MuJoCo C engine (via pybind11): MjModel MjData      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### Design principles

| Principle | Rationale |
|-----------|-----------|
| **Two-tier tools** | Atomic tools wrap MuJoCo 1-to-1; workflow tools combine them for real research tasks |
| **Rendering = supplementary** | Claude judges trajectories primarily via numerical time-series; images confirm |
| **Event-driven capture** | Snapshots triggered by physics events (contact change, force spike), not fixed intervals |
| **Graceful rendering degradation** | All sim/analysis tools work without GL; render tools return clear error |
| **Direct MjModel writes** | Parameter modification via numpy array writes, never XML reparse |
| **Version compat at runtime** | Feature flags set once at import; all tools route through compat layer |

---

## 2. Project Structure

```
mujoco-mcp-server/
├── pyproject.toml
├── README.md
├── src/
│   └── mujoco_mcp/
│       ├── __init__.py
│       ├── __main__.py                 # python -m mujoco_mcp
│       ├── server.py                   # FastMCP + lifespan
│       ├── compat.py                   # Version detection + API shims
│       ├── sim_manager.py              # SimSlot, multi-model, renderer lifecycle
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── simulation.py           # sim_load/step/forward/reset/get_state/set_state/record/list
│       │   ├── rendering.py            # render_snapshot, render_depth
│       │   ├── analysis.py             # analyze_contacts, compute_jacobian, compute_derivatives,
│       │   │                           #   read_sensors, analyze_energy, analyze_forces
│       │   ├── model.py                # modify_model, reload_from_xml
│       │   ├── batch.py                # run_sweep
│       │   ├── export.py               # export_csv, plot_data
│       │   ├── meta.py                 # server_diagnostics
│       │   └── workflows.py            # run_and_analyze, debug_contacts,
│       │                               #   evaluate_trajectory, compare_trajectories,
│       │                               #   render_figure_strip
│       ├── resources.py                # MCP Resources
│       ├── prompts.py                  # MCP Prompts
│       └── utils/
│           ├── __init__.py
│           ├── gl_setup.py             # Linux GL backend subprocess probe
│           ├── rendering.py            # render_to_base64, depth helpers
│           ├── serialization.py        # numpy→JSON
│           └── parallel.py             # ProcessPoolExecutor wrapper
├── models/
│   ├── box_drop.xml
│   └── two_body_contact.xml
└── tests/
    ├── conftest.py
    ├── test_compat.py
    ├── test_sim_tools.py
    ├── test_render_tools.py
    ├── test_analysis_tools.py
    └── test_workflows.py
```

---

## 3. Version Compatibility

### 3.1 Supported range

| Version | Status |
|---------|--------|
| 3.2 – 3.5 | Full support (recommended) |
| 3.0 – 3.1 | Full support |
| 2.3.0 – 2.3.7 | Supported with shims |
| < 2.3 or mujoco-py | Not supported |

### 3.2 `compat.py`

```python
"""Runtime feature detection. Import once, use everywhere."""

import mujoco
import numpy as np
from packaging.version import Version
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MUJOCO_VERSION = Version(mujoco.__version__.split(".post")[0])

# Feature flags (evaluated once at import)
HAS_RENDERER_CTX_MGR   = MUJOCO_VERSION >= Version("3.2.0")
HAS_CAMERA_STRING_NAME = MUJOCO_VERSION >= Version("3.0.0")
HAS_SCENE_OPTION_KWARG = MUJOCO_VERSION >= Version("2.3.3")

def get_version_info() -> dict:
    return {
        "mujoco_version": str(MUJOCO_VERSION),
        "renderer_ctx_mgr": HAS_RENDERER_CTX_MGR,
        "camera_string_name": HAS_CAMERA_STRING_NAME,
        "scene_option_kwarg": HAS_SCENE_OPTION_KWARG,
    }

# --- Renderer lifecycle (NEVER use `with`) ---

def create_renderer(model: mujoco.MjModel,
                    width: int = 640, height: int = 480) -> mujoco.Renderer:
    """Version-safe renderer creation. Caller must call .close() explicitly."""
    return mujoco.Renderer(model, height=height, width=width)

# --- Camera resolution (always int ID) ---

def resolve_camera(model: mujoco.MjModel, camera: Optional[str]) -> int:
    """Name → int ID. Returns -1 for free camera."""
    if camera is None:
        return -1
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        available = list_named(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        raise ValueError(f"Camera '{camera}' not found. Available: {available}")
    return cam_id

# --- Scene update (conditional kwargs) ---

def update_scene(renderer: mujoco.Renderer,
                 data: mujoco.MjData,
                 camera_id: int = -1,
                 scene_option: Optional[mujoco.MjvOption] = None):
    kw = {}
    if camera_id >= 0:
        kw["camera"] = camera_id
    if scene_option is not None and HAS_SCENE_OPTION_KWARG:
        kw["scene_option"] = scene_option
    renderer.update_scene(data, **kw)

# --- Name listing ---

def list_named(model: mujoco.MjModel, obj_type: int, count: int) -> list[str]:
    out = []
    for i in range(count):
        n = mujoco.mj_id2name(model, obj_type, i)
        if n:
            out.append(n)
    return out

def resolve_name(model: mujoco.MjModel, obj_type: int, name: str,
                 type_label: str) -> int:
    """Name → int ID with descriptive error on miss."""
    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid < 0:
        count_map = {
            mujoco.mjtObj.mjOBJ_BODY: model.nbody,
            mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
            mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
            mujoco.mjtObj.mjOBJ_SITE: model.nsite,
            mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
            mujoco.mjtObj.mjOBJ_SENSOR: model.nsensor,
            mujoco.mjtObj.mjOBJ_CAMERA: model.ncam,
        }
        available = list_named(model, obj_type, count_map.get(obj_type, 0))
        raise ValueError(f"{type_label} '{name}' not found. Available: {available}")
    return oid
```

---

## 4. Linux GL Backend Detection

**Critical rule**: `MUJOCO_GL` is read once when `import mujoco` first executes. Cannot change after.

```python
# src/mujoco_mcp/utils/gl_setup.py
"""Must be called BEFORE import mujoco anywhere in the process."""

import os, sys, subprocess, logging

logger = logging.getLogger(__name__)

def detect_and_set_gl_backend() -> str:
    """Probe EGL then OSMesa in subprocesses. Set env var for main process."""
    if "MUJOCO_GL" in os.environ:
        logger.info(f"GL backend from env: {os.environ['MUJOCO_GL']}")
        return os.environ["MUJOCO_GL"]

    for backend in ("egl", "osmesa"):
        if _probe_backend(backend):
            os.environ["MUJOCO_GL"] = backend
            logger.info(f"Auto-detected GL backend: {backend}")
            return backend

    logger.warning("No GL backend found. Rendering will be disabled.")
    return "none"

def _probe_backend(backend: str) -> bool:
    code = (
        f"import os; os.environ['MUJOCO_GL']='{backend}'; "
        "import mujoco; "
        "ctx=mujoco.GLContext(64,64); ctx.make_current(); ctx.free(); "
        "print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip() == "OK"
    except Exception:
        return False

def get_gl_diagnostics() -> dict:
    return {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "unset"),
        "egl_probe": _probe_backend("egl"),
        "osmesa_probe": _probe_backend("osmesa"),
        "nvidia_gpu": _nvidia_gpu(),
    }

def _nvidia_gpu() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "not found"
    except FileNotFoundError:
        return "nvidia-smi unavailable"
```

---

## 5. Server Entry Point

```python
# src/mujoco_mcp/server.py
"""
IMPORT ORDER IS CRITICAL:
1. detect_and_set_gl_backend()  ← sets MUJOCO_GL env var
2. import mujoco                ← reads MUJOCO_GL once
3. everything else
"""

import os, json, logging

from .utils.gl_setup import detect_and_set_gl_backend
_gl_backend = detect_and_set_gl_backend()            # STEP 1

import mujoco                                         # STEP 2

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context

from .sim_manager import SimManager
from .compat import get_version_info

logger = logging.getLogger(__name__)

@dataclass
class AppContext:
    sim_manager: SimManager

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    ver = get_version_info()
    logger.info(f"MuJoCo MCP Server starting: {ver}")

    manager = SimManager(
        enable_rendering=(_gl_backend != "none"),
        render_width=int(os.environ.get("MUJOCO_MCP_RENDER_WIDTH", "640")),
        render_height=int(os.environ.get("MUJOCO_MCP_RENDER_HEIGHT", "480")),
    )
    manager.init_rendering()

    try:
        yield AppContext(sim_manager=manager)
    finally:
        manager.cleanup()

mcp_server = FastMCP(
    "mujoco-sim",
    dependencies=["mujoco>=2.3.0", "numpy", "Pillow", "pandas",
                   "matplotlib", "packaging"],
    lifespan=app_lifespan,
)

# Register all tool modules
from .tools import simulation, rendering, analysis, model, batch, export, meta, workflows
from . import resources, prompts

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--transport", default="stdio",
                   choices=["stdio", "streamable-http"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()
    if args.transport == "stdio":
        mcp_server.run(transport="stdio")
    else:
        mcp_server.run(transport="streamable-http",
                       host=args.host, port=args.port)
```

```python
# src/mujoco_mcp/__main__.py
from .server import main
main()
```

---

## 6. SimManager

```python
# src/mujoco_mcp/sim_manager.py

import mujoco, numpy as np, threading, logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .compat import create_renderer, list_named, MUJOCO_VERSION

logger = logging.getLogger(__name__)

@dataclass
class SimSlot:
    name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: Optional[mujoco.Renderer] = None
    xml_path: Optional[str] = None
    xml_string: Optional[str] = None        # kept for reload / batch
    trajectory: list = field(default_factory=list)
    recording: bool = False

class SimManager:
    def __init__(self, enable_rendering=True,
                 render_width=640, render_height=480):
        self.slots: dict[str, SimSlot] = {}
        self.active_slot: Optional[str] = None
        self.render_width = render_width
        self.render_height = render_height
        self._rendering_enabled = enable_rendering
        self._rendering_available = False
        self._gl_context: Optional[mujoco.GLContext] = None
        self._lock = threading.Lock()

    # ---- lifecycle ----

    def init_rendering(self):
        if not self._rendering_enabled:
            return
        try:
            self._gl_context = mujoco.GLContext(
                self.render_width, self.render_height)
            self._gl_context.make_current()
            self._rendering_available = True
            logger.info("GL rendering ready")
        except Exception as e:
            self._rendering_available = False
            logger.warning(f"Rendering unavailable: {e}")

    @property
    def can_render(self) -> bool:
        return self._rendering_available

    def cleanup(self):
        for s in self.slots.values():
            if s.renderer:
                s.renderer.close()
        self.slots.clear()
        if self._gl_context:
            self._gl_context.free()
            self._gl_context = None

    # ---- slot management ----

    def load(self, name: str, *,
             xml_path: str | None = None,
             xml_string: str | None = None) -> dict:
        # close old slot if reloading same name
        if name in self.slots:
            old = self.slots.pop(name)
            if old.renderer:
                old.renderer.close()

        if xml_path:
            p = Path(xml_path)
            if not p.exists():
                raise FileNotFoundError(f"MJCF not found: {xml_path}")
            model = mujoco.MjModel.from_xml_path(str(p))
            stored_xml = p.read_text()
        elif xml_string:
            model = mujoco.MjModel.from_xml_string(xml_string)
            stored_xml = xml_string
        else:
            raise ValueError("Provide xml_path or xml_string")

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = None
        if self._rendering_available:
            try:
                renderer = create_renderer(model,
                    self.render_width, self.render_height)
            except Exception as e:
                logger.warning(f"Renderer failed for '{name}': {e}")

        slot = SimSlot(name=name, model=model, data=data,
                       renderer=renderer, xml_path=xml_path,
                       xml_string=stored_xml)
        self.slots[name] = slot
        self.active_slot = name
        return self._summary(slot)

    def get(self, name: str | None = None) -> SimSlot:
        target = name or self.active_slot
        if target is None:
            raise ValueError("No simulation loaded. Call sim_load first.")
        if target not in self.slots:
            raise ValueError(
                f"Slot '{target}' not found. Available: {list(self.slots.keys())}")
        return self.slots[target]

    def require_renderer(self, slot: SimSlot) -> mujoco.Renderer:
        if slot.renderer is None:
            raise RuntimeError(
                "Rendering unavailable on this node.\n"
                "Run server_diagnostics for details.")
        return slot.renderer

    def _summary(self, s: SimSlot) -> dict:
        m = s.model
        obj = mujoco.mjtObj
        return {
            "name": s.name,
            "mujoco_version": str(MUJOCO_VERSION),
            "nq": m.nq, "nv": m.nv, "nu": m.nu,
            "nbody": m.nbody, "ngeom": m.ngeom, "njnt": m.njnt,
            "nsite": m.nsite, "nsensor": m.nsensor, "ncam": m.ncam,
            "timestep": m.opt.timestep,
            "has_renderer": s.renderer is not None,
            "bodies":    list_named(m, obj.mjOBJ_BODY, m.nbody),
            "joints":    list_named(m, obj.mjOBJ_JOINT, m.njnt),
            "actuators": list_named(m, obj.mjOBJ_ACTUATOR, m.nu),
            "sensors":   list_named(m, obj.mjOBJ_SENSOR, m.nsensor),
            "cameras":   list_named(m, obj.mjOBJ_CAMERA, m.ncam),
        }
```

---

## 7. Tool Inventory (27 tools)

### 7.1 Atomic (low-level) — 18 tools

| # | Tool | File | GPU | Scope |
|---|------|------|-----|-------|
| 1 | `sim_load` | simulation.py | — | Load MJCF into named slot |
| 2 | `sim_step` | simulation.py | — | Step physics (max 100 000) |
| 3 | `sim_forward` | simulation.py | — | Forward dynamics, no time advance |
| 4 | `sim_reset` | simulation.py | — | Reset to t=0 |
| 5 | `sim_get_state` | simulation.py | — | Read qpos/qvel/bodies/sites |
| 6 | `sim_set_state` | simulation.py | — | Write qpos/qvel/ctrl or load keyframe |
| 7 | `sim_record` | simulation.py | — | start/stop/clear trajectory recording |
| 8 | `sim_list` | simulation.py | — | List all loaded slots |
| 9 | `render_snapshot` | rendering.py | Yes* | Single frame as PNG |
| 10 | `render_depth` | rendering.py | Yes* | Depth map as grayscale PNG |
| 11 | `analyze_contacts` | analysis.py | — | Contact pairs, forces, penetration |
| 12 | `compute_jacobian` | analysis.py | — | 6×nv Jacobian + SVD + manipulability |
| 13 | `compute_derivatives` | analysis.py | — | Linearized dynamics A B C D |
| 14 | `read_sensors` | analysis.py | — | Named sensor values |
| 15 | `analyze_energy` | analysis.py | — | KE, PE, per-body |
| 16 | `analyze_forces` | analysis.py | — | Applied, constraint, passive, actuator forces |
| 17 | `modify_model` | model.py | — | Direct numpy writes to MjModel fields |
| 18 | `reload_from_xml` | model.py | — | Full recompile from XML (structural changes) |

### 7.2 Workflow (mid-level) — 5 tools

| # | Tool | Scope |
|---|------|-------|
| 19 | `run_and_analyze` | Step N times → time-series + keyframe images |
| 20 | `debug_contacts` | Event-driven contact capture + force traces |
| 21 | `evaluate_trajectory` | Replay external trajectory → plausibility report |
| 22 | `compare_trajectories` | A/B comparison across two slots |
| 23 | `render_figure_strip` | Multi-timestamp frames for paper figures |

### 7.3 Batch / Export / Meta — 4 tools

| # | Tool | Scope |
|---|------|-------|
| 24 | `run_sweep` | Parallel parameter sweep → CSV + summary |
| 25 | `export_csv` | Trajectory → CSV file |
| 26 | `plot_data` | CSV → matplotlib PNG returned inline |
| 27 | `server_diagnostics` | Version, GL, loaded slots |

*Rendering tools work on OSMesa (no GPU) but slower.

---

## 8. Atomic Tool Implementations

### 8.1 simulation.py (tools 1–8)

```python
import json, mujoco, numpy as np
from ..server import mcp_server as mcp
from mcp.server.fastmcp import Context

MAX_STEPS = 100_000

@mcp.tool()
async def sim_load(ctx: Context, xml_path: str | None = None,
                   xml_string: str | None = None,
                   name: str = "default") -> str:
    """Load MuJoCo MJCF model into a named slot."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    summary = mgr.load(name, xml_path=xml_path, xml_string=xml_string)
    return json.dumps(summary, indent=2)

@mcp.tool()
async def sim_step(ctx: Context, n_steps: int = 1,
                   ctrl: list[float] | None = None,
                   sim_name: str | None = None) -> str:
    """Advance physics. Max 100 000 steps per call."""
    if not 1 <= n_steps <= MAX_STEPS:
        return json.dumps({"error": f"n_steps must be 1–{MAX_STEPS}"})
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    if ctrl is not None:
        if len(ctrl) != m.nu:
            return json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"})
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)
    for _ in range(n_steps):
        mujoco.mj_step(m, d)
        if slot.recording:
            slot.trajectory.append({
                "t": float(d.time),
                "qpos": d.qpos.copy().tolist(),
                "qvel": d.qvel.copy().tolist(),
            })
    return json.dumps({
        "time": d.time, "qpos": d.qpos.tolist(),
        "qvel": d.qvel.tolist(),
        "energy": [float(d.energy[0]), float(d.energy[1])],
        "n_contacts": int(d.ncon),
    }, indent=2)

@mcp.tool()
async def sim_forward(ctx: Context,
                      sim_name: str | None = None) -> str:
    """Recompute derived quantities (positions, forces, contacts)
    without advancing time. Call after sim_set_state or modify_model."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    mujoco.mj_forward(slot.model, slot.data)
    d = slot.data
    return json.dumps({"time": d.time, "n_contacts": int(d.ncon),
                       "energy": [float(d.energy[0]), float(d.energy[1])]})

@mcp.tool()
async def sim_reset(ctx: Context, sim_name: str | None = None) -> str:
    """Reset simulation to t=0, default qpos/qvel."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    mujoco.mj_resetData(slot.model, slot.data)
    mujoco.mj_forward(slot.model, slot.data)
    slot.trajectory.clear()
    slot.recording = False
    return json.dumps({"status": "reset", "time": 0.0})

@mcp.tool()
async def sim_get_state(ctx: Context, include_bodies: bool = False,
                        include_sites: bool = False,
                        sim_name: str | None = None) -> str:
    """Read current simulation state."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    state = {"time": d.time, "qpos": d.qpos.tolist(),
             "qvel": d.qvel.tolist(), "ctrl": d.ctrl.tolist()}
    if include_bodies:
        state["body_xpos"] = {
            n: d.body(n).xpos.tolist()
            for n in _iter_named(m, mujoco.mjtObj.mjOBJ_BODY, m.nbody)
        }
    if include_sites:
        state["site_xpos"] = {
            n: d.site(n).xpos.tolist()
            for n in _iter_named(m, mujoco.mjtObj.mjOBJ_SITE, m.nsite)
        }
    return json.dumps(state, indent=2)

@mcp.tool()
async def sim_set_state(ctx: Context,
                        qpos: list[float] | None = None,
                        qvel: list[float] | None = None,
                        ctrl: list[float] | None = None,
                        keyframe: int | None = None,
                        sim_name: str | None = None) -> str:
    """Write simulation state. Calls mj_forward automatically."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    if keyframe is not None:
        mujoco.mj_resetDataKeyframe(m, d, keyframe)
    else:
        if qpos is not None: d.qpos[:] = np.array(qpos)
        if qvel is not None: d.qvel[:] = np.array(qvel)
        if ctrl is not None: d.ctrl[:] = np.array(ctrl)
    mujoco.mj_forward(m, d)
    return json.dumps({"status": "ok", "time": d.time})

@mcp.tool()
async def sim_record(ctx: Context, action: str = "start",
                     sim_name: str | None = None) -> str:
    """Start/stop/clear trajectory recording.
    While recording, every sim_step appends qpos/qvel/time."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    if action == "start":
        slot.recording = True
    elif action == "stop":
        slot.recording = False
    elif action == "clear":
        slot.trajectory.clear(); slot.recording = False
    else:
        return json.dumps({"error": "action must be start/stop/clear"})
    return json.dumps({"recording": slot.recording,
                       "frames": len(slot.trajectory)})

@mcp.tool()
async def sim_list(ctx: Context) -> str:
    """List all loaded simulation slots."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    return json.dumps({
        "active": mgr.active_slot,
        "slots": {n: {"nq": s.model.nq, "nv": s.model.nv,
                       "time": s.data.time,
                       "recording": s.recording,
                       "traj_frames": len(s.trajectory),
                       "has_renderer": s.renderer is not None}
                  for n, s in mgr.slots.items()},
    }, indent=2)

def _iter_named(m, obj_type, count):
    for i in range(count):
        n = mujoco.mj_id2name(m, obj_type, i)
        if n and n != "world":
            yield n
```

### 8.2 rendering.py (tools 9–10)

```python
import json, base64, mujoco, numpy as np
from io import BytesIO
from PIL import Image
from mcp.types import TextContent, ImageContent
from mcp.server.fastmcp import Context
from ..server import mcp_server as mcp
from ..compat import resolve_camera, update_scene

@mcp.tool()
async def render_snapshot(ctx: Context, camera: str | None = None,
                          width: int | None = None, height: int | None = None,
                          show_contacts: bool = False,
                          sim_name: str | None = None) -> list:
    """Render current state as PNG image."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    renderer = mgr.require_renderer(slot)
    m, d = slot.model, slot.data

    cam_id = resolve_camera(m, camera)
    opt = None
    if show_contacts:
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    update_scene(renderer, d, camera_id=cam_id, scene_option=opt)
    pixels = renderer.render()

    img = Image.fromarray(pixels)
    if width or height:
        img = img.resize((width or img.width, height or img.height), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        ImageContent(type="image", data=b64, mimeType="image/png"),
        TextContent(type="text",
            text=f"t={d.time:.4f}s | {d.ncon} contacts | "
                 f"E=[{d.energy[0]:.3f}, {d.energy[1]:.3f}]"),
    ]

@mcp.tool()
async def render_depth(ctx: Context, camera: str | None = None,
                       sim_name: str | None = None) -> list:
    """Render depth map as grayscale PNG."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    renderer = mgr.require_renderer(slot)
    cam_id = resolve_camera(slot.model, camera)

    renderer.enable_depth_rendering(True)
    update_scene(renderer, slot.data, camera_id=cam_id)
    depth = renderer.render()
    renderer.enable_depth_rendering(False)

    d_norm = ((1.0 - depth) * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(d_norm, mode="L")
    buf = BytesIO(); img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return [
        ImageContent(type="image", data=b64, mimeType="image/png"),
        TextContent(type="text",
            text=f"Depth: min={float(depth.min()):.3f} max={float(depth.max()):.3f}"),
    ]
```

### 8.3 analysis.py (tools 11–16)

```python
import json, mujoco, numpy as np
from mcp.server.fastmcp import Context
from ..server import mcp_server as mcp
from ..compat import resolve_name, list_named

JACOBIAN_NV_THRESHOLD = 50

@mcp.tool()
async def analyze_contacts(ctx: Context, max_contacts: int = 20,
                           sim_name: str | None = None) -> str:
    """Active contacts: geom pairs, forces, penetration depth."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    contacts = []
    for i in range(min(d.ncon, max_contacts)):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
        force = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, force)
        contacts.append({
            "geom1": g1, "geom2": g2,
            "pos": c.pos.tolist(),
            "dist": float(c.dist),
            "normal_force": float(force[0]),
            "friction_force": force[1:3].tolist(),
            "condim": int(c.dim),
        })
    return json.dumps({"n_contacts": d.ncon, "contacts": contacts}, indent=2)

@mcp.tool()
async def compute_jacobian(ctx: Context, target: str,
                           target_type: str = "site",
                           full_matrix: bool = False,
                           sim_name: str | None = None) -> str:
    """End-effector Jacobian (6×nv). Auto-truncates for large models."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))

    type_map = {"site": mujoco.mjtObj.mjOBJ_SITE,
                "body": mujoco.mjtObj.mjOBJ_BODY,
                "geom": mujoco.mjtObj.mjOBJ_GEOM}
    fn_map = {"site": mujoco.mj_jacSite,
              "body": mujoco.mj_jacBody,
              "geom": mujoco.mj_jacGeom}
    if target_type not in type_map:
        return json.dumps({"error": f"target_type must be site/body/geom"})

    oid = resolve_name(m, type_map[target_type], target, target_type)
    fn_map[target_type](m, d, jacp, jacr, oid)

    J = np.vstack([jacp, jacr])
    sv = np.linalg.svd(J, compute_uv=False)
    rank = int(np.sum(sv > 1e-10))
    manip = float(np.prod(sv[sv > 1e-10])) if rank > 0 else 0.0
    cond = float(sv[0] / sv[max(rank-1,0)]) if rank > 0 else float("inf")

    pos_accessor = {"site": lambda: d.site(target).xpos,
                    "body": lambda: d.body(target).xpos,
                    "geom": lambda: d.geom(target).xpos}

    result = {"target": target, "target_type": target_type, "nv": m.nv,
              "position": pos_accessor[target_type]().tolist(),
              "rank": rank, "singular_values": sv.tolist(),
              "manipulability": manip, "condition_number": cond}

    if m.nv <= JACOBIAN_NV_THRESHOLD or full_matrix:
        result["jacp"] = jacp.tolist()
        result["jacr"] = jacr.tolist()
    else:
        result["note"] = f"Matrix omitted (nv={m.nv}>{JACOBIAN_NV_THRESHOLD}). Set full_matrix=True."
    return json.dumps(result, indent=2)

@mcp.tool()
async def compute_derivatives(ctx: Context,
                              sim_name: str | None = None) -> str:
    """Linearized transition: x_next = A x + B u. Uses mjd_transitionFD."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    nv, nu, ns = m.nv, m.nu, m.nsensordata
    A = np.zeros((2*nv, 2*nv)); B = np.zeros((2*nv, nu))
    C = np.zeros((ns, 2*nv)) if ns else None
    D = np.zeros((ns, nu)) if ns else None
    mujoco.mjd_transitionFD(m, d, 1e-6, True, A, B, C, D)
    eigs = np.abs(np.linalg.eigvals(A))
    res = {"A_shape": [2*nv, 2*nv], "B_shape": [2*nv, nu],
           "max_eig_magnitude": float(eigs.max()),
           "discrete_stable": bool(eigs.max() < 1.0)}
    if 2*nv <= 40:
        res["A"] = A.tolist(); res["B"] = B.tolist()
    return json.dumps(res, indent=2)

@mcp.tool()
async def read_sensors(ctx: Context, sensor_names: list[str] | None = None,
                       sim_name: str | None = None) -> str:
    """Read named sensor values (or all sensors)."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    result = {}
    indices = range(m.nsensor)
    if sensor_names:
        indices = [resolve_name(m, mujoco.mjtObj.mjOBJ_SENSOR, n, "sensor")
                   for n in sensor_names]
    for i in (indices if sensor_names is None else range(len(sensor_names))):
        sid = i if sensor_names is None else indices[i]
        name = sensor_names[i] if sensor_names else mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_SENSOR, sid)
        if name:
            adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
            result[name] = d.sensordata[adr:adr+dim].tolist()
    return json.dumps(result, indent=2)

@mcp.tool()
async def analyze_energy(ctx: Context, sim_name: str | None = None) -> str:
    """Potential energy, kinetic energy, total."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    d = slot.data
    return json.dumps({
        "potential": float(d.energy[0]),
        "kinetic": float(d.energy[1]),
        "total": float(d.energy[0] + d.energy[1]),
    })

@mcp.tool()
async def analyze_forces(ctx: Context, sim_name: str | None = None) -> str:
    """Joint-space force decomposition."""
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    res = {"qfrc_applied": d.qfrc_applied.tolist(),
           "qfrc_constraint": d.qfrc_constraint.tolist(),
           "qfrc_passive": d.qfrc_passive.tolist(),
           "qfrc_bias": d.qfrc_bias.tolist(),
           "qfrc_actuator": d.qfrc_actuator.tolist(),
           "qacc": d.qacc.tolist()}
    if m.nu > 0:
        res["actuator_force"] = d.actuator_force.tolist()
    return json.dumps(res, indent=2)
```

### 8.4 model.py (tools 17–18)

```python
import json, mujoco, numpy as np
from mcp.server.fastmcp import Context
from ..server import mcp_server as mcp

@mcp.tool()
async def modify_model(ctx: Context, modifications: list[dict],
                       sim_name: str | None = None) -> str:
    """Modify compiled MjModel fields in-place via numpy writes.
    No recompilation. Changes take effect after auto mj_forward.

    Each entry: {element, name, field, value}
    element = "geom"|"joint"|"body"|"actuator"|"site"|"option"
    For "option": omit name, use field/value on model.opt directly.

    Examples:
      {"element":"geom","name":"box","field":"friction","value":[0.5,0.005,0.001]}
      {"element":"option","field":"timestep","value":0.002}
    """
    slot = ctx.request_context.lifespan_context.sim_manager.get(sim_name)
    m, d = slot.model, slot.data
    results = []
    for mod in modifications:
        elem, field, value = mod["element"], mod["field"], mod["value"]
        try:
            if elem == "option":
                attr = getattr(m.opt, field)
                old = attr.copy().tolist() if isinstance(attr, np.ndarray) else float(attr)
                if isinstance(attr, np.ndarray):
                    attr[:] = np.asarray(value)
                else:
                    setattr(m.opt, field, value)
                results.append({"elem": elem, "field": field, "old": old, "new": value})
            else:
                name = mod["name"]
                # joint → jnt alias
                accessor = getattr(m, "jnt" if elem == "joint" else elem)(name)
                attr = getattr(accessor, field)
                old = attr.copy().tolist() if isinstance(attr, np.ndarray) else float(attr)
                if isinstance(attr, np.ndarray):
                    attr[:] = np.asarray(value)
                else:
                    setattr(accessor, field, value)
                results.append({"elem": elem, "name": name, "field": field,
                                "old": old, "new": value})
        except Exception as e:
            results.append({"elem": elem, "field": field, "error": str(e)})
    mujoco.mj_forward(m, d)
    return json.dumps({"modifications": results}, indent=2)

@mcp.tool()
async def reload_from_xml(ctx: Context, xml_string: str,
                          sim_name: str | None = None) -> str:
    """Full reload from XML. Use only for structural changes
    (add/remove bodies). Resets simulation state."""
    mgr = ctx.request_context.lifespan_context.sim_manager
    name = sim_name or mgr.active_slot or "default"
    summary = mgr.load(name, xml_string=xml_string)
    return json.dumps(summary, indent=2)
```

---

## 9. Workflow Tool Implementations

These are the research-daily-driver tools. They compose atomic tools internally.

### 9.1 `run_and_analyze`

```python
@mcp.tool()
async def run_and_analyze(
    ctx: Context,
    n_steps: int = 1000,
    ctrl: list[float] | None = None,
    capture_every_n: int = 200,
    track: list[str] | None = None,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Run simulation and return time-series data + keyframe images.

    This is the primary "observe a trajectory" tool. Returns
    physics data at EVERY timestep plus sparse visual keyframes.

    Args:
        n_steps: Total steps (max 100k).
        ctrl: Constant control vector. None = current.
        capture_every_n: Render keyframe every N steps.
        track: Quantities to record per step. Options:
          "qpos", "qvel", "energy", "contact_count",
          "sensor:<name>", "body_xpos:<name>"
          Default: ["energy", "contact_count"]
        camera: Camera name for keyframes.

    Returns:
        TextContent with JSON time-series,
        then ImageContent for each keyframe.
    """
    from mcp.types import TextContent, ImageContent

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data
    track = track or ["energy", "contact_count"]

    if ctrl is not None:
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    timeseries = []
    keyframes = []
    cam_id = resolve_camera(m, camera) if camera else -1

    for step in range(min(n_steps, MAX_STEPS)):
        mujoco.mj_step(m, d)

        # Record tracked quantities every step
        row = {"t": round(d.time, 6)}
        for t in track:
            if t == "energy":
                row["E_pot"] = float(d.energy[0])
                row["E_kin"] = float(d.energy[1])
            elif t == "contact_count":
                row["ncon"] = int(d.ncon)
            elif t == "qpos":
                row["qpos"] = d.qpos.tolist()
            elif t == "qvel":
                row["qvel"] = d.qvel.tolist()
            elif t.startswith("sensor:"):
                sname = t.split(":", 1)[1]
                sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, sname)
                if sid >= 0:
                    adr, dim = m.sensor_adr[sid], m.sensor_dim[sid]
                    row[t] = d.sensordata[adr:adr+dim].tolist()
            elif t.startswith("body_xpos:"):
                bname = t.split(":", 1)[1]
                row[t] = d.body(bname).xpos.tolist()
        timeseries.append(row)

        # Keyframe capture
        if step % capture_every_n == 0 and mgr.can_render and slot.renderer:
            update_scene(slot.renderer, d, camera_id=cam_id)
            pixels = slot.renderer.render()
            img = Image.fromarray(pixels)
            buf = BytesIO(); img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            keyframes.append(
                ImageContent(type="image", data=b64, mimeType="image/png"))

    # Build response: data first, then images
    out = [TextContent(type="text", text=json.dumps({
        "n_steps": len(timeseries),
        "sim_time": [timeseries[0]["t"], timeseries[-1]["t"]],
        "timeseries": timeseries,
    }))]
    out.extend(keyframes)
    return out
```

### 9.2 `debug_contacts`

```python
@mcp.tool()
async def debug_contacts(
    ctx: Context,
    n_steps: int = 500,
    ctrl: list[float] | None = None,
    force_threshold: float | None = None,
    penetration_threshold: float = 0.005,
    capture_on_change: bool = True,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Run simulation with event-triggered contact capture.

    Renders snapshot ONLY when contact topology changes or
    force/penetration thresholds are exceeded. Records full
    contact force time-series regardless.

    Returns: event log + force traces + triggered snapshots.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data
    if ctrl is not None:
        d.ctrl[:] = np.array(ctrl, dtype=np.float64)

    cam_id = resolve_camera(m, camera) if camera else -1
    events, traces, keyframes = [], [], []
    prev_pairs = set()

    for step in range(min(n_steps, MAX_STEPS)):
        mujoco.mj_step(m, d)
        cur_pairs = set()
        step_forces = {}
        should_capture = False

        for ci in range(d.ncon):
            c = d.contact[ci]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
            pair = tuple(sorted([g1, g2]))
            cur_pairs.add(pair)
            force = np.zeros(6)
            mujoco.mj_contactForce(m, d, ci, force)
            step_forces[f"{pair[0]}↔{pair[1]}"] = {
                "fn": float(force[0]), "dist": float(c.dist)}

            if force_threshold and abs(force[0]) > force_threshold:
                events.append({"t": d.time, "event": "force_exceeded",
                               "pair": list(pair), "fn": float(force[0])})
                should_capture = True
            if c.dist < -penetration_threshold:
                events.append({"t": d.time, "event": "penetration",
                               "pair": list(pair), "depth": float(-c.dist)})
                should_capture = True

        new = cur_pairs - prev_pairs
        lost = prev_pairs - cur_pairs
        if new or lost:
            for p in new:
                events.append({"t": d.time, "event": "contact_made", "pair": list(p)})
            for p in lost:
                events.append({"t": d.time, "event": "contact_lost", "pair": list(p)})
            if capture_on_change:
                should_capture = True
        prev_pairs = cur_pairs

        traces.append({"t": d.time, "ncon": d.ncon, "forces": step_forces})

        if should_capture and mgr.can_render and slot.renderer:
            opt = mujoco.MjvOption()
            opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
            update_scene(slot.renderer, d, camera_id=cam_id, scene_option=opt)
            pixels = slot.renderer.render()
            img = Image.fromarray(pixels)
            buf = BytesIO(); img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            keyframes.append(ImageContent(type="image", data=b64, mimeType="image/png"))

    out = [TextContent(type="text", text=json.dumps({
        "n_events": len(events), "events": events,
        "contact_trace_length": len(traces),
        "traces": traces,
    }))]
    out.extend(keyframes)
    return out
```

### 9.3 `evaluate_trajectory`

```python
@mcp.tool()
async def evaluate_trajectory(
    ctx: Context,
    trajectory: list[dict] | None = None,
    trajectory_csv: str | None = None,
    n_keyframes: int = 6,
    energy_threshold: float = 1.0,
    penetration_limit: float = 0.01,
    camera: str | None = None,
    sim_name: str | None = None,
) -> list:
    """Replay external trajectory (e.g. from IPOPT/CasADi) and
    check physical plausibility.

    Sets qpos/qvel per frame, calls mj_forward (NOT mj_step),
    then checks: energy conservation, joint limits, penetration,
    velocity smoothness.

    Returns: plausibility score + violation report + keyframes.
    """
    import pandas as pd

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    # Load trajectory
    if trajectory_csv:
        df = pd.read_csv(trajectory_csv)
        frames = df.to_dict("records")
    elif trajectory:
        frames = trajectory
    else:
        return [TextContent(type="text",
            text=json.dumps({"error": "Provide trajectory or trajectory_csv"}))]

    cam_id = resolve_camera(m, camera) if camera else -1
    kf_indices = set(np.linspace(0, len(frames)-1, n_keyframes, dtype=int))
    violations, energy_trace, keyframes = [], [], []
    prev_E = None

    for i, frame in enumerate(frames):
        # Set state without integrating
        qpos_key = [k for k in frame if k.startswith("qpos")]
        qvel_key = [k for k in frame if k.startswith("qvel")]
        if "qpos" in frame:
            d.qpos[:] = np.array(frame["qpos"])
        elif qpos_key:
            # CSV format: qpos_0, qpos_1, ...
            d.qpos[:] = np.array([frame[k] for k in sorted(qpos_key)])
        if "qvel" in frame:
            d.qvel[:] = np.array(frame["qvel"])
        elif qvel_key:
            d.qvel[:] = np.array([frame[k] for k in sorted(qvel_key)])

        mujoco.mj_forward(m, d)
        t = frame.get("time", frame.get("t", i * m.opt.timestep))
        E = float(d.energy[0] + d.energy[1])

        # Energy check
        if prev_E is not None and abs(E - prev_E) > energy_threshold:
            violations.append({"t": t, "type": "energy_jump",
                               "delta": round(abs(E - prev_E), 4)})
        prev_E = E
        energy_trace.append({"t": t, "E": E})

        # Penetration check
        for ci in range(d.ncon):
            if d.contact[ci].dist < -penetration_limit:
                g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                        d.contact[ci].geom1) or "?"
                g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                        d.contact[ci].geom2) or "?"
                violations.append({"t": t, "type": "penetration",
                    "pair": [g1, g2], "depth": round(-d.contact[ci].dist, 5)})

        # Joint limit check
        for j in range(m.njnt):
            if m.jnt_limited[j]:
                q = d.qpos[m.jnt_qposadr[j]]
                lo, hi = m.jnt_range[j]
                if q < lo - 1e-4 or q > hi + 1e-4:
                    jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or str(j)
                    violations.append({"t": t, "type": "joint_limit",
                        "joint": jname, "value": round(float(q), 4),
                        "range": [float(lo), float(hi)]})

        # Keyframe
        if i in kf_indices and mgr.can_render and slot.renderer:
            update_scene(slot.renderer, d, camera_id=cam_id)
            pixels = slot.renderer.render()
            img = Image.fromarray(pixels)
            buf = BytesIO(); img.save(buf, format="PNG", optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            keyframes.append(ImageContent(type="image", data=b64, mimeType="image/png"))

    score = max(0, 100 - len(violations) * 5)
    out = [TextContent(type="text", text=json.dumps({
        "plausibility_score": score,
        "n_frames": len(frames),
        "n_violations": len(violations),
        "violations": violations[:30],
        "energy_trace": energy_trace,
    }))]
    out.extend(keyframes)
    return out
```

### 9.4 `compare_trajectories` and `render_figure_strip`

Omitted here for space — follow the same patterns as above:
- `compare_trajectories`: loads two slots' trajectories, overlays plots, renders side-by-side keyframes, computes RMSE.
- `render_figure_strip`: takes list of timestamps, sets state to nearest recorded frame, renders snapshot per timestamp.

---

## 10. Batch

### `run_sweep`

Key implementation details (beyond v1.1):
- `_apply_param()` uses dot notation: `"geom.box.friction"`, `"option.timestep"`
- Child processes never touch GL. Simulation only.
- Per-task timeout via `future.result(timeout=300)`.
- Returns summary JSON; full results written to CSV.

See v1.2 patch for complete `_apply_param` implementation — adopted verbatim.

---

## 11. Error Handling

All tool functions are wrapped with `@safe_tool`:

```python
import functools, traceback, json, mujoco

def safe_tool(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except mujoco.FatalError as e:
            return json.dumps({"error": "mujoco_fatal", "message": str(e)})
        except FileNotFoundError as e:
            return json.dumps({"error": "file_not_found", "message": str(e)})
        except ValueError as e:
            return json.dumps({"error": "invalid_input", "message": str(e)})
        except RuntimeError as e:
            return json.dumps({"error": "runtime", "message": str(e)})
        except Exception as e:
            return json.dumps({"error": type(e).__name__, "message": str(e),
                               "tb": traceback.format_exc()[-400:]})
    return wrapper
```

---

## 12. Configuration

### pyproject.toml

```toml
[project]
name = "mujoco-mcp-server"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.26", "mujoco>=2.3.0", "numpy>=1.24",
    "Pillow>=10.0", "pandas>=2.0", "matplotlib>=3.7",
    "packaging>=21.0",
]
[project.scripts]
mujoco-mcp = "mujoco_mcp.server:main"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Claude Code config

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

### Environment variables

| Var | Default | Note |
|-----|---------|------|
| `MUJOCO_GL` | auto | `egl`/`osmesa`; auto-probed if unset |
| `MUJOCO_MCP_RENDER_WIDTH` | 640 | |
| `MUJOCO_MCP_RENDER_HEIGHT` | 480 | |
| `MUJOCO_MCP_NO_RENDER` | 0 | Set 1 to skip GL init entirely |
| `MUJOCO_MCP_MAX_WORKERS` | 8 | For run_sweep |

---

## 13. Implementation Plan

### Phase 1 — MVP (Day 1–3)

| File | Contents |
|------|----------|
| `__init__.py`, `__main__.py` | Package entry |
| `compat.py` | Section 3.2 |
| `utils/gl_setup.py` | Section 4 |
| `server.py` | Section 5 |
| `sim_manager.py` | Section 6 |
| `tools/simulation.py` | 8 atomic sim tools (Section 8.1) |
| `tools/rendering.py` | render_snapshot, render_depth (Section 8.2) |
| `tools/meta.py` | server_diagnostics |
| `pyproject.toml` | Section 12 |
| **Test** | Load box XML → step 1000 → verify z < 0 → render |

### Phase 2 — Analysis + Workflow (Day 4–7)

| File | Contents |
|------|----------|
| `tools/analysis.py` | 6 analysis tools (Section 8.3) |
| `tools/model.py` | modify_model, reload_from_xml (Section 8.4) |
| `tools/workflows.py` | run_and_analyze, debug_contacts, evaluate_trajectory (Section 9) |
| `resources.py` | MCP Resources |
| `prompts.py` | MCP Prompts |
| **Test** | Contact debug on two-body model; trajectory evaluation |

### Phase 3 — Batch + Polish (Day 8–10)

| File | Contents |
|------|----------|
| `tools/batch.py` | run_sweep with _apply_param + timeout |
| `tools/export.py` | export_csv, plot_data |
| `tools/workflows.py` | compare_trajectories, render_figure_strip |
| Error wrapping | @safe_tool on all tools |
| **Test** | Friction sweep, PSC Bridges-2 integration |

---

## 14. Audit Checklist

| # | Item | Status |
|---|------|--------|
| 1 | GL env set before `import mujoco` | ✅ subprocess probe in gl_setup.py |
| 2 | Renderer never used as context manager | ✅ explicit .close() in cleanup |
| 3 | Camera always resolved to int ID | ✅ compat.resolve_camera |
| 4 | scene_option kwarg conditional on version | ✅ compat.update_scene |
| 5 | render tools degrade gracefully | ✅ require_renderer raises clear error |
| 6 | mj_forward called after every state change | ✅ sim_set_state, modify_model, sim_load |
| 7 | Child processes (batch) never touch GL | ✅ _run_single_experiment sim-only |
| 8 | Name resolution errors are descriptive | ✅ compat.resolve_name lists available |
| 9 | sim_step has max limit | ✅ 100k cap |
| 10 | Jacobian output auto-truncated | ✅ nv > 50 returns SVD summary only |
| 11 | Model param changes use numpy writes | ✅ modify_model, not XML edit |
| 12 | xml_string stored for batch/reload | ✅ SimSlot.xml_string |
| 13 | Old slot cleaned up on reload | ✅ manager.load closes old renderer |
| 14 | Workflow tools capture events, not fixed intervals | ✅ debug_contacts |
| 15 | Trajectory eval uses mj_forward, not mj_step | ✅ evaluate_trajectory |
| 16 | Batch has per-task timeout | ✅ 300s default |
