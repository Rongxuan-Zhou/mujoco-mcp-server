# Franka on Vention Table Scene Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Open MuJoCo live viewer with Franka Panda mounted at center of short edge of Vention table, facing inward.

**Architecture:** Decimate Vention STL to ~100k faces; use MuJoCo 3.x `MjSpec` + `frame.attach()` to compose scene programmatically (table mesh + Franka robot) and launch viewer.

**Tech Stack:** Python 3.10, MuJoCo 3.2.6, trimesh 4.x, mujoco.viewer

---

## Task 1: Decimate the Table STL

**Files:**
- Create: `scripts/decimate_table.py`
- Output: `models/vention_table_decimated.stl`

**Step 1: Create the decimation script**

```python
#!/usr/bin/env python3
"""Decimate Vention table STL from ~1.2M faces to ~100k faces."""
import trimesh, pathlib

SRC = pathlib.Path("/home/rongxuan_zhou/Downloads/VentionAssembly_123900_v13.stl")
DST = pathlib.Path(__file__).parent.parent / "models" / "vention_table_decimated.stl"
TARGET_FACES = 100_000

mesh = trimesh.load(str(SRC))
print(f"原始面数: {len(mesh.faces):,}")
ratio = TARGET_FACES / len(mesh.faces)
decimated = mesh.simplify_quadratic_decimation(TARGET_FACES)
print(f"精简后面数: {len(decimated.faces):,}")
decimated.export(str(DST))
print(f"已保存: {DST}")
```

**Step 2: Run it**

```bash
mkdir -p /home/rongxuan_zhou/mujoco_mcp/models
python3 /home/rongxuan_zhou/mujoco_mcp/scripts/decimate_table.py
```

Expected output:
```
原始面数: 1,186,572
精简后面数: ~100,000
已保存: .../models/vention_table_decimated.stl
```

---

## Task 2: Create the Launch Script

**Files:**
- Create: `scripts/launch_franka_scene.py`

Coordinate system:
- STL units: mm → scale 0.001 → meters
- Table bounds after scale: X[−0.024, 2.005] Y[−0.116, 0.775] Z[−1.879, −0.652]
- Body pos to center table and set bottom at z=0: `pos=[-0.9905, -0.3295, 1.879]`
- Table top height: 1.227 m
- Robot frame: `pos=[1.015, 0, 1.227]`, quat=180° around Z = `[0,0,0,1]`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Launch MuJoCo viewer: Franka Panda on Vention table."""
import pathlib, numpy as np
import mujoco
import mujoco.viewer

MENAGERIE = pathlib.Path("/home/rongxuan_zhou/mujoco_menagerie/franka_emika_panda")
TABLE_STL = pathlib.Path(__file__).parent.parent / "models" / "vention_table_decimated.stl"

# --- Build scene ---
panda_spec = mujoco.MjSpec.from_file(str(MENAGERIE / "panda.xml"))

scene = mujoco.MjSpec()
scene.option.timestep = 0.002

# Lighting
light = scene.worldbody.add_light()
light.pos = [0, 0, 3]
light.dir = [0, 0, -1]
light.directional = True

# Floor
floor = scene.worldbody.add_geom()
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [5, 5, 0.1]
floor.rgba = [0.6, 0.6, 0.6, 1.0]

# Vention table mesh asset
table_mesh = scene.add_mesh()
table_mesh.name = "vention_table"
table_mesh.file = str(TABLE_STL)
table_mesh.scale = [0.001, 0.001, 0.001]

# Table body: translate so bottom at z=0, centered in XY
# X center of STL bounds: (-0.024+2.005)/2 = 0.9905 m → offset -0.9905
# Y center of STL bounds: (-0.116+0.775)/2 = 0.3295 m → offset -0.3295
# Z: STL z_min = -1.879 m → offset +1.879 to bring to 0
table_body = scene.worldbody.add_body()
table_body.name = "table"
table_body.pos = [-0.9905, -0.3295, 1.879]
table_geom = table_body.add_geom()
table_geom.type = mujoco.mjtGeom.mjGEOM_MESH
table_geom.meshname = "vention_table"
table_geom.rgba = [0.75, 0.78, 0.82, 1.0]
table_geom.contype = 0      # visual only
table_geom.conaffinity = 0

# Table surface collision box (so robot doesn't fall through)
# After transform: table top at z=1.227, spans X[-1.015,1.015] Y[-0.446,0.446]
surface_geom = table_body.add_geom()
surface_geom.type = mujoco.mjtGeom.mjGEOM_BOX
# pos in table_body frame: center of top surface
# table_body frame origin = STL origin (after transform applied externally)
# In STL frame (mm scaled): top at z=-0.652, center XY=(0.9905, 0.3295)
surface_geom.pos = [0.9905, 0.3295, -0.652]
surface_geom.size = [1.015, 0.446, 0.01]
surface_geom.rgba = [0, 0, 0, 0]   # invisible
surface_geom.contype = 1
surface_geom.conaffinity = 1

# Robot frame: center of +X short edge, on table surface, facing -X (inward)
# 180° around Z: quat [w,x,y,z] = [0, 0, 0, 1]
robot_frame = scene.worldbody.add_frame()
robot_frame.name = "robot_mount"
robot_frame.pos = [1.015, 0, 1.227]
robot_frame.quat = [0, 0, 0, 1]    # 180° around Z

# Attach Franka Panda to frame
robot_frame.attach(panda_spec, "panda_", "")

# --- Compile ---
model = scene.compile()
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print(f"模型编译成功: nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
print("打开 MuJoCo 渲染窗口...")

# --- Launch viewer ---
mujoco.viewer.launch(model, data)
```

**Step 2: Verify the script runs**

```bash
python3 /home/rongxuan_zhou/mujoco_mcp/scripts/launch_franka_scene.py
```

Expected: MuJoCo viewer window opens showing Franka on Vention table.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `table_mesh` not found | Check `TABLE_STL` path; re-run decimate script |
| Robot floating in air | Check `frame.pos` Z = table top height (1.227) |
| Robot buried in table | Check coordinate direction: STL Z_max = -0.652 m (in original) → world top = 1.227 m |
| Viewer crashes | Try `MUJOCO_GL=egl python3 launch_franka_scene.py` |
| `attach` prefix error | Prefix must be non-empty; use `"panda_"` |
