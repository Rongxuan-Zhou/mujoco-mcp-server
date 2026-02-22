# Design: Franka on Vention Table Scene

**Date:** 2026-02-21
**Goal:** Open MuJoCo live viewer with Franka Panda arm mounted at center of short edge of Vention table, facing inward.

## Inputs
- Table STL: `/home/rongxuan_zhou/Downloads/VentionAssembly_123900_v13.stl`
  - 1,186,572 faces, units in mm
  - Bounds: X[−24, 2005] Y[−116, 775] Z[−1879, −652] mm
  - Long axis: X (2029 mm), Short axis: Y (891 mm), Height: 1227 mm
- Franka model: `/home/rongxuan_zhou/mujoco_menagerie/franka_emika_panda/panda.xml`

## Approach: Decimate + Full Mesh (Plan A)
Decimate STL 1.19M → ~100k faces via trimesh, then load as MuJoCo mesh asset.

## File Structure
```
models/
├── vention_table_decimated.stl     # decimated mesh
└── franka_vention_scene.xml        # main scene XML
```
Franka assets referenced via absolute path to menagerie.

## Coordinate Transform (mm → m, floor at z=0)
| Param | Value |
|---|---|
| mesh scale | `0.001 0.001 0.001` |
| body pos X | −0.9905 m (center table in X) |
| body pos Y | −0.3295 m (center table in Y) |
| body pos Z | +1.879 m (bring table bottom to z=0) |

Table top height: **1.227 m**

## Franka Placement
- Position: `(1.015, 0, 1.227)` — center of +X short edge, on table surface
- Rotation: 180° around Z (`euler="0 0 3.14159"`) → arm faces −X (inward)

## Steps
1. Decimate STL: trimesh `simplify_quadratic_decimation` → target 100k faces → `models/vention_table_decimated.stl`
2. Create `models/franka_vention_scene.xml` with table mesh + Franka include
3. Load XML via `mcp__mujoco-sim__sim_load` or Python script
4. Open live viewer via `mcp__mujoco-sim__viewer_open`
