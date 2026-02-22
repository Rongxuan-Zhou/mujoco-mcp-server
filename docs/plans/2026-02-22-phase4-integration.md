# Phase 4 集成计划：空间推理 + 参考仓库全功能

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有 27 工具的基础上，集成空间推理（4工具）、Menagerie 加载器（3工具）、高级控制器（4工具）、传感器融合（2工具）、多机器人协调（4工具）、RL 集成（2工具），共增加 19 个 MCP 工具，并新增 3 个 Prompt 模板。

**Architecture:** 参考仓库的核心类（MenagerieLoader、RobotController、SensorFusion、MultiRobotCoordinator、MuJoCoRLEnvironment）直接复制并做最小化适配（去掉 viewer_client 依赖），然后在 `tools/` 目录下为每个模块写薄 MCP 包装层。SimSlot 新增三个可选字段用于保存控制器/传感器/RL 环境状态。

**Tech Stack:** MuJoCo 3.x、FastMCP、numpy、scipy（controllers 依赖）、gymnasium（RL，可选）

**参考仓库路径:** `/tmp/ref_repo/src/mujoco_mcp/`

---

## 关键文件映射

```
当前项目:                              参考仓库来源:
src/mujoco_mcp/
├── menagerie_loader.py       ←  复制自 ref_repo/src/mujoco_mcp/menagerie_loader.py
├── advanced_controllers.py   ←  复制自 ref_repo/.../advanced_controllers.py
├── sensor_feedback.py        ←  复制自 ref_repo/.../sensor_feedback.py
├── multi_robot_coordinator.py ← 复制后去掉 viewer_client 依赖
├── rl_integration.py         ←  复制后去掉 viewer_client 依赖
├── sim_manager.py            ←  新增 3 个可选字段
├── tools/
│   ├── spatial.py            ←  全新（Phase 4a）
│   ├── menagerie.py          ←  全新（Phase 4b）
│   ├── control.py            ←  全新（Phase 4c）
│   ├── sensor_fusion.py      ←  全新（Phase 4d）
│   ├── coordination.py       ←  全新（Phase 4e）
│   └── rl_env.py             ←  全新（Phase 4f）
├── server.py                 ←  新增 6 个 import 行
└── prompts.py                ←  新增 3 个 Prompt
```

---

## Task 1: 复制参考仓库三个无依赖的工具库

**Files:**
- Create: `src/mujoco_mcp/menagerie_loader.py`
- Create: `src/mujoco_mcp/advanced_controllers.py`
- Create: `src/mujoco_mcp/sensor_feedback.py`

**Step 1: 直接复制三个文件**

```bash
cp /tmp/ref_repo/src/mujoco_mcp/menagerie_loader.py \
   /home/rongxuan_zhou/mujoco_mcp/src/mujoco_mcp/

cp /tmp/ref_repo/src/mujoco_mcp/advanced_controllers.py \
   /home/rongxuan_zhou/mujoco_mcp/src/mujoco_mcp/

cp /tmp/ref_repo/src/mujoco_mcp/sensor_feedback.py \
   /home/rongxuan_zhou/mujoco_mcp/src/mujoco_mcp/
```

**Step 2: 验证导入正常**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
uv run python -c "
from src.mujoco_mcp.menagerie_loader import MenagerieLoader
from src.mujoco_mcp.advanced_controllers import PIDController, TrajectoryPlanner, PIDConfig
from src.mujoco_mcp.sensor_feedback import SensorFusion, SensorType
print('OK: all three modules imported successfully')
"
```

Expected: `OK: all three modules imported successfully`

**Step 3: Commit**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
git add src/mujoco_mcp/menagerie_loader.py \
        src/mujoco_mcp/advanced_controllers.py \
        src/mujoco_mcp/sensor_feedback.py
git commit -m "feat: copy menagerie_loader, advanced_controllers, sensor_feedback from ref repo"
```

---

## Task 2: 适配 multi_robot_coordinator.py（去掉 viewer_client 依赖）

**Files:**
- Create: `src/mujoco_mcp/multi_robot_coordinator.py`
- Create: `tests/test_coordinator.py`

**Step 1: 写失败测试**

```python
# tests/test_coordinator.py
import pytest
from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator

def test_coordinator_headless_init():
    """Coordinator must init without viewer_client."""
    coord = MultiRobotCoordinator()
    assert coord.viewer_client is None

def test_add_robot():
    coord = MultiRobotCoordinator()
    ok = coord.add_robot("robot_1", "franka_panda", {"manipulation": True})
    assert ok is True
    assert "robot_1" in coord.robot_states

def test_formation_task():
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    coord.add_robot("r2", "ur5e", {"manipulation": True})
    task_id = coord.formation_control(["r1", "r2"], "line", spacing=1.0)
    assert task_id.startswith("formation_")
```

**Step 2: 运行确认测试失败**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
uv run pytest tests/test_coordinator.py -v 2>&1 | head -20
```

Expected: ImportError 或 ModuleNotFoundError（文件还不存在）

**Step 3: 复制并修改文件**

```bash
cp /tmp/ref_repo/src/mujoco_mcp/multi_robot_coordinator.py \
   /home/rongxuan_zhou/mujoco_mcp/src/mujoco_mcp/
```

修改 `src/mujoco_mcp/multi_robot_coordinator.py`，找到以下两处并修改：

```python
# 删除这行（在文件顶部 imports 中）：
from .viewer_client import MuJoCoViewerClient as ViewerClient

# 将 __init__ 中的：
self.viewer_client = viewer_client or MuJoCoViewerClient()
# 改为：
self.viewer_client = viewer_client  # None = headless mode
```

具体用 Edit 工具操作（不要手动搜索，用精确字符串替换）：

```python
# old_string（精确匹配）:
"from .viewer_client import MuJoCoViewerClient\n"

# new_string:
""
```

```python
# old_string:
"        self.viewer_client = viewer_client or MuJoCoViewerClient()\n"

# new_string:
"        self.viewer_client = viewer_client  # None = headless mode\n"
```

**Step 4: 运行测试确认通过**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
uv run pytest tests/test_coordinator.py -v
```

Expected:
```
PASSED tests/test_coordinator.py::test_coordinator_headless_init
PASSED tests/test_coordinator.py::test_add_robot
PASSED tests/test_coordinator.py::test_formation_task
```

**Step 5: Commit**

```bash
git add src/mujoco_mcp/multi_robot_coordinator.py tests/test_coordinator.py
git commit -m "feat: add multi_robot_coordinator (headless, no viewer_client)"
```

---

## Task 3: 适配 rl_integration.py（去掉 viewer_client 依赖）

**Files:**
- Create: `src/mujoco_mcp/rl_integration.py`
- Create: `tests/test_rl.py`

**Step 1: 写失败测试**

```python
# tests/test_rl.py
import pytest

def test_rl_import_no_viewer():
    """RL module must import without viewer_client."""
    from mujoco_mcp.rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType, ActionSpaceType
    assert MuJoCoRLEnvironment is not None

def test_rl_env_init():
    pytest.importorskip("gymnasium")
    from mujoco_mcp.rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType, ActionSpaceType
    config = RLConfig(
        robot_type="franka_panda",
        task_type=TaskType.REACHING,
        max_episode_steps=100,
        observation_space_size=14,
        action_space_size=7,
    )
    env = MuJoCoRLEnvironment(config)
    assert env.viewer_client is None
    assert env.action_space is not None
```

**Step 2: 运行确认失败**

```bash
uv run pytest tests/test_rl.py -v 2>&1 | head -20
```

**Step 3: 复制并修改文件**

```bash
cp /tmp/ref_repo/src/mujoco_mcp/rl_integration.py \
   /home/rongxuan_zhou/mujoco_mcp/src/mujoco_mcp/
```

在 `src/mujoco_mcp/rl_integration.py` 中做两处修改：

```python
# 删除 viewer_client import（文件第19行）：
# old_string:
"from .viewer_client import MuJoCoViewerClient\n"
# new_string:
""
```

```python
# 修改 MuJoCoRLEnvironment.__init__ 中：
# old_string:
"        self.viewer_client = MuJoCoViewerClient()\n"
# new_string:
"        self.viewer_client = None  # headless mode\n"
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_rl.py -v
```

Expected: `PASSED tests/test_rl.py::test_rl_import_no_viewer` (第二个测试根据 gymnasium 是否安装而定)

**Step 5: Commit**

```bash
git add src/mujoco_mcp/rl_integration.py tests/test_rl.py
git commit -m "feat: add rl_integration (headless, no viewer_client)"
```

---

## Task 4: 扩展 SimSlot 和 SimManager（添加可选字段）

**Files:**
- Modify: `src/mujoco_mcp/sim_manager.py`
- Create: `tests/test_sim_manager_slots.py`

**Step 1: 写失败测试**

```python
# tests/test_sim_manager_slots.py
from mujoco_mcp.sim_manager import SimSlot, SimManager
import mujoco

BOX_XML = """<mujoco><worldbody>
  <body name="box"><geom type="box" size=".1 .1 .1"/></body>
</worldbody></mujoco>"""

def test_simslot_has_optional_fields():
    """SimSlot must have controller, sensor_manager, rl_env fields."""
    sm = SimManager(enable_rendering=False)
    sm.load("t", xml_string=BOX_XML)
    slot = sm.get("t")
    assert hasattr(slot, "controller")
    assert slot.controller is None
    assert hasattr(slot, "sensor_manager")
    assert slot.sensor_manager is None
    assert hasattr(slot, "rl_env")
    assert slot.rl_env is None

def test_simmanager_has_coordinator():
    """SimManager must have a coordinator field."""
    sm = SimManager(enable_rendering=False)
    assert hasattr(sm, "coordinator")
    assert sm.coordinator is None
```

**Step 2: 运行确认失败**

```bash
uv run pytest tests/test_sim_manager_slots.py -v 2>&1 | head -20
```

**Step 3: 修改 sim_manager.py**

在 `src/mujoco_mcp/sim_manager.py` 的 `SimSlot` dataclass 中，找到 `passive_viewer` 行后添加三个字段：

```python
# 在 sim_manager.py 中：
# old_string（精确）:
"    passive_viewer: Optional[object] = None  # mujoco.viewer Handle (launch_passive)\n"
# new_string:
"""    passive_viewer: Optional[object] = None  # mujoco.viewer Handle (launch_passive)
    controller: Optional[object] = None       # Phase 4c: RobotController
    sensor_manager: Optional[object] = None   # Phase 4d: SensorManager
    rl_env: Optional[object] = None           # Phase 4f: MuJoCoRLEnvironment
"""
```

在 `SimManager.__init__` 中，在 `self._lock = threading.Lock()` 行后添加：

```python
# old_string:
"        self._lock = threading.Lock()\n"
# new_string:
"        self._lock = threading.Lock()\n        self.coordinator: Optional[object] = None  # Phase 4e: MultiRobotCoordinator\n"
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_sim_manager_slots.py -v
```

Expected: both tests PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/sim_manager.py tests/test_sim_manager_slots.py
git commit -m "feat: extend SimSlot with controller/sensor_manager/rl_env optional fields"
```

---

## Task 5: Phase 4a — tools/spatial.py（空间推理，4工具）

**Files:**
- Create: `src/mujoco_mcp/tools/spatial.py`
- Create: `tests/test_spatial.py`

**Step 1: 写失败测试**

```python
# tests/test_spatial.py
import pytest
import json
import numpy as np
from mujoco_mcp.sim_manager import SimManager
from mujoco_mcp.tools.spatial import (
    _body_name_to_id, _collect_subtree, _body_aabb_impl, _surface_anchor_impl,
    _SURFACE_AXIS, _SURFACE_SIGN,
)

BOX_XML = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="table" pos="0 0 0.5">
      <geom type="box" size="1.0 0.5 0.4" rgba="0.7 0.5 0.3 1"/>
    </body>
    <body name="ball" pos="0 0 1.5">
      <geom type="sphere" size="0.1"/>
      <body name="ball_child" pos="0 0 0.2">
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>"""

import mujoco as mj

@pytest.fixture
def model_data():
    model = mj.MjModel.from_xml_string(BOX_XML)
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    return model, data

def test_body_name_to_id_found(model_data):
    model, _ = model_data
    bid = _body_name_to_id(model, "table")
    assert bid > 0

def test_body_name_to_id_not_found(model_data):
    model, _ = model_data
    with pytest.raises(ValueError, match="not found"):
        _body_name_to_id(model, "nonexistent")

def test_collect_subtree_leaf(model_data):
    model, _ = model_data
    table_id = _body_name_to_id(model, "table")
    ids = _collect_subtree(model, table_id)
    assert table_id in ids

def test_collect_subtree_with_children(model_data):
    model, _ = model_data
    ball_id = _body_name_to_id(model, "ball")
    child_id = _body_name_to_id(model, "ball_child")
    ids = _collect_subtree(model, ball_id)
    assert ball_id in ids
    assert child_id in ids

def test_body_aabb_box(model_data):
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    # table geom: box size [1.0, 0.5, 0.4], body pos [0, 0, 0.5]
    # Expected: x in [-1, 1], y in [-0.5, 0.5], z in [0.1, 0.9]
    assert lo[2] == pytest.approx(0.1, abs=1e-4)
    assert hi[2] == pytest.approx(0.9, abs=1e-4)

def test_surface_anchor_top_center(model_data):
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "center")
    # Top center = center x,y, max z
    assert pt[2] == pytest.approx(hi[2], abs=1e-6)
    assert pt[0] == pytest.approx((lo[0] + hi[0]) / 2, abs=1e-6)

def test_surface_anchor_top_plus_x(model_data):
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "+x")
    # +x edge on top: x = hi[0], y = center, z = hi[2]
    assert pt[0] == pytest.approx(hi[0], abs=1e-6)
    assert pt[2] == pytest.approx(hi[2], abs=1e-6)

def test_surface_anchor_invalid_surface(model_data):
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", False)
    with pytest.raises(ValueError, match="surface must be one of"):
        _surface_anchor_impl(lo, hi, "invalid", "center")
```

**Step 2: 运行确认失败**

```bash
uv run pytest tests/test_spatial.py -v 2>&1 | head -30
```

Expected: ImportError（spatial.py 还不存在）

**Step 3: 创建 tools/spatial.py**

```python
# src/mujoco_mcp/tools/spatial.py
"""Phase 4a: Spatial reasoning tools for natural-language scene setup.

These tools let Claude query object positions and compute placement coordinates
without manually calculating numbers — bridging natural language to MuJoCo coords.
"""

import json
import numpy as np
import mujoco
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool


# ─── Private helpers ─────────────────────────────────────────────────────────

def _body_name_to_id(model: mujoco.MjModel, name: str) -> int:
    """Resolve body name to ID. Raises ValueError with available names if not found."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        available = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
            for i in range(model.nbody)
        ]
        raise ValueError(f"Body {name!r} not found. Available: {available}")
    return bid


def _collect_subtree(model: mujoco.MjModel, root_id: int) -> list[int]:
    """BFS: collect all body IDs in the subtree rooted at root_id (inclusive)."""
    result = []
    queue = [root_id]
    while queue:
        bid = queue.pop(0)
        result.append(bid)
        for child in range(model.nbody):
            if child != bid and model.body_parentid[child] == bid:
                queue.append(child)
    return result


def _geom_aabb_world(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (lo, hi) AABB of one geom in world coordinates. Returns (None,None) for planes."""
    pos = data.geom_xpos[geom_id].copy()
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    gtype = model.geom_type[geom_id]
    size = model.geom_size[geom_id]

    if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
        return None, None  # Infinite — skip

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        half = size[:3].copy()
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        half = np.full(3, size[0])
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r, h = size[0], size[1]
        half = np.array([r, r, h])
    elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r, h = size[0], size[1]
        half = np.array([r, r, h + r])
    elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        aabb = model.mesh_aabb[mesh_id]          # [cx, cy, cz, hx, hy, hz] in mesh local frame
        center_local = aabb[:3]
        half = aabb[3:6]
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    lp = center_local + np.array([sx * half[0], sy * half[1], sz * half[2]])
                    corners.append(pos + mat @ lp)
        arr = np.array(corners)
        return arr.min(axis=0), arr.max(axis=0)
    else:
        # Ellipsoid, SDF, etc. — use bounding sphere
        half = np.full(3, size[0])

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(pos + mat @ np.array([sx * half[0], sy * half[1], sz * half[2]]))
    arr = np.array(corners)
    return arr.min(axis=0), arr.max(axis=0)


def _body_aabb_impl(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, include_children: bool
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB for a body (optionally including its subtree)."""
    root_id = _body_name_to_id(model, body_name)
    body_ids = _collect_subtree(model, root_id) if include_children else [root_id]

    lo_list, hi_list = [], []
    for bid in body_ids:
        n_geoms = model.body_geomnum[bid]
        start = model.body_geomadr[bid]
        for gid in range(start, start + n_geoms):
            lo, hi = _geom_aabb_world(model, data, gid)
            if lo is not None:
                lo_list.append(lo)
                hi_list.append(hi)

    if not lo_list:
        # Body has no geoms — return zero-size box at body CoM
        xpos = data.xpos[root_id]
        return xpos.copy(), xpos.copy()

    return np.min(lo_list, axis=0), np.max(hi_list, axis=0)


_SURFACE_AXIS = {
    "top": 2, "bottom": 2,
    "+x": 0, "-x": 0,
    "+y": 1, "-y": 1,
}
_SURFACE_SIGN = {
    "top": +1, "bottom": -1,
    "+x": +1, "-x": -1,
    "+y": +1, "-y": -1,
}
_ANCHOR_OFFSETS = {
    "center": {},
    "+x": {0: None},   # None means "use hi[axis]"; filled dynamically
    "-x": {0: None},
    "+y": {1: None},
    "-y": {1: None},
    "+x+y": {0: None, 1: None},
    "+x-y": {0: None, 1: None},
    "-x+y": {0: None, 1: None},
    "-x-y": {0: None, 1: None},
}


def _surface_anchor_impl(
    lo: np.ndarray, hi: np.ndarray, surface: str, anchor: str
) -> np.ndarray:
    """Convert AABB + surface/anchor description to a world-frame point."""
    if surface not in _SURFACE_AXIS:
        raise ValueError(f"surface must be one of {list(_SURFACE_AXIS)}, got {surface!r}")

    anchor_map = {
        "center": {},
        "+x": {0: hi[0]},  "-x": {0: lo[0]},
        "+y": {1: hi[1]},  "-y": {1: lo[1]},
        "+z": {2: hi[2]},  "-z": {2: lo[2]},
        "+x+y": {0: hi[0], 1: hi[1]},
        "+x-y": {0: hi[0], 1: lo[1]},
        "-x+y": {0: lo[0], 1: hi[1]},
        "-x-y": {0: lo[0], 1: lo[1]},
    }
    if anchor not in anchor_map:
        raise ValueError(f"anchor must be one of {list(anchor_map)}, got {anchor!r}")

    center = (lo + hi) / 2.0
    pt = center.copy()

    # Set the primary face axis
    axis = _SURFACE_AXIS[surface]
    pt[axis] = hi[axis] if _SURFACE_SIGN[surface] > 0 else lo[axis]

    # Apply anchor offsets within the face (skip primary axis)
    for dim, val in anchor_map[anchor].items():
        if dim != axis:
            pt[dim] = val

    return pt


# ─── MCP Tools ───────────────────────────────────────────────────────────────

@mcp.tool()
@safe_tool
async def scene_map(ctx: Context, sim_name: str | None = None) -> str:
    """Return a hierarchical map of all bodies in the scene with world positions.

    Use this first before any spatial query to learn body names and scene layout.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"bodies": [{"id": int, "name": str, "parent": str|null, "pos": [x,y,z]}]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data

    bodies = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
        parent_id = model.body_parentid[i]
        parent_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) or f"<body:{parent_id}>"
            if i > 0 else None
        )
        bodies.append({"id": i, "name": name, "parent": parent_name, "pos": data.xpos[i].tolist()})

    return json.dumps({"bodies": bodies}, indent=2)


@mcp.tool()
@safe_tool
async def body_aabb(
    ctx: Context,
    body_name: str,
    sim_name: str | None = None,
    include_children: bool = True,
) -> str:
    """Compute axis-aligned bounding box (AABB) of a body in world coordinates.

    Args:
        body_name: Name of the body (use scene_map to find available names).
        sim_name: Slot name (default slot if None).
        include_children: Include child bodies in the AABB (default True).

    Returns:
        JSON: {"min": [x,y,z], "max": [x,y,z], "center": [x,y,z],
               "size": [sx,sy,sz], "top_z": float}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, body_name, include_children)
    center = ((lo + hi) / 2).tolist()
    size = (hi - lo).tolist()
    return json.dumps({
        "body": body_name,
        "min": lo.tolist(),
        "max": hi.tolist(),
        "center": center,
        "size": size,
        "top_z": float(hi[2]),
    }, indent=2)


@mcp.tool()
@safe_tool
async def surface_anchor(
    ctx: Context,
    body_name: str,
    surface: str,
    anchor: str = "center",
    sim_name: str | None = None,
) -> str:
    """Get world coordinates of a specific point on a body's surface.

    Args:
        body_name: Name of the body.
        surface: Face — "top"|"bottom"|"+x"|"-x"|"+y"|"-y".
        anchor: Point within the face — "center"|"+x"|"-x"|"+y"|"-y"|
                "+x+y"|"+x-y"|"-x+y"|"-x-y".
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"body": str, "surface": str, "anchor": str, "pos": [x, y, z]}

    Example:
        surface_anchor("table", "top", "+x", "center")
        → returns center of the +X edge on the table's top surface
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, body_name, include_children=True)
    pt = _surface_anchor_impl(lo, hi, surface, anchor)
    return json.dumps({
        "body": body_name,
        "surface": surface,
        "anchor": anchor,
        "pos": pt.tolist(),
    }, indent=2)


@mcp.tool()
@safe_tool
async def compute_placement(
    ctx: Context,
    target_body: str,
    surface: str,
    anchor: str = "center",
    object_half_height: float = 0.0,
    sim_name: str | None = None,
) -> str:
    """Compute the world position to place an object on a body's surface.

    Pure spatial computation — does NOT modify the simulation.
    Use the returned placement_pos with modify_model() or sim_set_state().

    Args:
        target_body: Body to place the object on.
        surface: Surface face — "top"|"bottom"|"+x"|"-x"|"+y"|"-y".
        anchor: Anchor point within face — "center"|"+x"|"-x"|"+y"|"-y"|etc.
        object_half_height: Half-height of the object being placed (m).
                            Added in the surface normal direction to avoid interpenetration.
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"placement_pos": [x,y,z], "surface_pos": [x,y,z],
               "target_body": str, "surface": str, "anchor": str}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    lo, hi = _body_aabb_impl(slot.model, slot.data, target_body, include_children=True)
    surface_pt = _surface_anchor_impl(lo, hi, surface, anchor)

    # Offset in the surface normal direction
    axis = _SURFACE_AXIS[surface]
    sign = _SURFACE_SIGN[surface]
    placement = surface_pt.copy()
    placement[axis] += sign * object_half_height

    return json.dumps({
        "target_body": target_body,
        "surface": surface,
        "anchor": anchor,
        "surface_pos": surface_pt.tolist(),
        "placement_pos": placement.tolist(),
    }, indent=2)
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_spatial.py -v
```

Expected: 所有 8 个测试 PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/spatial.py tests/test_spatial.py
git commit -m "feat: add spatial reasoning tools (scene_map, body_aabb, surface_anchor, compute_placement)"
```

---

## Task 6: Phase 4b — tools/menagerie.py（Menagerie 加载，3工具）

**Files:**
- Create: `src/mujoco_mcp/tools/menagerie.py`
- Create: `tests/test_menagerie.py`

**Step 1: 写失败测试**

```python
# tests/test_menagerie.py
import pytest
import json
from mujoco_mcp.menagerie_loader import MenagerieLoader

def test_available_categories():
    loader = MenagerieLoader()
    models = loader.get_available_models()
    assert "arms" in models
    assert "quadrupeds" in models
    assert "franka_emika_panda" in models["arms"]

def test_list_by_category():
    loader = MenagerieLoader()
    arms = loader.get_available_models()["arms"]
    assert len(arms) > 0

def test_loader_cache_dir_created(tmp_path):
    loader = MenagerieLoader(cache_dir=str(tmp_path / "cache"))
    assert (tmp_path / "cache").exists()

# Note: 不测试 download（需要网络，在 CI 中 skip）
@pytest.mark.network
def test_validate_model_franka():
    loader = MenagerieLoader()
    result = loader.validate_model("franka_emika_panda")
    assert result["valid"] is True
    assert result["n_bodies"] > 0
```

**Step 2: 运行本地测试（跳过网络测试）**

```bash
uv run pytest tests/test_menagerie.py -v -m "not network"
```

Expected: 前3个测试 PASSED

**Step 3: 创建 tools/menagerie.py**

```python
# src/mujoco_mcp/tools/menagerie.py
"""Phase 4b: MuJoCo Menagerie model loader MCP tools.

Enables loading 39+ robot models by name without managing XML paths manually.
Models are downloaded from google-deepmind/mujoco_menagerie and cached locally.
"""

import json
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool
from ..menagerie_loader import MenagerieLoader

_loader = MenagerieLoader()  # Singleton with local file cache


@mcp.tool()
@safe_tool
async def list_menagerie_models(
    ctx: Context,
    category: str | None = None,
) -> str:
    """List available MuJoCo Menagerie robot models by category.

    Args:
        category: Filter by category:
                  "arms"|"quadrupeds"|"humanoids"|"grippers"|
                  "mobile_manipulators"|"drones". None = all categories.

    Returns:
        JSON: {"categories": {category: [model_name, ...]}} or {category: [...]}
    """
    models = _loader.get_available_models()
    if category:
        if category not in models:
            raise ValueError(f"Unknown category {category!r}. Available: {list(models)}")
        return json.dumps({category: models[category]}, indent=2)
    return json.dumps({"categories": models}, indent=2)


@mcp.tool()
@safe_tool
async def validate_menagerie_model(ctx: Context, model_name: str) -> str:
    """Download and validate a Menagerie model. Returns body/joint/actuator counts.

    Args:
        model_name: Menagerie model name, e.g. "franka_emika_panda", "unitree_go2".

    Returns:
        JSON: {"valid": bool, "n_bodies": int, "n_joints": int, "n_actuators": int,
               "xml_size": int}
    """
    result = _loader.validate_model(model_name)
    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def load_menagerie_model(
    ctx: Context,
    model_name: str,
    sim_name: str = "default",
    scene_name: str | None = None,
) -> str:
    """Download, resolve XML includes, and load a Menagerie model into a sim slot.

    Models are cached in /tmp/mujoco_menagerie/ after the first download.

    Args:
        model_name: Menagerie model name, e.g. "franka_emika_panda", "unitree_go2".
        sim_name: Target simulation slot (created if not exists).
        scene_name: Optional label for the generated scene XML.

    Returns:
        JSON: sim_load summary (nq, nv, nu, nbody, bodies, joints, actuators, ...)
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    xml = _loader.create_scene_xml(model_name, scene_name)
    result = sm.load(sim_name, xml_string=xml)
    result["menagerie_model"] = model_name
    return json.dumps(result, indent=2)
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_menagerie.py -v -m "not network"
```

Expected: PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/menagerie.py tests/test_menagerie.py
git commit -m "feat: add menagerie MCP tools (list/validate/load_menagerie_model)"
```

---

## Task 7: Phase 4c — tools/control.py（高级控制，4工具）

**Files:**
- Create: `src/mujoco_mcp/tools/control.py`
- Create: `tests/test_control.py`

**Step 1: 写失败测试**

```python
# tests/test_control.py
import pytest
import json
import numpy as np
from mujoco_mcp.advanced_controllers import (
    PIDConfig, PIDController, TrajectoryPlanner, create_arm_controller
)

def test_pid_controller_convergence():
    """PID controller must reduce error over time."""
    config = PIDConfig(kp=10.0, ki=0.1, kd=1.0)
    ctrl = PIDController(config)
    error_history = []
    pos = 0.0
    target = 1.0
    dt = 0.02
    for _ in range(50):
        output = ctrl.update(target, pos, dt)
        pos += output * dt
        error_history.append(abs(target - pos))
    # After 50 steps, error should be less than initial
    assert error_history[-1] < error_history[0]

def test_min_jerk_trajectory_shape():
    """Minimum jerk trajectory must have correct shape."""
    start = np.zeros(7)
    end = np.ones(7)
    positions, velocities, accels = TrajectoryPlanner.minimum_jerk_trajectory(
        start, end, duration=2.0, frequency=100.0
    )
    assert positions.shape == (200, 7)   # 2.0s * 100Hz = 200 waypoints
    assert velocities.shape == (200, 7)

def test_min_jerk_boundary_conditions():
    """Trajectory must start at start_pos and end at end_pos."""
    start = np.array([0.0, 0.5, -1.0])
    end = np.array([1.0, -0.5, 0.5])
    positions, _, _ = TrajectoryPlanner.minimum_jerk_trajectory(start, end, 1.0)
    np.testing.assert_allclose(positions[0], start, atol=1e-6)
    np.testing.assert_allclose(positions[-1], end, atol=1e-4)

def test_create_arm_controller():
    ctrl = create_arm_controller("franka_panda")
    assert ctrl.n_joints == 7
    assert len(ctrl.pid_controllers) == 7
```

**Step 2: 运行确认测试通过（helpers 测试，与 control.py 无关）**

```bash
uv run pytest tests/test_control.py -v
```

Expected: 所有 4 个测试 PASSED（测试的是 advanced_controllers.py 中的类）

**Step 3: 创建 tools/control.py**

```python
# src/mujoco_mcp/tools/control.py
"""Phase 4c: Advanced robot controller MCP tools.

Provides PID + trajectory control accessible via MCP.
Controllers are stored per sim slot and persist across tool calls.
"""

import json
import numpy as np
import mujoco
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool, _viewer_sync
from ..advanced_controllers import (
    TrajectoryPlanner,
    create_arm_controller, create_quadruped_controller, create_humanoid_controller,
)

_FACTORY = {
    "arm": create_arm_controller,
    "quadruped": create_quadruped_controller,
    "humanoid": create_humanoid_controller,
}


def _get_controller(slot):
    ctrl = getattr(slot, "controller", None)
    if ctrl is None:
        raise ValueError(
            "No controller for this slot. Call create_controller() first."
        )
    return ctrl


@mcp.tool()
@safe_tool
async def create_controller(
    ctx: Context,
    robot_type: str = "franka_panda",
    controller_kind: str = "arm",
    sim_name: str | None = None,
) -> str:
    """Create a PID+trajectory controller for a robot in a sim slot.

    Args:
        robot_type: Preset — "franka_panda"|"ur5e"|"anymal_c"|"go2"|"g1"|"h1".
        controller_kind: Category — "arm"|"quadruped"|"humanoid".
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {"created": true, "robot_type": str, "controller_kind": str, "n_joints": int}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    factory = _FACTORY.get(controller_kind, create_arm_controller)
    ctrl = factory(robot_type)
    slot.controller = ctrl
    return json.dumps({
        "created": True,
        "robot_type": robot_type,
        "controller_kind": controller_kind,
        "n_joints": ctrl.n_joints,
    }, indent=2)


@mcp.tool()
@safe_tool
async def plan_trajectory(
    ctx: Context,
    start_qpos: list[float],
    end_qpos: list[float],
    duration: float = 2.0,
    trajectory_type: str = "min_jerk",
    frequency: float = 100.0,
    sim_name: str | None = None,
) -> str:
    """Plan a smooth joint-space trajectory and store it on the slot's controller.

    Args:
        start_qpos: Start joint positions (radians), e.g. [0, -0.785, 0, -2.356, 0, 1.571, 0.785].
        end_qpos: End joint positions (radians).
        duration: Trajectory duration in seconds.
        trajectory_type: "min_jerk" (5th-order polynomial) or "spline" (cubic).
        frequency: Sampling frequency in Hz (default 100).
        sim_name: Slot name. Controller must already be created via create_controller().

    Returns:
        JSON: {"trajectory_type": str, "n_waypoints": int, "duration": float,
               "preview_positions": [[...], [...], [...]]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    ctrl = _get_controller(slot)

    start = np.array(start_qpos)
    end = np.array(end_qpos)

    if trajectory_type == "min_jerk":
        positions, _, _ = TrajectoryPlanner.minimum_jerk_trajectory(
            start, end, duration, frequency=frequency
        )
    else:
        waypoints = np.array([start, end])
        times = np.array([0.0, duration])
        positions, _, _ = TrajectoryPlanner.spline_trajectory(waypoints, times, frequency)

    # Store in controller for step_controller() to consume
    ctrl.set_trajectory(np.array([start, end]), np.array([0.0, duration]))

    return json.dumps({
        "trajectory_type": trajectory_type,
        "n_waypoints": positions.shape[0],
        "duration": duration,
        "frequency": frequency,
        "preview_positions": positions[:3].tolist(),
    }, indent=2)


@mcp.tool()
@safe_tool
async def step_controller(
    ctx: Context,
    n_steps: int = 1,
    sim_name: str | None = None,
) -> str:
    """Execute N physics steps with PID+trajectory-tracking control.

    Each step: get target from trajectory → compute PID → apply ctrl → mj_step.

    Args:
        n_steps: Number of physics steps (each = model.opt.timestep seconds).
        sim_name: Slot name.

    Returns:
        JSON: {"steps_executed": int, "final_qpos": [...], "trajectory_done": bool, "time": float}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data
    ctrl = _get_controller(slot)

    trajectory_done = False
    for _ in range(n_steps):
        target = ctrl.get_trajectory_command()
        if target is None:
            trajectory_done = True
            break
        current_qpos = data.qpos[:ctrl.n_joints]
        commands = ctrl.pid_control(target, current_qpos)
        data.ctrl[:min(len(commands), model.nu)] = commands[:model.nu]
        mujoco.mj_step(model, data)

    if slot.recording:
        slot.trajectory.append({
            "t": float(data.time),
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
        })
    _viewer_sync(slot)

    return json.dumps({
        "steps_executed": n_steps,
        "final_qpos": data.qpos[:ctrl.n_joints].tolist(),
        "trajectory_done": trajectory_done,
        "time": float(data.time),
    }, indent=2)


@mcp.tool()
@safe_tool
async def get_controller_state(ctx: Context, sim_name: str | None = None) -> str:
    """Get controller state: current qpos, target, error, trajectory status.

    Args:
        sim_name: Slot name.

    Returns:
        JSON: {"current_qpos": [...], "time": float, "trajectory_active": bool,
               "target_qpos": [...], "error": [...]}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data
    ctrl = _get_controller(slot)

    target = ctrl.get_trajectory_command()
    current_qpos = data.qpos[:ctrl.n_joints].tolist()

    result = {
        "current_qpos": current_qpos,
        "time": float(data.time),
        "trajectory_active": target is not None,
        "n_joints": ctrl.n_joints,
    }
    if target is not None:
        result["target_qpos"] = target.tolist()
        result["error"] = (np.array(target) - np.array(current_qpos)).tolist()

    return json.dumps(result, indent=2)
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_control.py -v
```

Expected: 所有 4 个测试 PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/control.py tests/test_control.py
git commit -m "feat: add control MCP tools (create_controller, plan_trajectory, step_controller, get_controller_state)"
```

---

## Task 8: Phase 4d — tools/sensor_fusion.py（传感器融合，2工具）

**Files:**
- Create: `src/mujoco_mcp/tools/sensor_fusion.py`
- Create: `tests/test_sensor_fusion.py`

**Step 1: 写失败测试**

```python
# tests/test_sensor_fusion.py
import pytest
import json
import numpy as np
from mujoco_mcp.sensor_feedback import (
    SensorFusion, SensorType, SensorReading, JointSensorProcessor,
    LowPassFilter, create_robot_sensor_suite,
)
import time

def test_low_pass_filter():
    """Low pass filter must smooth noisy signal."""
    flt = LowPassFilter(cutoff_freq=5.0, n_channels=3)
    noisy = np.array([1.0, 0.0, 1.0])
    out1 = flt.update(noisy)
    out2 = flt.update(noisy)
    # Output should not jump to 1.0 immediately — smoothed
    assert out1[0] < 1.0

def test_sensor_fusion_single():
    """Fusion of single sensor just returns its data."""
    fusion = SensorFusion()
    fusion.add_sensor("jp", SensorType.JOINT_POSITION, weight=1.0)
    data = np.array([0.1, 0.2, 0.3])
    reading = SensorReading(
        sensor_id="jp", sensor_type=SensorType.JOINT_POSITION,
        timestamp=time.time(), data=data
    )
    fused = fusion.fuse_sensor_data([reading])
    np.testing.assert_allclose(fused["joint_position"], data)

def test_create_robot_sensor_suite_franka():
    mgr = create_robot_sensor_suite("franka_panda", n_joints=7)
    assert "joint_positions" in mgr.sensors
    assert "joint_velocities" in mgr.sensors
    assert "base_imu" in mgr.sensors
    assert "end_effector_ft" in mgr.sensors  # franka has F/T sensor

def test_joint_sensor_processor():
    proc = JointSensorProcessor("jp", SensorType.JOINT_POSITION, n_joints=3)
    reading = proc.process_raw_data([0.1, 0.2, 0.3])
    assert reading.sensor_type == SensorType.JOINT_POSITION
    assert len(reading.data) == 3
```

**Step 2: 运行测试确认 helpers 通过**

```bash
uv run pytest tests/test_sensor_fusion.py -v
```

Expected: 所有 4 个测试 PASSED（测试的是 sensor_feedback.py）

**Step 3: 创建 tools/sensor_fusion.py**

```python
# src/mujoco_mcp/tools/sensor_fusion.py
"""Phase 4d: Sensor fusion MCP tools.

Wraps sensor_feedback.py for MCP access.
Reads qpos/qvel from MuJoCo, applies per-sensor noise filtering,
then fuses multiple readings via weighted average.
"""

import json
import time
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool
from ..sensor_feedback import SensorFusion, SensorType, SensorReading


def _get_sensor_manager(slot):
    mgr = getattr(slot, "sensor_manager", None)
    if mgr is None:
        raise ValueError(
            "No sensor manager for this slot. Call configure_sensor_fusion() first."
        )
    return mgr


@mcp.tool()
@safe_tool
async def configure_sensor_fusion(
    ctx: Context,
    robot_type: str = "franka_panda",
    n_joints: int | None = None,
    sim_name: str | None = None,
) -> str:
    """Configure multi-sensor fusion pipeline for a robot in a sim slot.

    Creates joint position/velocity sensors, IMU, and F/T sensor (for arms).
    Each sensor includes a low-pass noise filter.

    Args:
        robot_type: Robot type for sensor suite — "franka_panda"|"ur5e"|"anymal_c"|"go2".
        n_joints: Override joint count (auto-detected from model.nv if None).
        sim_name: Slot name.

    Returns:
        JSON: {"sensors_configured": [sensor_id, ...], "robot_type": str, "n_joints": int}
    """
    from ..sensor_feedback import create_robot_sensor_suite
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    nj = n_joints if n_joints is not None else slot.model.nv
    manager = create_robot_sensor_suite(robot_type, nj)
    slot.sensor_manager = manager
    return json.dumps({
        "sensors_configured": list(manager.sensors.keys()),
        "robot_type": robot_type,
        "n_joints": nj,
    }, indent=2)


@mcp.tool()
@safe_tool
async def get_fused_state(ctx: Context, sim_name: str | None = None) -> str:
    """Read and fuse all sensor data for the current simulation state.

    Reads qpos/qvel from MuJoCo, passes through noise filters, and fuses
    all readings via weighted average. Call configure_sensor_fusion() first.

    Args:
        sim_name: Slot name.

    Returns:
        JSON: {"joint_position": [...], "joint_velocity": [...],
               "imu": [...], "timestamp": float}
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    slot = sm.get(sim_name)
    model, data = slot.model, slot.data
    mgr = _get_sensor_manager(slot)

    now = time.time()
    readings = []
    for sensor_id, processor in mgr.sensors.items():
        if "joint_positions" in sensor_id:
            raw = data.qpos[:model.nv].tolist()
        elif "joint_velocities" in sensor_id:
            raw = data.qvel[:model.nv].tolist()
        elif "imu" in sensor_id:
            raw = [0.0, 0.0, 9.81, 0.0, 0.0, 0.0]  # [ax, ay, az, gx, gy, gz]
        elif "end_effector_ft" in sensor_id:
            raw = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # [fx, fy, fz, tx, ty, tz]
        else:
            continue
        readings.append(processor.process_raw_data(raw))

    fusion = SensorFusion()
    for sid, proc in mgr.sensors.items():
        fusion.add_sensor(sid, proc.sensor_type, weight=1.0)

    fused = fusion.fuse_sensor_data(readings)
    result = {"timestamp": now}
    for k, v in fused.items():
        result[k] = v.tolist()
    return json.dumps(result, indent=2)
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_sensor_fusion.py -v
```

Expected: 所有 PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/sensor_fusion.py tests/test_sensor_fusion.py
git commit -m "feat: add sensor_fusion MCP tools (configure_sensor_fusion, get_fused_state)"
```

---

## Task 9: Phase 4e — tools/coordination.py（多机器人协调，4工具）

**Files:**
- Create: `src/mujoco_mcp/tools/coordination.py`
- Modify: `tests/test_coordinator.py` (添加 MCP tools 测试)

**Step 1: 在 test_coordinator.py 末尾添加新测试**

```python
# 在 tests/test_coordinator.py 末尾追加：

def test_get_system_status():
    from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    status = coord.get_system_status()
    assert "num_robots" in status
    assert status["num_robots"] == 1
    assert "r1" in status["robots"]

def test_collision_check_no_collision():
    from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator
    import numpy as np
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    coord.add_robot("r2", "ur5e", {"manipulation": True})
    # Robots are at default positions — no collision expected
    collisions = []
    robot_ids = list(coord.robot_states.keys())
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r1, r2 = robot_ids[i], robot_ids[j]
            if coord.collision_checker.check_collision(
                coord.robot_states[r1], coord.robot_states[r2]
            ):
                collisions.append([r1, r2])
    assert len(collisions) == 0
```

**Step 2: 运行确认新测试通过**

```bash
uv run pytest tests/test_coordinator.py -v
```

Expected: 所有 5 个测试 PASSED

**Step 3: 创建 tools/coordination.py**

```python
# src/mujoco_mcp/tools/coordination.py
"""Phase 4e: Multi-robot coordination MCP tools.

A global MultiRobotCoordinator singleton manages robot registration,
task allocation, and collision detection across all sim slots.
"""

import json
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool
from ..multi_robot_coordinator import MultiRobotCoordinator, TaskType

_coordinator: MultiRobotCoordinator | None = None


def _get_coordinator() -> MultiRobotCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = MultiRobotCoordinator()
    return _coordinator


@mcp.tool()
@safe_tool
async def coordinator_add_robot(
    ctx: Context,
    robot_id: str,
    robot_type: str,
    capabilities: dict | None = None,
) -> str:
    """Register a robot in the multi-robot coordinator.

    Args:
        robot_id: Unique identifier for this robot instance, e.g. "arm_left".
        robot_type: Type — "franka_panda"|"ur5e"|"anymal_c"|"go2".
        capabilities: Dict of capabilities, e.g. {"manipulation": true, "mobility": false}.

    Returns:
        JSON: {"added": bool, "robot_id": str, "robot_type": str}
    """
    coord = _get_coordinator()
    caps = capabilities or {"manipulation": True, "mobility": False}
    ok = coord.add_robot(robot_id, robot_type, caps)
    return json.dumps({"added": ok, "robot_id": robot_id, "robot_type": robot_type}, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_assign_task(
    ctx: Context,
    task_type: str,
    robot_ids: list[str],
    parameters: dict | None = None,
) -> str:
    """Assign a coordinated task to a set of robots.

    Args:
        task_type: "cooperative_manipulation"|"formation_control".
        robot_ids: List of robot IDs (must be registered via coordinator_add_robot).
        parameters:
            For "formation_control": {"formation": "line"|"circle", "spacing": float}
            For "cooperative_manipulation": {"target_object": str,
                                             "<robot_id>_approach": [x, y, z], ...}

    Returns:
        JSON: {"task_id": str, "status": "pending"}
    """
    import numpy as np
    coord = _get_coordinator()
    params = parameters or {}
    type_map = {
        "cooperative_manipulation": TaskType.COOPERATIVE_MANIPULATION,
        "formation_control": TaskType.FORMATION_CONTROL,
    }
    if task_type not in type_map:
        raise ValueError(f"task_type must be one of {list(type_map)}, got {task_type!r}")

    if task_type == "formation_control":
        task_id = coord.formation_control(
            robot_ids,
            params.get("formation", "line"),
            params.get("spacing", 1.0),
        )
    else:
        approaches = {
            rid: np.array(params.get(f"{rid}_approach", [0.0, 0.0, 0.0]))
            for rid in robot_ids
        }
        task_id = coord.cooperative_manipulation(
            robot_ids, params.get("target_object", "object"), approaches
        )

    return json.dumps({"task_id": task_id, "status": "pending"}, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_get_status(ctx: Context) -> str:
    """Get multi-robot coordinator system status.

    Returns:
        JSON: {"running": bool, "num_robots": int, "pending_tasks": int,
               "active_tasks": int, "completed_tasks": int,
               "robots": {robot_id: status_string}}
    """
    coord = _get_coordinator()
    status = coord.get_system_status()
    status["robots"] = {k: v.value for k, v in status["robots"].items()}
    return json.dumps(status, indent=2)


@mcp.tool()
@safe_tool
async def coordinator_check_collisions(ctx: Context) -> str:
    """Run pairwise collision detection across all registered robots.

    Returns:
        JSON: {"collisions": [[robot1_id, robot2_id], ...], "count": int}
    """
    coord = _get_coordinator()
    collisions = []
    robot_ids = list(coord.robot_states.keys())
    for i in range(len(robot_ids)):
        for j in range(i + 1, len(robot_ids)):
            r1, r2 = robot_ids[i], robot_ids[j]
            if coord.collision_checker.check_collision(
                coord.robot_states[r1], coord.robot_states[r2]
            ):
                collisions.append([r1, r2])
    return json.dumps({"collisions": collisions, "count": len(collisions)}, indent=2)
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_coordinator.py -v
```

Expected: 所有测试 PASSED

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/coordination.py tests/test_coordinator.py
git commit -m "feat: add coordination MCP tools (add_robot, assign_task, get_status, check_collisions)"
```

---

## Task 10: Phase 4f — tools/rl_env.py（RL 集成，2工具）

**Files:**
- Create: `src/mujoco_mcp/tools/rl_env.py`
- Modify: `tests/test_rl.py` (追加工具测试)

**Step 1: 在 tests/test_rl.py 末尾追加测试**

```python
# 追加到 tests/test_rl.py：

def test_rl_env_creates_spaces():
    pytest.importorskip("gymnasium")
    from mujoco_mcp.rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType, ActionSpaceType
    config = RLConfig(
        robot_type="franka_panda",
        task_type=TaskType.REACHING,
        max_episode_steps=100,
        observation_space_size=14,
        action_space_size=7,
    )
    env = MuJoCoRLEnvironment(config)
    assert env.observation_space is not None
    assert env.action_space is not None
    assert env.action_space.shape == (7,)
```

**Step 2: 运行确认测试通过**

```bash
uv run pytest tests/test_rl.py -v
```

**Step 3: 创建 tools/rl_env.py**

```python
# src/mujoco_mcp/tools/rl_env.py
"""Phase 4f: Reinforcement Learning environment MCP tools.

Wraps MuJoCoRLEnvironment (Gymnasium-compatible) as MCP tools.
Requires: pip install gymnasium
"""

import json
import numpy as np
from mcp.server.fastmcp import Context

from .._registry import mcp
from .. import safe_tool

try:
    import gymnasium  # noqa: F401
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False

try:
    from ..rl_integration import MuJoCoRLEnvironment, RLConfig, TaskType, ActionSpaceType
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
        robot_type: Robot type for action/observation space — "franka_panda"|"ur5e"|etc.
        task_type: RL task — "reaching"|"balancing"|"walking".
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
```

**Step 4: 运行测试**

```bash
uv run pytest tests/test_rl.py -v
```

Expected: 所有测试 PASSED（网络/gym 测试按条件跳过）

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/rl_env.py tests/test_rl.py
git commit -m "feat: add RL environment MCP tools (create_rl_env, rl_step)"
```

---

## Task 11: 注册所有新工具 + 新增 3 个 Prompt

**Files:**
- Modify: `src/mujoco_mcp/server.py`
- Modify: `src/mujoco_mcp/prompts.py`

**Step 1: 修改 server.py，添加 6 个新工具模块导入**

在 `src/mujoco_mcp/server.py` 中，找到：

```python
from .tools import simulation, rendering, meta  # noqa: E402, F401
from .tools import analysis, model, batch, export, workflows, viewer  # noqa: E402, F401
```

替换为：

```python
from .tools import simulation, rendering, meta  # noqa: E402, F401
from .tools import analysis, model, batch, export, workflows, viewer  # noqa: E402, F401
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env  # noqa: E402, F401
```

**Step 2: 在 prompts.py 末尾追加 3 个新 Prompt**

```python
# 追加到 src/mujoco_mcp/prompts.py 末尾：


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
   → Identify all body names in the scene

2. body_aabb(body_name="<target_body>", sim_name="{scene_slot}")
   → Get bounding box: min/max/center/size/top_z
   → Determine which axis is the "short edge" from the size values

3. surface_anchor(body_name="<target>", surface="top", anchor="+x", sim_name="{scene_slot}")
   → Get exact world coordinates of the placement anchor point
   → Adjust surface/anchor based on user description:
     - "center of table" → surface="top", anchor="center"
     - "short edge center" → determine shorter axis, use "+x" or "+y"
     - "corner" → "+x+y", "+x-y", etc.

4. compute_placement(target_body="<target>", surface="top", anchor="+x",
                     object_half_height=<half_height>, sim_name="{scene_slot}")
   → Returns placement_pos accounting for object size

5. Apply the placement:
   modify_model(modifications=[{{"element":"body","name":"<obj>","field":"pos","value":<placement_pos>}}])

6. render_snapshot() → Verify visually
"""


@mcp.prompt()
async def load_and_control_robot(
    model_name: str = "franka_emika_panda",
    task: str = "move joints to home position",
) -> str:
    """Guided prompt: load a Menagerie robot and set up trajectory control."""
    return f"""Load {model_name} and control it to {task}.

Step 1 — Load model:
  load_menagerie_model(model_name="{model_name}", sim_name="robot")

Step 2 — Check what was loaded:
  sim_get_state(sim_name="robot", include_bodies=True)
  → Note nq (joint count) and current qpos

Step 3 — Create controller:
  create_controller(robot_type="{model_name.replace('-', '_')}", controller_kind="arm", sim_name="robot")

Step 4 — Get current joint positions:
  sim_get_state(sim_name="robot")
  → Use qpos[:n_joints] as start_qpos

Step 5 — Plan trajectory to target:
  plan_trajectory(
      start_qpos=[...],   # current qpos
      end_qpos=[0, -0.785, 0, -2.356, 0, 1.571, 0.785],  # Franka home
      duration=3.0,
      trajectory_type="min_jerk",
      sim_name="robot"
  )

Step 6 — Execute (100 Hz * 3s = 300 steps):
  step_controller(n_steps=300, sim_name="robot")

Step 7 — Verify:
  get_controller_state(sim_name="robot")
  render_snapshot(sim_name="robot")
"""


@mcp.prompt()
async def multi_robot_workflow(
    robot_count: int = 2,
    task: str = "formation_control",
) -> str:
    """Guided prompt: set up and coordinate multiple robots."""
    return f"""Coordinate {robot_count} robots in a {task} task.

Step 1 — Load robots into separate slots:
  load_menagerie_model(model_name="franka_emika_panda", sim_name="robot_1")
  load_menagerie_model(model_name="universal_robots_ur5e", sim_name="robot_2")

Step 2 — Register with coordinator:
  coordinator_add_robot(robot_id="r1", robot_type="franka_panda",
                        capabilities={{"manipulation": true}})
  coordinator_add_robot(robot_id="r2", robot_type="ur5e",
                        capabilities={{"manipulation": true}})

Step 3 — Check status:
  coordinator_get_status()

Step 4 — Assign task:
  # For formation control:
  coordinator_assign_task(
      task_type="formation_control",
      robot_ids=["r1", "r2"],
      parameters={{"formation": "line", "spacing": 1.0}}
  )

Step 5 — Check for collisions:
  coordinator_check_collisions()

Step 6 — Create controllers and move:
  create_controller(robot_type="franka_panda", controller_kind="arm", sim_name="robot_1")
  create_controller(robot_type="ur5e", controller_kind="arm", sim_name="robot_2")
  plan_trajectory(start_qpos=[...], end_qpos=[...], sim_name="robot_1")
  step_controller(n_steps=300, sim_name="robot_1")
"""
```

**Step 3: 验证服务器启动正常**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
MUJOCO_MCP_NO_RENDER=1 timeout 5 uv run python -c "
import sys
sys.argv = ['mujoco_mcp', '--transport', 'stdio']
from mujoco_mcp.server import mcp
print(f'Tools registered: {len(mcp._tool_manager._tools)}')
" 2>&1 | grep -E "Tools|Error|error"
```

Expected: `Tools registered: 46` (27 + 19)

**Step 4: Commit**

```bash
git add src/mujoco_mcp/server.py src/mujoco_mcp/prompts.py
git commit -m "feat: register Phase 4 tools and add 3 new MCP Prompts (place_object, load_and_control, multi_robot)"
```

---

## Task 12: 集成验证测试

**Files:**
- Create: `tests/test_integration_phase4.py`

**Step 1: 创建集成测试**

```python
# tests/test_integration_phase4.py
"""Integration tests: verify Phase 4 tools work together on a real scene."""
import pytest
import json
import mujoco
import numpy as np

from mujoco_mcp.sim_manager import SimManager
from mujoco_mcp.tools.spatial import _body_aabb_impl, _surface_anchor_impl

# Simple scene: table + ball on top
SCENE_XML = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="table" pos="0 0 0.45">
      <geom type="box" size="1.0 0.5 0.45" rgba="0.7 0.5 0.3 1"/>
    </body>
    <body name="ball" pos="0 0 1.05">
      <freejoint/>
      <geom type="sphere" size="0.1" rgba="0.2 0.5 0.9 1"/>
    </body>
  </worldbody>
</mujoco>"""

@pytest.fixture
def md():
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data

def test_table_top_center(md):
    """surface_anchor('table','top','center') must be at table top center."""
    model, data = md
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "center")
    assert pt[2] == pytest.approx(0.90, abs=1e-3)   # 0.45 + 0.45 = 0.90m
    assert pt[0] == pytest.approx(0.0, abs=1e-3)
    assert pt[1] == pytest.approx(0.0, abs=1e-3)

def test_table_plus_x_edge(md):
    """surface_anchor('table','top','+x','center') → center of short +X edge."""
    model, data = md
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "+x")
    assert pt[0] == pytest.approx(1.0, abs=1e-3)    # hi[0] = +1.0m
    assert pt[2] == pytest.approx(0.90, abs=1e-3)   # table top height

def test_compute_placement_above_table(md):
    """compute_placement with ball radius 0.1 → placed 0.1m above table top."""
    model, data = md
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "center")
    # Add half ball height (0.1m radius)
    from mujoco_mcp.tools.spatial import _SURFACE_AXIS, _SURFACE_SIGN
    axis = _SURFACE_AXIS["top"]
    sign = _SURFACE_SIGN["top"]
    placement = pt.copy()
    placement[axis] += sign * 0.1
    assert placement[2] == pytest.approx(1.00, abs=1e-3)  # 0.90 + 0.10

def test_menagerie_loader_models_list():
    """MenagerieLoader must return categories with at least 'arms'."""
    from mujoco_mcp.menagerie_loader import MenagerieLoader
    loader = MenagerieLoader()
    models = loader.get_available_models()
    assert "arms" in models
    assert len(models["arms"]) > 5

def test_pid_controller_tracks_target():
    """PID controller must converge to target within 100 steps."""
    from mujoco_mcp.advanced_controllers import PIDConfig, PIDController
    config = PIDConfig(kp=20.0, ki=0.5, kd=2.0)
    ctrl = PIDController(config)
    pos = 0.0
    target = 1.0
    dt = 0.02
    for _ in range(100):
        output = ctrl.update(target, pos, dt)
        pos += output * dt
    assert abs(pos - target) < 0.1

def test_sensor_suite_creation():
    from mujoco_mcp.sensor_feedback import create_robot_sensor_suite
    mgr = create_robot_sensor_suite("franka_panda", 7)
    assert "joint_positions" in mgr.sensors
    assert "end_effector_ft" in mgr.sensors

def test_coordinator_add_and_status():
    from mujoco_mcp.multi_robot_coordinator import MultiRobotCoordinator
    coord = MultiRobotCoordinator()
    coord.add_robot("r1", "franka_panda", {"manipulation": True})
    coord.add_robot("r2", "ur5e", {"manipulation": True})
    status = coord.get_system_status()
    assert status["num_robots"] == 2
```

**Step 2: 运行集成测试**

```bash
cd /home/rongxuan_zhou/mujoco_mcp
uv run pytest tests/test_integration_phase4.py -v
```

Expected: 所有 7 个测试 PASSED

**Step 3: 运行全部测试确认无回归**

```bash
uv run pytest tests/ -v --ignore=tests/test_integration_phase4.py -x 2>&1 | tail -20
```

**Step 4: 最终 Commit**

```bash
git add tests/test_integration_phase4.py
git commit -m "test: add Phase 4 integration tests (spatial, menagerie, controllers, sensors, coordination)"
```

---

## 完成清单

- [ ] Task 1: 复制 3 个工具库文件
- [ ] Task 2: 适配 multi_robot_coordinator.py
- [ ] Task 3: 适配 rl_integration.py
- [ ] Task 4: 扩展 SimSlot/SimManager
- [ ] Task 5: tools/spatial.py（4工具）
- [ ] Task 6: tools/menagerie.py（3工具）
- [ ] Task 7: tools/control.py（4工具）
- [ ] Task 8: tools/sensor_fusion.py（2工具）
- [ ] Task 9: tools/coordination.py（4工具）
- [ ] Task 10: tools/rl_env.py（2工具）
- [ ] Task 11: 注册工具 + 新 Prompt
- [ ] Task 12: 集成验证测试

**最终工具总数: 27（现有）+ 19（Phase 4）= 46 个 MCP 工具**
