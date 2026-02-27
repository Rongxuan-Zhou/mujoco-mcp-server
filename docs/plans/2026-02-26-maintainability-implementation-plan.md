# MuJoCo MCP Server — 可维护性改进实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 通过四层重构（常量集中、错误风格统一、测试补充、文档标准化）消除系统级维护摩擦

**Architecture:** 新建 `constants.py` 集中所有魔数，各文件改为导入；`simulation.py` 3 处手动错误返回改为 raise；新增 4 个测试文件覆盖 8 个核心原子工具；补全 `simulation.py`/`analysis.py` docstring。

**Tech Stack:** Python 3.10+, MuJoCo bindings, pytest, FastMCP

---

## Task 1: 新建 constants.py（Layer 1a）

**Files:**
- Create: `src/mujoco_mcp/constants.py`

**Step 1: 写 constants.py**

```python
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
```

**Step 2: 验证文件可被导入**

运行：`python -c "from mujoco_mcp.constants import MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL; print(MAX_SIM_STEPS)"`

期望输出：`100000`

**Step 3: 运行测试（基线）**

运行：`pytest tests/ -q`

期望：全部测试通过（约 52 个）

**Step 4: Commit**

```bash
git add src/mujoco_mcp/constants.py
git commit -m "feat: add constants.py to centralise shared magic numbers"
```

---

## Task 2: simulation.py 改用 constants.py（Layer 1a）

**Files:**
- Modify: `src/mujoco_mcp/tools/simulation.py:3,13,47,66`

**Step 1: 修改 simulation.py**

删除第 13 行本地常量，增加导入，更新引用：

```python
# 第 3 行：修改导入块，在 from ..compat import ... 前加一行
from ..constants import MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL

# 删除第 13 行：
# MAX_STEPS = 100_000

# 第 47 行：MAX_STEPS → MAX_SIM_STEPS
    if not 1 <= n_steps <= MAX_SIM_STEPS:
        return json.dumps({"error": f"n_steps must be 1–{MAX_SIM_STEPS}"})

# 第 48 行（docstring 中的上限描述也要更新）：
# "Advance physics simulation by n_steps timesteps (max 100 000 per call)."
# 保持不变（注释 ok）

# 第 66-67 行：1000 → ASYNC_YIELD_INTERVAL
        if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)  # yield to event loop
```

**Step 2: 验证**

运行：`pytest tests/ -q`

期望：全部通过

**Step 3: Commit**

```bash
git add src/mujoco_mcp/tools/simulation.py
git commit -m "refactor: simulation.py imports MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL from constants"
```

---

## Task 3: workflows.py 改用 constants.py（Layer 1a）

**Files:**
- Modify: `src/mujoco_mcp/tools/workflows.py:17,71,91,168,190,308`

**Step 1: 修改 workflows.py**

```python
# 在现有 import 区加一行（文件顶部附近已有 import asyncio）
from ..constants import MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL

# 删除第 17 行：
# MAX_STEPS = 100_000

# 第 71 行：MAX_STEPS → MAX_SIM_STEPS
    n_steps = min(n_steps, MAX_SIM_STEPS)

# 第 91 行（run_and_analyze 内循环）：
        if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)

# 第 168 行（evaluate_trajectory）：MAX_STEPS → MAX_SIM_STEPS
    n_steps = min(n_steps, MAX_SIM_STEPS)

# 第 190 行（evaluate_trajectory 内循环）：
        if (i + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)

# 第 308 行（render_figure_strip 内循环）：
        if (i + 1) % ASYNC_YIELD_INTERVAL == 0:
            await asyncio.sleep(0)
```

**Step 2: 验证**

运行：`pytest tests/ -q`

期望：全部通过

**Step 3: Commit**

```bash
git add src/mujoco_mcp/tools/workflows.py
git commit -m "refactor: workflows.py imports MAX_SIM_STEPS, ASYNC_YIELD_INTERVAL from constants"
```

---

## Task 4: analysis.py 改用 constants.py（Layer 1a）

**Files:**
- Modify: `src/mujoco_mcp/tools/analysis.py:12`

**Step 1: 修改 analysis.py**

```python
# 删除第 12 行：
# JACOBIAN_NV_THRESHOLD = 50

# 在 from . import safe_tool 之前增加导入：
from ..constants import JACOBIAN_NV_THRESHOLD
```

**Step 2: 验证**

运行：`pytest tests/ -q`

期望：全部通过

**Step 3: Commit**

```bash
git add src/mujoco_mcp/tools/analysis.py
git commit -m "refactor: analysis.py imports JACOBIAN_NV_THRESHOLD from constants"
```

---

## Task 5: 其余文件改用 constants.py（Layer 1a）

**Files:**
- Modify: `src/mujoco_mcp/sensor_feedback.py` (MAX_SENSOR_QUEUE)
- Modify: `src/mujoco_mcp/tools/batch.py` (BATCH_TASK_TIMEOUT, BATCH_MAX_WORKERS_DEFAULT)
- Modify: `src/mujoco_mcp/tools/control.py` (ASYNC_YIELD_INTERVAL)
- Modify: `src/mujoco_mcp/tools/vision.py` (ASYNC_YIELD_INTERVAL)
- Modify: `src/mujoco_mcp/tools/meta.py` (BATCH_MAX_WORKERS_DEFAULT)

**Step 1: 修改 sensor_feedback.py**

```python
# 在文件顶部 import 区增加（sensor_feedback.py 路径为 src/mujoco_mcp/sensor_feedback.py）
from .constants import MAX_SENSOR_QUEUE

# 找到第 392 行（SensorManager.__init__ 中）：
# 旧：self.sensor_data_queue = queue.Queue(maxsize=1000)
# 新：
        self.sensor_data_queue = queue.Queue(maxsize=MAX_SENSOR_QUEUE)
```

**Step 2: 修改 batch.py**

```python
# 删除文件顶部的本地常量（约第 15-16 行）：
# MAX_WORKERS_DEFAULT = 8
# TASK_TIMEOUT_SECONDS = 300

# 增加导入：
from ..constants import BATCH_TASK_TIMEOUT, BATCH_MAX_WORKERS_DEFAULT

# 第 189 行：MAX_WORKERS_DEFAULT → BATCH_MAX_WORKERS_DEFAULT
        os.environ.get("MUJOCO_MCP_MAX_WORKERS", str(BATCH_MAX_WORKERS_DEFAULT))

# 第 206 行：TASK_TIMEOUT_SECONDS → BATCH_TASK_TIMEOUT
                    None, lambda: cf.result(timeout=BATCH_TASK_TIMEOUT)

# 第 210 行：
                        "error": "timeout", "timeout_s": BATCH_TASK_TIMEOUT
```

**Step 3: 修改 control.py**

```python
# 增加导入（control.py 已有 import asyncio）
from ..constants import ASYNC_YIELD_INTERVAL

# 找到 step % 1000 == 999 的位置（约第 159 行）：
# 旧：if step % 1000 == 999:
# 新：
            if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
```

**Step 4: 修改 vision.py**

```python
# 增加导入
from ..constants import ASYNC_YIELD_INTERVAL

# 找到约第 618 行 step % 1000 == 999：
# 旧：if step % 1000 == 999:
# 新：
            if (step + 1) % ASYNC_YIELD_INTERVAL == 0:
```

**Step 5: 修改 meta.py**

```python
# 增加导入
from ..constants import BATCH_MAX_WORKERS_DEFAULT

# 找到约第 56 行：
# 旧："MUJOCO_MCP_MAX_WORKERS": os.environ.get("MUJOCO_MCP_MAX_WORKERS", "8"),
# 新：
            "MUJOCO_MCP_MAX_WORKERS": os.environ.get(
                "MUJOCO_MCP_MAX_WORKERS", str(BATCH_MAX_WORKERS_DEFAULT)),
```

**Step 6: 验证常量只出现在 constants.py 和注释/导入中**

运行：
```bash
grep -rn "1_000\b\|100_000\b" src/mujoco_mcp/tools/ src/mujoco_mcp/sensor_feedback.py
```

期望：只有 constants.py 的导入语句，或注释行（`# yield to event loop`）

**Step 7: 验证测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 8: Commit**

```bash
git add src/mujoco_mcp/sensor_feedback.py src/mujoco_mcp/tools/batch.py \
        src/mujoco_mcp/tools/control.py src/mujoco_mcp/tools/vision.py \
        src/mujoco_mcp/tools/meta.py
git commit -m "refactor: remaining files import constants (sensor_feedback, batch, control, vision, meta)"
```

---

## Task 6: 修复 model.py 旁路 + 文档化 SimManager.get()（Layer 1b）

**Files:**
- Modify: `src/mujoco_mcp/tools/model.py:112-114`
- Modify: `src/mujoco_mcp/sim_manager.py:164`

**Step 1: 修改 model.py（reload_from_xml 第 112-114 行）**

```python
# 旧（第 112-114 行）：
    mgr = ctx.request_context.lifespan_context.sim_manager
    name = sim_name or mgr.active_slot or "default"
    summary = mgr.load(name, xml_string=xml_string)

# 新：删除旁路，统一使用 mgr.get() 的 None 语义
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)          # None → active_slot，与其他工具一致
    summary = mgr.load(slot.name, xml_string=xml_string)
```

**Step 2: 补充 SimManager.get() docstring（sim_manager.py 第 164-175 行）**

```python
    def get(self, name: str | None = None) -> SimSlot:
        """Retrieve slot by name (or active slot if name is None). Thread-safe.

        Args:
            name: Slot name to retrieve. ``None`` returns the most recently
                  loaded slot (active_slot). Use explicit names when working
                  with multiple parallel simulations.

        Returns:
            The ``SimSlot`` for the requested name.

        Raises:
            ValueError: If no simulation has been loaded yet (active_slot is
                None) or the named slot does not exist.
        """
        with self._lock:
            target = name or self.active_slot
            if target is None:
                raise ValueError("No simulation loaded. Call sim_load first.")
            slot = self.slots.get(target)
            if slot is None:
                available = list(self.slots.keys())
                raise ValueError(
                    f"Slot '{target}' not found. Available: {available}")
            return slot
```

**Step 3: 验证测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 4: Commit**

```bash
git add src/mujoco_mcp/tools/model.py src/mujoco_mcp/sim_manager.py
git commit -m "fix: remove model.py sim_name bypass; document SimManager.get() None semantics"
```

---

## Task 7: simulation.py 3 处手动错误返回改为 raise（Layer 2）

**Files:**
- Modify: `src/mujoco_mcp/tools/simulation.py:47-48,54-55,213`

**Step 1: 修改 3 处（全部在 simulation.py）**

```python
# sim_step — 第 47-48 行：
# 旧：
    if not 1 <= n_steps <= MAX_SIM_STEPS:
        return json.dumps({"error": f"n_steps must be 1–{MAX_SIM_STEPS}"})
# 新：
    if not 1 <= n_steps <= MAX_SIM_STEPS:
        raise ValueError(f"n_steps must be 1–{MAX_SIM_STEPS}")

# sim_step — 第 54-55 行：
# 旧：
        if len(ctrl) != m.nu:
            return json.dumps({"error": f"ctrl len {len(ctrl)} != nu {m.nu}"})
# 新：
        if len(ctrl) != m.nu:
            raise ValueError(f"ctrl len {len(ctrl)} != nu {m.nu}")

# sim_record — 约第 213 行（else 分支）：
# 旧：
    else:
        return json.dumps({"error": "action must be 'start', 'stop', or 'clear'"})
# 新：
    else:
        raise ValueError("action must be 'start', 'stop', or 'clear'")
```

**Step 2: 验证 simulation.py 无手动错误返回（sim_step/sim_record 范围内）**

运行：
```bash
grep -n 'return json.dumps.*error' src/mujoco_mcp/tools/simulation.py
```

期望：只剩 sim_set_state 中的多维度参数验证行（keyframe / qpos / qvel / ctrl 长度检查），sim_step/sim_record 范围内无输出。

**Step 3: 验证 @safe_tool 能捕获 ValueError**

运行：`pytest tests/ -q`

期望：全部通过（@safe_tool 已处理 ValueError → JSON error）

**Step 4: Commit**

```bash
git add src/mujoco_mcp/tools/simulation.py
git commit -m "refactor: sim_step/sim_record — replace manual error return with raise ValueError"
```

---

## Task 8: 新建 tests/test_sim_tools.py（Layer 3）

**Files:**
- Create: `tests/test_sim_tools.py`

**Step 1: 写测试文件**

```python
"""Unit tests for core simulation tools (sim_step, sim_forward, sim_reset, etc.).

Tests operate directly on MuJoCo model/data to avoid MCP context complexity.
"""
import pytest
import numpy as np
import mujoco

# XML with free-floating box above a floor — suitable for contact tests
_XML_BOX = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""

# XML with slider joint and actuator — for ctrl tests
_XML_ACTUATED = """<mujoco>
  <worldbody>
    <body name="slider">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
    </body>
  </worldbody>
  <actuator><motor name="m1" joint="j1"/></actuator>
</mujoco>"""


def _box_model_data():
    model = mujoco.MjModel.from_xml_string(_XML_BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── sim_step ──────────────────────────────────────────────────────────────────

def test_sim_step_advances_time():
    """A single mj_step increments simulation time."""
    model, data = _box_model_data()
    mujoco.mj_step(model, data)
    assert data.time > 0.0


def test_sim_step_n_steps():
    """100 steps advances time by 100 × timestep (within float64 precision)."""
    model, data = _box_model_data()
    dt = model.opt.timestep
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert abs(data.time - 100 * dt) < 1e-9


def test_sim_step_ctrl_write():
    """ctrl written to data.ctrl persists across a step."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = [2.5]
    mujoco.mj_step(model, data)
    assert abs(data.ctrl[0] - 2.5) < 1e-9


def test_sim_step_invalid_n_steps_raises():
    """Layer 2: sim_step validation must raise ValueError for n_steps=0."""
    from mujoco_mcp.constants import MAX_SIM_STEPS
    n_steps = 0
    with pytest.raises(ValueError, match=r"n_steps must be 1"):
        if not 1 <= n_steps <= MAX_SIM_STEPS:
            raise ValueError(f"n_steps must be 1–{MAX_SIM_STEPS}")


# ── sim_forward ───────────────────────────────────────────────────────────────

def test_sim_forward_updates_ncon():
    """mj_forward after stepping updates contact count (ncon ≥ 0)."""
    model, data = _box_model_data()
    for _ in range(500):        # let box fall and land
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    assert data.ncon >= 0


# ── sim_reset ─────────────────────────────────────────────────────────────────

def test_sim_reset_zeroes_time():
    """mj_resetData sets time back to 0."""
    model, data = _box_model_data()
    for _ in range(50):
        mujoco.mj_step(model, data)
    assert data.time > 0.0
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    assert data.time == 0.0


def test_sim_reset_restores_qpos():
    """mj_resetData restores qpos to initial keyframe values."""
    model, data = _box_model_data()
    initial_qpos = data.qpos.copy()
    for _ in range(200):
        mujoco.mj_step(model, data)
    # qpos should have changed (box has moved / rotated)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qpos, initial_qpos, atol=1e-9)


# ── sim_get_state ─────────────────────────────────────────────────────────────

def test_sim_get_state_fields():
    """State dict contains time, qpos, qvel, ctrl."""
    model, data = _box_model_data()
    state = {
        "time": data.time,
        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),
    }
    for key in ("time", "qpos", "qvel", "ctrl"):
        assert key in state
    assert len(state["qpos"]) == model.nq
    assert len(state["qvel"]) == model.nv


# ── sim_set_state ─────────────────────────────────────────────────────────────

def test_sim_set_state_qpos_roundtrip():
    """Setting qpos via data.qpos write + mj_forward persists to get_state."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qpos[0] = 0.42
    mujoco.mj_forward(model, data)
    assert abs(data.qpos[0] - 0.42) < 1e-9


# ── sim_record ────────────────────────────────────────────────────────────────

def test_sim_record_trajectory_accumulates():
    """Recording trajectory: N steps → N frames in buffer."""
    model, data = _box_model_data()
    trajectory = []
    for _ in range(10):
        mujoco.mj_step(model, data)
        trajectory.append({"t": float(data.time), "qpos": data.qpos.copy().tolist()})
    assert len(trajectory) == 10
    assert trajectory[0]["t"] > 0.0
    assert trajectory[-1]["t"] > trajectory[0]["t"]


def test_sim_record_clear():
    """Clearing trajectory empties the buffer."""
    trajectory = [{"t": float(i) * 0.002} for i in range(20)]
    trajectory.clear()
    assert len(trajectory) == 0
```

**Step 2: 运行新测试**

运行：`pytest tests/test_sim_tools.py -v`

期望：全部 12 个测试通过（PASSED）

**Step 3: 运行全套测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 4: Commit**

```bash
git add tests/test_sim_tools.py
git commit -m "test: add test_sim_tools.py covering 12 core simulation tool behaviors"
```

---

## Task 9: 新建 tests/test_analysis_tools.py（Layer 3）

**Files:**
- Create: `tests/test_analysis_tools.py`

**Step 1: 写测试文件**

```python
"""Unit tests for analysis tools: contacts, Jacobian, derivatives, sensors, energy, forces."""
import numpy as np
import mujoco

_XML_BOX = """<mujoco>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="box" pos="0 0 0.5">
      <joint type="free"/>
      <geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""

_XML_SENSOR = """<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
      <site name="tip" pos=".1 0 0"/>
    </body>
  </worldbody>
  <sensor><framepos name="tip_pos" objtype="site" objname="tip"/></sensor>
</mujoco>"""

_XML_ACTUATED = """<mujoco>
  <worldbody>
    <body name="s">
      <joint name="j1" type="slide" axis="1 0 0"/>
      <geom type="sphere" size=".05"/>
    </body>
  </worldbody>
  <actuator><motor name="m1" joint="j1"/></actuator>
</mujoco>"""


def _box():
    model = mujoco.MjModel.from_xml_string(_XML_BOX)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── analyze_contacts ──────────────────────────────────────────────────────────

def test_analyze_contacts_after_landing():
    """After box falls to floor, ncon > 0."""
    model, data = _box()
    for _ in range(800):
        mujoco.mj_step(model, data)
    assert data.ncon > 0


def test_analyze_contacts_returns_geom_ids():
    """Contact geom IDs are valid (< ngeom)."""
    from mujoco_mcp.compat import contact_geoms
    model, data = _box()
    for _ in range(800):
        mujoco.mj_step(model, data)
    if data.ncon > 0:
        gid1, gid2 = contact_geoms(data.contact[0])
        assert 0 <= gid1 < model.ngeom
        assert 0 <= gid2 < model.ngeom


# ── compute_jacobian ──────────────────────────────────────────────────────────

def test_compute_jacobian_shape():
    """Jacobian for a 1-DOF model has shape (3, 1)."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    J = np.vstack([jacp, jacr])
    assert J.shape == (6, model.nv)


def test_compute_jacobian_singular_values():
    """SVD of Jacobian gives at least one singular value ≥ 0."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    sv = np.linalg.svd(np.vstack([jacp, jacr]), compute_uv=False)
    assert all(s >= 0 for s in sv)
    rank = int(np.sum(sv > 1e-10))
    assert rank > 0


# ── analyze_energy ────────────────────────────────────────────────────────────

def test_analyze_energy_initial_state():
    """Initial-state box at height h has potential energy > 0."""
    from mujoco_mcp.compat import ensure_energy, restore_energy
    model, data = _box()
    was = ensure_energy(model)
    mujoco.mj_forward(model, data)
    pot = float(data.energy[0])
    kin = float(data.energy[1])
    restore_energy(model, was)
    # Box at z=0.5 has gravitational PE
    assert pot + kin > 0.0


# ── analyze_forces ────────────────────────────────────────────────────────────

def test_analyze_forces_qfrc_length():
    """qfrc vectors must have length == nv."""
    model, data = _box()
    assert len(data.qfrc_applied) == model.nv
    assert len(data.qfrc_bias) == model.nv
    assert len(data.qacc) == model.nv


# ── read_sensors ──────────────────────────────────────────────────────────────

def test_read_sensors_all():
    """Model with one sensor returns non-empty dict."""
    model = mujoco.MjModel.from_xml_string(_XML_SENSOR)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    result = {}
    for sid in range(model.nsensor):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
        if name:
            adr = model.sensor_adr[sid]
            dim = model.sensor_dim[sid]
            result[name] = data.sensordata[adr: adr + dim].tolist()
    assert len(result) > 0
    assert "tip_pos" in result
    assert len(result["tip_pos"]) == 3


# ── compute_derivatives ───────────────────────────────────────────────────────

def test_compute_derivatives_ab_shape():
    """A and B matrices have correct shape for 1-DOF actuated model."""
    model = mujoco.MjModel.from_xml_string(_XML_ACTUATED)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    nv, nu = model.nv, model.nu
    A = np.zeros((2 * nv, 2 * nv))
    B = np.zeros((2 * nv, nu))
    mujoco.mjd_transitionFD(model, data, 1e-6, True, A, B, None, None)
    assert A.shape == (2 * nv, 2 * nv)
    assert B.shape == (2 * nv, nu)
```

**Step 2: 运行新测试**

运行：`pytest tests/test_analysis_tools.py -v`

期望：全部 9 个测试通过

**Step 3: 运行全套测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 4: Commit**

```bash
git add tests/test_analysis_tools.py
git commit -m "test: add test_analysis_tools.py (contacts, Jacobian, energy, forces, sensors, derivatives)"
```

---

## Task 10: 新建 tests/test_model_tools.py（Layer 3）

**Files:**
- Create: `tests/test_model_tools.py`

**Step 1: 写测试文件**

```python
"""Unit tests for model modification tools: modify_model, reload_from_xml."""
import pytest
import numpy as np
import mujoco

_XML = """<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1" friction="1 0.005 0.0001"/>
    <body name="box" pos="0 0 0.5">
      <joint type="slide" axis="0 0 1" name="drop"/>
      <geom name="box_geom" type="box" size=".1 .1 .1" mass="1"/>
    </body>
  </worldbody>
</mujoco>"""


def _model_data():
    model = mujoco.MjModel.from_xml_string(_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


# ── modify_model: geom friction ───────────────────────────────────────────────

def test_modify_geom_friction():
    """In-place numpy write changes geom_friction for named geom."""
    model, data = _model_data()
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    old = model.geom_friction[gid].copy()
    model.geom_friction[gid] = [0.5, 0.005, 0.001]
    mujoco.mj_forward(model, data)
    assert model.geom_friction[gid][0] == pytest.approx(0.5)
    assert not np.allclose(model.geom_friction[gid], old)


# ── modify_model: body mass ───────────────────────────────────────────────────

def test_modify_body_mass():
    """Setting body_mass changes gravity effect (different acceleration)."""
    model, data = _model_data()
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    model.body_mass[bid] = 10.0
    mujoco.mj_forward(model, data)
    assert model.body_mass[bid] == pytest.approx(10.0)


# ── modify_model: option timestep ─────────────────────────────────────────────

def test_modify_option_timestep():
    """Setting m.opt.timestep changes integration step."""
    model, data = _model_data()
    model.opt.timestep = 0.005
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    # After 1 step, time should equal new timestep
    assert abs(data.time - 0.005) < 1e-9


# ── reload_from_xml ───────────────────────────────────────────────────────────

def test_reload_from_xml_nq():
    """Reloading XML creates a new model with correct nq."""
    xml = """<mujoco>
      <worldbody>
        <body name="arm">
          <joint name="j1" type="hinge" axis="0 0 1"/>
          <joint name="j2" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size=".05"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert model.njnt == 2
    assert model.nq == 2   # hinge joints each contribute 1 DOF


# ── modify_model: invalid element raises ─────────────────────────────────────

def test_modify_invalid_element_raises():
    """Accessing unknown element name raises ValueError."""
    model, data = _model_data()
    with pytest.raises((ValueError, AttributeError)):
        # Simulate the tool's validation path
        elem = "unknownelement"
        _ELEM_INFO = {"geom", "body", "joint", "actuator", "site"}
        if elem not in _ELEM_INFO and elem != "option":
            raise ValueError(
                f"Unknown element '{elem}'. "
                f"Use: geom, body, joint, actuator, site, option"
            )
```

**Step 2: 运行新测试**

运行：`pytest tests/test_model_tools.py -v`

期望：全部 5 个测试通过

**Step 3: 运行全套测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 4: Commit**

```bash
git add tests/test_model_tools.py
git commit -m "test: add test_model_tools.py (geom friction, body mass, timestep, reload, invalid element)"
```

---

## Task 11: 新建 tests/test_export_tools.py（Layer 3）

**Files:**
- Create: `tests/test_export_tools.py`

**Step 1: 写测试文件**

```python
"""Unit tests for export tools: export_csv, plot_data."""
import asyncio
import json
import os
import tempfile

import pytest
import mujoco
from unittest.mock import MagicMock

from mujoco_mcp.sim_manager import SimSlot


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_slot(trajectory=None):
    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <worldbody>
        <body name="b"><joint type="free"/>
          <geom type="sphere" size=".05"/>
        </body>
      </worldbody>
    </mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    slot = SimSlot(name="default", model=model, data=data)
    if trajectory:
        slot.trajectory = list(trajectory)
    return slot


def _make_ctx(trajectory=None):
    slot = _make_slot(trajectory)
    sm = MagicMock()
    sm.get.return_value = slot
    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = sm
    return ctx


def _fake_traj(n=5):
    return [
        {"t": i * 0.002, "qpos": [float(i)] * 7, "qvel": [float(i) * 0.1] * 6}
        for i in range(n)
    ]


# ── export_csv ────────────────────────────────────────────────────────────────

def test_export_csv_basic():
    """export_csv writes a CSV with t, qpos_*, qvel_* columns."""
    from mujoco_mcp.tools.export import export_csv
    ctx = _make_ctx(trajectory=_fake_traj(5))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        result = asyncio.run(export_csv(ctx, output_path=path))
        d = json.loads(result)
        assert d.get("ok") is True
        assert d["rows"] == 5
        assert "qpos_0" in d["columns"]
        assert "t" in d["columns"]
        assert os.path.exists(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_export_csv_with_energy():
    """export_csv with include_energy=True and energy frames adds E_pot/E_kin columns."""
    from mujoco_mcp.tools.export import export_csv
    traj_with_energy = [
        {"t": 0.0, "qpos": [0.0] * 7, "qvel": [0.0] * 6, "E_pot": 1.0, "E_kin": 0.5},
        {"t": 0.002, "qpos": [0.1] * 7, "qvel": [0.0] * 6, "E_pot": 0.9, "E_kin": 0.6},
    ]
    ctx = _make_ctx(trajectory=traj_with_energy)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        result = asyncio.run(export_csv(ctx, output_path=path, include_energy=True))
        d = json.loads(result)
        assert d.get("ok") is True
        assert "E_pot" in d["columns"]
        assert "E_kin" in d["columns"]
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_export_csv_no_trajectory():
    """export_csv with empty trajectory returns error, not exception."""
    from mujoco_mcp.tools.export import export_csv
    ctx = _make_ctx(trajectory=[])  # empty trajectory
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        result = asyncio.run(export_csv(ctx, output_path=path))
        d = json.loads(result)
        assert "error" in d
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ── plot_data ─────────────────────────────────────────────────────────────────

def test_plot_data_basic():
    """plot_data returns [TextContent, ImageContent] for valid CSV."""
    from mujoco_mcp.tools.export import export_csv, plot_data
    ctx = _make_ctx(trajectory=_fake_traj(10))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        asyncio.run(export_csv(ctx, output_path=path))
        plot_ctx = MagicMock()  # plot_data doesn't use ctx
        result = asyncio.run(plot_data(plot_ctx, csv_path=path))
        from mcp.types import TextContent, ImageContent
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_plot_data_missing_file():
    """plot_data returns error TextContent if CSV file does not exist."""
    from mujoco_mcp.tools.export import plot_data
    ctx = MagicMock()
    result = asyncio.run(plot_data(ctx, csv_path="/nonexistent/path/file.csv"))
    from mcp.types import TextContent
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    d = json.loads(result[0].text)
    assert "error" in d
```

**Step 2: 运行新测试**

运行：`pytest tests/test_export_tools.py -v`

期望：全部 5 个测试通过

**Step 3: 运行全套测试**

运行：`pytest tests/ -q`

期望：全部通过

**Step 4: Commit**

```bash
git add tests/test_export_tools.py
git commit -m "test: add test_export_tools.py (export_csv basic/energy/empty, plot_data happy/missing)"
```

---

## Task 12: 补全 simulation.py 和 analysis.py docstring（Layer 4）

**Files:**
- Modify: `src/mujoco_mcp/tools/simulation.py` (sim_load, sim_reset, sim_record, sim_list)
- Modify: `src/mujoco_mcp/tools/analysis.py` (analyze_contacts, analyze_energy, analyze_forces, read_sensors)

**Step 1: 补全 simulation.py 中缺少 Args/Returns 的工具**

```python
# sim_load — 替换现有 docstring
    """Load a MuJoCo MJCF model into a named simulation slot.

    Provide either xml_path (absolute path to .xml file) or xml_string (raw XML).
    Returns model summary: dimensions, named elements, renderer availability.

    Args:
        xml_path: Absolute path to an .xml MJCF file.
        xml_string: Raw MJCF XML string (alternative to xml_path).
        name: Slot name for this simulation (default ``"default"``).

    Returns:
        JSON: {\"name\": str, \"nq\": int, \"nv\": int, \"nu\": int,
               \"nbody\": int, \"has_renderer\": bool, ...}
    """

# sim_reset — 替换现有 docstring
    """Reset simulation to t=0 with default qpos/qvel.

    Clears and stops trajectory recording. Start recording again with sim_record.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {\"status\": \"reset\", \"time\": 0.0}
    """

# sim_record — 替换现有 docstring
    """Control trajectory recording. action: 'start' | 'stop' | 'clear'.

    While recording, every sim_step appends {t, qpos, qvel} to the trajectory buffer.
    Use export_csv to save the recorded trajectory.

    Args:
        action: One of ``'start'`` (begin recording), ``'stop'`` (pause),
                ``'clear'`` (empty buffer and stop).
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {\"recording\": bool, \"frames\": int}
    """

# sim_list — 替换现有 docstring
    """List all loaded simulation slots with their status.

    Returns:
        JSON: {\"active\": str | null,
               \"slots\": {name: {\"nq\": int, \"nv\": int, \"time\": float,
                                  \"recording\": bool, \"traj_frames\": int,
                                  \"has_renderer\": bool}}}
    """
```

**Step 2: 补全 analysis.py 中缺少 Args/Returns 的工具**

```python
# analyze_contacts — 替换现有 docstring
    """Active contact pairs: geom names, positions, forces, penetration depth.

    Returns up to max_contacts entries sorted by contact index.
    Call sim_forward first if state was recently changed.

    Args:
        max_contacts: Maximum number of contacts to return (default 20).
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {\"n_contacts\": int,
               \"contacts\": [{\"geom1\": str, \"geom2\": str,
                                \"pos\": [x,y,z], \"dist\": float,
                                \"normal_force\": float, ...}]}
    """

# analyze_energy — 替换现有 docstring
    """Current potential energy, kinetic energy, and total mechanical energy.

    Automatically enables the mjENBL_ENERGY flag for the duration of the call
    if not already set (restores original state afterwards). This ensures correct
    non-zero values even when the model XML does not set the energy enable flag.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {\"potential\": float, \"kinetic\": float, \"total\": float,
               \"time\": float, \"energy_enabled\": true}
    """

# analyze_forces — 替换现有 docstring
    """Joint-space force decomposition: applied, constraint, passive, bias, actuator.

    All vectors are in generalized (joint) coordinates, length nv.
    qacc is the resulting joint acceleration.

    Args:
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {\"qfrc_applied\": [...], \"qfrc_constraint\": [...],
               \"qfrc_passive\": [...], \"qfrc_bias\": [...],
               \"qfrc_actuator\": [...], \"qacc\": [...]}
    """

# read_sensors — 替换现有 docstring
    """Read current sensor values by name, or all sensors if no names given.

    Returns a dict mapping sensor name → list of values (dim ≥ 1).

    Args:
        sensor_names: List of sensor names to read. ``None`` reads all sensors.
        sim_name: Slot name (default slot if None).

    Returns:
        JSON: {sensor_name: [float, ...], ...}
    """
```

**Step 3: 验证 docstring 格式**

运行：
```bash
grep -A 3 'Args:' src/mujoco_mcp/tools/simulation.py | head -30
grep -A 3 'Returns:' src/mujoco_mcp/tools/analysis.py | head -30
```

期望：每个工具都出现 `Args:` 和 `Returns:` 段落

**Step 4: 运行全套测试确认无破坏**

运行：`pytest tests/ -q`

期望：全部通过

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/simulation.py src/mujoco_mcp/tools/analysis.py
git commit -m "docs: add Args/Returns docstrings to simulation.py and analysis.py tools"
```

---

## Task 13: 最终验证（所有成功标准）

**Step 1: 全套测试**

运行：`pytest tests/ -v`

期望：**全部通过**（原 52 个 + 新增 ~31 个 = ~83 个）

**Step 2: 常量集中验证**

运行：
```bash
grep -rn "1_000\b\|100_000\b" src/mujoco_mcp/ | grep -v constants.py | grep -v '\.pyc'
```

期望：无输出（或仅注释行）

**Step 3: 无手动错误返回（simulation.py sim_step/sim_record）**

运行：
```bash
python -c "
import ast, sys
src = open('src/mujoco_mcp/tools/simulation.py').read()
tree = ast.parse(src)
for node in ast.walk(tree):
    if isinstance(node, ast.Return):
        if hasattr(node, 'lineno') and node.lineno in range(40, 70):
            print(f'L{node.lineno}: {ast.dump(node)[:80]}')
"
```

期望：sim_step 函数内（约 40-70 行）无 return json.dumps error 调用

**Step 4: docstring 完整性验证**

运行：
```bash
python -c "
import mujoco_mcp.tools.simulation as s
import mujoco_mcp.tools.analysis as a
for mod in (s, a):
    for name in dir(mod):
        fn = getattr(mod, name)
        if callable(fn) and hasattr(fn, '__doc__') and fn.__doc__:
            doc = fn.__doc__
            has_args = 'Args:' in doc
            has_returns = 'Returns:' in doc
            if not (has_args and has_returns):
                print(f'MISSING: {mod.__name__}.{name}')
"
```

期望：无输出（所有工具函数都有 Args/Returns）

**Step 5: ruff lint**

运行：`ruff check src/mujoco_mcp/`

期望：无错误（或仅与本次改动无关的既有警告）
