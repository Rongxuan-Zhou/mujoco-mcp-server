# MuJoCo MCP Server — 可维护性与健壮性改进设计

**日期**: 2026-02-26
**哲学框架**: 局部最优不等于全局最优
**范围**: 结构整理（A）+ 测试补充（B），完整执行

---

## 背景

经过三轮架构审计（2026-02-25~26），52 个 MCP 工具在单文件层面均已修复各类问题（asyncio、viewer sync、sensor 竞态等）。但从系统层面观察，存在 5 类**全局性**问题：

1. **常量分散**：`MAX_STEPS` 在两个文件中独立定义；`1000 步 yield 间隔`散布 4 个文件
2. **接口不一致**：`sim_name=None` 在不同工具中有两套处理逻辑
3. **错误风格混合**：手动 `return json.dumps({"error":...})` 与 `raise` → `@safe_tool` 并存
4. **测试覆盖盲区**：8 个核心原子工具（sim_load/step/forward 等）无任何单元测试
5. **文档格式不统一**：部分工具缺少 `Args/Returns` 段落

这些问题在局部看都"能运行"，但系统层面形成了维护摩擦：修改常量需改 4 处、理解错误行为需学两套约定、调试核心工具无测试保障。

---

## 方案：层次化重构（Layer-by-Layer）

每层解决一类**系统级**问题，层间独立，可逐层审查合并。

```
Layer 1  全局约定层   constants.py + sim_name 语义统一
Layer 2  健壮性层     错误处理风格统一（异常驱动）
Layer 3  验证层       4 个新测试文件（真实 MuJoCo）
Layer 4  文档层       docstring 格式统一
```

---

## Layer 1：全局约定层

### 1.1 新建 `src/mujoco_mcp/constants.py`

集中定义所有跨文件共用的魔数：

```python
# src/mujoco_mcp/constants.py

# 仿真步数上限（simulation.py + workflows.py 各有一份，统一到此）
MAX_SIM_STEPS = 100_000

# asyncio yield 间隔（simulation/control/vision/workflows 4 文件，统一到此）
ASYNC_YIELD_INTERVAL = 1_000

# Jacobian 矩阵显示阈值
JACOBIAN_NV_THRESHOLD = 50

# 传感器数据队列上限
MAX_SENSOR_QUEUE = 1_000

# ProcessPoolExecutor 子进程超时（秒）
BATCH_TASK_TIMEOUT = 300

# 最大 worker 数（默认值，可被 MUJOCO_MCP_MAX_WORKERS 环境变量覆盖）
BATCH_MAX_WORKERS_DEFAULT = 8

# 渲染尺寸（默认值，可被环境变量覆盖）
DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480
```

**影响文件**：`simulation.py`、`workflows.py`、`control.py`、`vision.py`、`analysis.py`、`sensor_feedback.py`、`batch.py`、`meta.py`、`server.py`

每个文件从 `from ..constants import ...` 导入，删除本地定义。

### 1.2 统一 `sim_name=None` 语义

**现状**：`model.py` 中存在手动回退链 `sim_name or mgr.active_slot or "default"`，绕过 `SimManager.get()` 接口。

**修改**：
- 删除 `model.py` 中的旁路回退逻辑
- `model.py` 统一使用 `mgr.get(sim_name)`，与其他工具行为一致
- 在 `SimManager.get()` 的 docstring 中明确文档化 `None` 语义：
  > "name=None 时，返回 active_slot 对应的槽位；无 active_slot 则抛 ValueError"

**哲学注脚**：`model.py` 的旁路在局部看更"防御性"，但系统层面它使 `sim_name=None` 的行为依赖于读者看了哪个文件，是典型的局部优化破坏全局一致性。

---

## Layer 2：健壮性层

### 2.1 统一错误处理约定

**约定**：全项目采用**异常驱动**风格，由 `@safe_tool` 统一捕获并格式化错误响应。

**手动防守式（删除）**：
```python
# simulation.py — 现有（删除）
if not 1 <= n_steps <= MAX_SIM_STEPS:
    return json.dumps({"error": f"n_steps must be 1–{MAX_SIM_STEPS}"})
```

**异常驱动式（保留/统一）**：
```python
# 改为
if not 1 <= n_steps <= MAX_SIM_STEPS:
    raise ValueError(f"n_steps must be 1–{MAX_SIM_STEPS}")
```

**修改范围**：`simulation.py` 中 3 处手动错误返回

**不改**：`export.py` 中 `os.path.exists()` 检查是合理的业务逻辑，不是参数校验，保留。

### 2.2 `SimManager.get()` 文档化

在 `get()` 方法补充完整 docstring，明确 `None` 语义和异常条件，不改逻辑。

---

## Layer 3：验证层（测试补充）

### 测试策略

- 风格：与现有测试完全一致（真实 MuJoCo，使用 `conftest.py` 的 `model_data` fixture）
- 范围：直接测试工具的**私有实现逻辑**（绕过 MCP context 测复杂度）+ 少量工具级别集成测试
- GL 渲染工具：跳过（用 `pytest.mark.skipif` 标注需要 display 的测试）

### 新增测试文件

#### `tests/test_sim_tools.py`

覆盖所有 8 个基础仿真工具：

| 测试用例 | 工具 | 验证内容 |
|--------|------|--------|
| `test_sim_step_advances_time` | sim_step | 步进后 time > 0 |
| `test_sim_step_n_steps` | sim_step | n=100 步，time ≈ 100×dt |
| `test_sim_step_ctrl` | sim_step | ctrl 写入后 data.ctrl 变化 |
| `test_sim_step_invalid_n_steps` | sim_step | n_steps=0 抛 ValueError |
| `test_sim_forward_updates_contacts` | sim_forward | 调用后 ncon 有效 |
| `test_sim_reset` | sim_reset | reset 后 time=0，qpos 回初始 |
| `test_sim_get_state` | sim_get_state | 返回包含 qpos/qvel/ctrl |
| `test_sim_set_state_qpos` | sim_set_state | 设置 qpos 后 get_state 一致 |
| `test_sim_record_trajectory` | sim_record | start→step→stop 后 trajectory 有帧 |
| `test_sim_record_clear` | sim_record | clear 后 trajectory 为空 |

#### `tests/test_analysis_tools.py`

| 测试用例 | 工具 | 验证内容 |
|--------|------|--------|
| `test_analyze_contacts` | analyze_contacts | 掉落后接触数 > 0 |
| `test_compute_jacobian` | compute_jacobian | 返回 rank/singular_values |
| `test_analyze_energy` | analyze_energy | potential + kinetic > 0（初始态） |
| `test_analyze_forces` | analyze_forces | qfrc 向量长度等于 nv |
| `test_read_sensors_all` | read_sensors | 有传感器时返回非空 dict |
| `test_compute_derivatives` | compute_derivatives | A/B 矩阵形状正确 |

#### `tests/test_model_tools.py`

| 测试用例 | 工具 | 验证内容 |
|--------|------|--------|
| `test_modify_geom_friction` | modify_model | 修改后 geom_friction 值改变 |
| `test_modify_body_mass` | modify_model | 修改后 body_mass 值改变 |
| `test_modify_option_timestep` | modify_model | 修改后 opt.timestep 改变 |
| `test_reload_from_xml` | reload_from_xml | 重载后 nq 等参数正确 |
| `test_modify_invalid_element` | modify_model | 无效 element 抛异常 |

#### `tests/test_export_tools.py`

| 测试用例 | 工具 | 验证内容 |
|--------|------|--------|
| `test_export_csv_basic` | export_csv | 写出 CSV 包含 t/qpos 列 |
| `test_export_csv_with_energy` | export_csv | include_energy=True 时有 E_pot/E_kin 列 |
| `test_export_csv_no_trajectory` | export_csv | 无录制轨迹时返回 error |
| `test_plot_data_basic` | plot_data | 返回 [TextContent, ImageContent] |
| `test_plot_data_missing_file` | plot_data | 文件不存在时返回 error |

---

## Layer 4：文档层

### 对象

`simulation.py` 和 `analysis.py` 中缺少 `Args/Returns` 段落的工具，补充至与 `spatial.py` 一致的完整格式。

### 模板

```python
"""单行功能描述。

可选的详细说明（多句话）。

Args:
    param1: 参数说明，包含类型和约束（如 max 100k）。
    param2: 参数说明。
    sim_name: Slot name (default slot if None).

Returns:
    JSON: {"field1": type, "field2": type, ...}
"""
```

---

## 范围控制（不做）

| 项目 | 原因 |
|------|------|
| Pydantic 参数模型 | 引入新依赖，收益有限，过度工程化 |
| mypy --strict | 现有代码无完整类型标注，短期内无法达标 |
| GL 渲染工具测试 | 需要 display，CI 环境难以满足 |
| 多线程并发安全测试 | 超出当前范围，需专门设计 |
| ProcessPoolExecutor 单例 | 边际收益低 |

---

## 成功标准

- `pytest tests/` 全部通过（新增测试 + 现有 52 个）
- `grep -r "1_000\|100_000\|100000" src/` 仅出现在 `constants.py` 导入或注释中
- `grep -r 'return json.dumps.*error' src/mujoco_mcp/tools/simulation.py` 无结果
- 所有工具的 docstring 包含 `Args:` 和 `Returns:` 段落（simulation.py, analysis.py）

---

## 执行顺序

```
1. 新建 constants.py，各文件改为导入
2. 删除 model.py 旁路，更新 SimManager.get() 文档
3. 将 simulation.py 手动 error return 改为 raise
4. 新建 4 个测试文件，运行全套测试
5. 补全 simulation.py 和 analysis.py 的 docstring
6. 最终 pytest + ruff check 验证
```
