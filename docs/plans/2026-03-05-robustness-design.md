# Robustness 工具组 — 设计文档

**日期：** 2026-03-05
**状态：** 已批准

---

## 概述

新增 `src/mujoco_mcp/tools/robustness.py`，提供 3 个 MCP 工具，支持扰动鲁棒性分析与域随机化。

---

## 工具规格

### 1. `apply_perturbation`

向指定 body 施加力/力矩脉冲，观察系统恢复行为。

**参数：**
```
sim_name: str | None
body_name: str                       # 施力 body
force: list[float] = [0, 0, 0]      # 世界系 N
torque: list[float] = [0, 0, 0]     # 世界系 N·m
n_steps: int = 50                    # 施力持续步数
recovery_steps: int = 100            # 观察恢复步数
with_controller: bool = False        # 恢复阶段调用 slot.controller
recovery_threshold: float = 0.1     # |qvel| 低于此值视为恢复
```

**实现：**
- 手动保存 `qpos/qvel/act/ctrl/xfrc_applied`（在原始 slot data 上操作）
- 施力期间：`d.xfrc_applied[body_id, :3] = force; d.xfrc_applied[body_id, 3:] = torque`
- `with_controller=True` 时：恢复阶段每步调用 `slot.controller.compute(model, data)`
- 完成后恢复原始状态
- 文档注明：有状态控制器（PID）从扰动时刻起冷启动，不回溯历史

**返回：**
```json
{
  "body": "torso",
  "applied_force": [10, 0, 0],
  "applied_torque": [0, 0, 0],
  "max_qvel_deviation": 3.14,
  "recovery_time_steps": 42,
  "recovered": true,
  "trajectory": [[...], ...]
}
```

---

### 2. `stability_analysis`

在多力幅值 × 多球面方向上重复扰动，量化稳定裕度。

**参数：**
```
sim_name: str | None
body_name: str
force_magnitudes: list[float] = [1, 5, 10, 20]
n_directions: int = 8                # Fibonacci 球面采样方向数
n_steps: int = 50
recovery_steps: int = 200
with_controller: bool = False
recovery_threshold: float = 0.1
```

**实现：**
- 方向生成：Fibonacci 球体采样（`n_directions` 个均匀球面向量），不依赖重力方向假设
- 内部调用 `_apply_perturbation_impl`，复用所有状态保存逻辑
- 在外层试验循环中（每次试验后）`await asyncio.sleep(0)` 让出事件循环

**返回：**
```json
{
  "stability_margin": 10.0,
  "failure_ratio": 0.25,
  "n_trials": 32,
  "results": [
    {"magnitude": 10, "direction": [0.57, 0.57, 0.57], "recovered": true, "recovery_steps": 78},
    ...
  ]
}
```

---

### 3. `randomize_dynamics`

从参数分布中采样 N 组物理参数，运行仿真，输出指标统计和可选 CSV。

**参数：**
```
sim_name: str | None
param_distributions: dict            # 与 run_sweep 相同点标记路径
  # "geom.sphere.mass": {"type": "uniform", "low": 0.5, "high": 2.0}
  # "option.timestep":  {"type": "normal",  "mean": 0.01, "std": 0.001}
  # "geom.floor.friction": {"type": "log_uniform", "low": 0.1, "high": 5.0}
n_samples: int = 20
eval_steps: int = 200
metric: str = "energy"               # "energy" | "max_speed" | "distance"
goal_qpos: list[float] | None = None # metric="distance" 时使用
export_csv: str | None = None        # 可选 CSV 路径
random_seed: int | None = None       # 可复现
```

**分布类型：**
- `uniform`：均匀采样 `[low, high]`
- `normal`：正态采样 `mean ± std`（不裁剪）
- `log_uniform`：对数均匀 `exp(uniform(log(low), log(high)))`

**Metric 定义：**
- `energy`：eval_steps 末的总机械能
- `max_speed`：eval_steps 期间 `|qvel|` 的最大值（越小越稳定）
- `distance`：末态与 `goal_qpos` 的速度参数化距离（`mj_differentiatePos`）

**实现：**
- 原地参数修改 + 恢复：`_get_param(model, path)` 保存，`_set_param(model, path, val)` 修改，完成后恢复
- robustness.py 内联精简版 `_get_param` / `_set_param`（约 30 行，支持 geom/body/joint/actuator/site/option 前缀），不依赖 batch.py 私有函数
- 每个样本运行在原始 slot 的临时 MjData 副本上（不修改 slot.data）
- 每 N 个样本后 `await asyncio.sleep(0)`（via `asyncio.sleep` 在 async 工具层）

**返回：**
```json
{
  "n_samples": 20,
  "metric": "max_speed",
  "mean": 1.23,
  "std": 0.45,
  "min": 0.31,
  "max": 2.87,
  "worst_params": {"geom.sphere.mass": 1.9, "option.timestep": 0.013},
  "best_params":  {"geom.sphere.mass": 0.7, "option.timestep": 0.009},
  "samples": [...],   // 前5最差 + 前5最优
  "csv_path": "/tmp/rand_20260305.csv"
}
```

---

## 关键实现约定

| 约定 | 细节 |
|------|------|
| 状态隔离 | apply_perturbation 在原始 slot data 上操作，手动保存/恢复 |
| 参数隔离 | randomize_dynamics 在原始 model 上原地修改+恢复；临时 MjData 运行仿真 |
| 装饰器顺序 | `@mcp.tool()` 外层，`@safe_tool` 内层 |
| 事件循环 | 所有 3 个 async 工具在调用 `_impl` 前 `await asyncio.sleep(0)` |
| `_impl` 模式 | 每个工具有对应 `_XXX_impl` 同步函数（供测试直接调用） |
| 错误处理 | 所有异常通过 `raise` 传播，`@safe_tool` 统一捕获 |

---

## 测试模型

- `models/box_drop.xml`：测试 apply_perturbation（箱子扰动恢复）
- `tests/test_optimization.py` 中的 SLIDER_XML：测试 randomize_dynamics（质量/摩擦随机化）

---

## 工具数量

| 工具 | 描述 |
|------|------|
| `apply_perturbation` | 施加力矩脉冲，观察恢复 |
| `stability_analysis` | 多方向/幅值扫描，给出稳定裕度 |
| `randomize_dynamics` | 域随机化 N 样本，统计鲁棒性指标 |

**新增 3 个 MCP 工具，总计 62 个工具。**

---

## 排除项（YAGNI）

- `robustness_report`：组合工具计算量过大（n_samples × n_magnitudes × n_directions），移除
- `compare_robustness`：用户可手动对比两个 slot 的 randomize_dynamics 结果
