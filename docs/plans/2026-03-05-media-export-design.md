# Media Export 工具组 — 设计文档

**日期：** 2026-03-05
**状态：** 已批准

---

## 概述

新增 3 个 MCP 工具，支持视频导出（MP4/GIF）、完整状态日志（CSV）、轨迹可视化（相图/3D 路径）。面向 RL 研究与控制分析两类场景。

- `export_video`：新建 `src/mujoco_mcp/tools/media.py`
- `export_state_log`：扩展 `src/mujoco_mcp/tools/export.py`
- `plot_trajectory`：扩展 `src/mujoco_mcp/tools/export.py`
- 附带：`simulation.py` `sim_step` 录制时追加 `ctrl` 字段（向后兼容）

**工具总计：62 → 65，测试总计：124 → 136**

---

## 工具规格

### 1. `export_video`

从 `slot.trajectory` 逐帧重放状态，渲染后编码为视频。

**参数：**
```
sim_name: str | None
output_path: str              # 扩展名决定格式，或由 fmt 覆盖
fps: int = 30
fmt: str = "mp4"              # "mp4" | "gif"
camera: str | None = None
width: int = 640
height: int = 480
```

**实现：**
- 从 `slot.trajectory` 取所有帧
- 初始化 renderer：`mgr.require_renderer(slot)`
- 遍历每帧：`data.qpos[:] = frame["qpos"]`、`data.qvel[:] = frame["qvel"]`、`mj_forward`、`render()` → `uint8 [H,W,3]`
- GIF：Pillow `Image.save(save_all=True, append_images=[...], duration=1000//fps, loop=0)`（无额外依赖）
- MP4：`imageio.get_writer(output_path, fps=fps)` → `.append_data(frame)` → `.close()`（需 `imageio[ffmpeg]`，运行时检测）
- 每 50 帧 `await asyncio.sleep(0)`（async MCP wrapper 层渲染循环中）
- 使用独立 `render_data = mujoco.MjData(model)` 渲染，从不修改 slot.data
- `finally`：`renderer.close()`（slot.data 未被修改，无需还原）

**可选依赖：**
- GIF：Pillow（已有）
- MP4：`pip install -e ".[media]"` → `imageio>=2.31` + `imageio[ffmpeg]`
- 运行时 `ImportError` → `raise RuntimeError("MP4 requires imageio[ffmpeg]: pip install 'mujoco-mcp[media]'")`

**返回：**
```json
{
  "ok": true,
  "path": "/tmp/rollout.mp4",
  "frames": 300,
  "duration_s": 10.0,
  "format": "mp4"
}
```

---

### 2. `export_state_log`

完整状态 CSV，`include` 参数控制导出哪些列。

**参数：**
```
sim_name: str | None
output_path: str
include: list[str] = ["qpos", "qvel"]
```

**`include` tag 说明：**

| tag | 产生列 | 数据来源 |
|-----|--------|---------|
| `"qpos"` | `qpos_0 … qpos_{nq-1}` | trajectory 直接读取 |
| `"qvel"` | `qvel_0 … qvel_{nv-1}` | trajectory 直接读取 |
| `"ctrl"` | `ctrl_0 … ctrl_{nu-1}` | trajectory（需 sim_step 存储） |
| `"body_xpos:<name>"` | `<name>_x`, `<name>_y`, `<name>_z` | mj_forward 后 `d.xpos[body_id]` |
| `"body_xquat:<name>"` | `<name>_qw`, `<name>_qx`, `<name>_qy`, `<name>_qz` | mj_forward 后 `d.xquat[body_id]` |
| `"sensors"` | `sensor_<name_or_i>` × nsensor | mj_forward 后 `d.sensordata` |
| `"contacts"` | `contact_count`, `max_contact_force` | mj_forward 后遍历 `d.contact[:d.ncon]` |

**实现：**
- 解析 `include`，构建列头列表
- 遍历 `slot.trajectory` 每帧：设 `qpos/qvel`，`mj_forward`（若需要运动学量）
- `csv.DictWriter` 逐行写入
- `ctrl` 若帧中无该字段则写 `""` 并追加 warning
- contacts 注明：`mj_forward` 所得接触为静态几何近似，精确接触用 `debug_contacts`

**返回：**
```json
{
  "ok": true,
  "path": "/tmp/state.csv",
  "rows": 500,
  "columns": ["t", "qpos_0", "torso_x", "torso_y", "torso_z", "contact_count"],
  "warning": "ctrl not found in trajectory frames (run sim_step with recording)"
}
```

---

### 3. `plot_trajectory`

专用轨迹可视化，从 CSV 读取（可由 `export_csv` 或 `export_state_log` 生成）。

**参数：**
```
csv_path: str
plot_type: str              # "phase" | "path3d"
dof: int = 0               # phase 模式：第 N 个自由度
body: str | None = None    # path3d 模式：查找 <body>_x/y/z 列
output_path: str | None = None   # None → inline base64 PNG
title: str = ""
```

**`"phase"` 模式：**
- 读 `qpos_{dof}` vs `qvel_{dof}` 列
- matplotlib 线+散点，起点绿色标记，终点红色标记
- x 轴：`qpos_{dof} [rad/m]`，y 轴：`qvel_{dof} [rad·s⁻¹/m·s⁻¹]`

**`"path3d"` 模式：**
- 读 `<body>_x`, `<body>_y`, `<body>_z` 列（需 `export_state_log` 含 `"body_xpos:<body>"`）
- `mpl_toolkits.mplot3d` 3D 折线，viridis colormap 映射时间
- 标注起点/终点

**返回：** `[ImageContent(base64 PNG), TextContent(摘要)]`，与 `plot_data` 相同格式

---

## `sim_step` 录制扩展

`simulation.py` 中录制路径从：
```python
slot.trajectory.append({"t": data.time, "qpos": qpos.tolist(), "qvel": qvel.tolist()})
```
改为：
```python
slot.trajectory.append({
    "t": data.time,
    "qpos": data.qpos.tolist(),
    "qvel": data.qvel.tolist(),
    "ctrl": data.ctrl.tolist(),
})
```

向后兼容：`export_csv`、`render_figure_strip`、`evaluate_trajectory`、`compare_trajectories` 均只读 `t/qpos/qvel`，新增 `ctrl` 字段不影响任何现有消费者。

---

## 关键实现约定

| 约定 | 细节 |
|------|------|
| 装饰器顺序 | `@mcp.tool()` 外层，`@safe_tool` 内层 |
| 事件循环 | `export_video`：每 50 帧 `await asyncio.sleep(0)` |
| 状态恢复 | `export_video` 使用独立 `render_data`，slot.data 全程未被修改，`finally` 只需 `renderer.close()` |
| 可选依赖 | MP4 运行时检测 `imageio`，`ImportError` → `RuntimeError` 友好提示 |
| GIF 依赖 | Pillow（已有），零新依赖 |
| `_impl` 模式 | `_export_video_impl`、`_export_state_log_impl`、`_plot_trajectory_impl` 供测试直接调用 |
| contacts 精度 | 文档注明：重放接触为 `mj_forward` 静态近似 |

---

## 依赖变化

`pyproject.toml` 新增：
```toml
[project.optional-dependencies]
media = ["imageio>=2.31", "imageio[ffmpeg]"]
```

---

## 测试计划

**新增约 12 个测试（合计 136）：**

`tests/test_media.py`（6 个）：
- `test_export_video_gif_creates_file`：GIF 文件存在且可被 Pillow 打开，帧数正确
- `test_export_video_gif_frame_count`：帧数等于轨迹长度
- `test_export_video_empty_trajectory_raises`：空轨迹 → `ValueError`
- `test_export_video_state_restored_after`：导出后 `qpos/qvel` 恢复原值
- `test_export_video_mp4_requires_imageio`：无 imageio 时 → `RuntimeError` 含 "imageio"
- `test_export_video_invalid_format_raises`：`fmt="avi"` → `ValueError`

扩展 `tests/test_export_tools.py`（6 个）：
- `test_export_state_log_qpos_qvel_ctrl`：ctrl 列存在（录制时保存）
- `test_export_state_log_body_xpos`：`body_xpos:<name>` 产生 `_x/_y/_z` 列
- `test_export_state_log_contacts`：`contacts` 产生 `contact_count`、`max_contact_force`
- `test_export_state_log_sensors`：`sensors` 产生 `sensor_*` 列（若模型有传感器）
- `test_plot_trajectory_phase_returns_image`：返回 ImageContent
- `test_plot_trajectory_path3d_returns_image`：返回 ImageContent（需含 body_xpos 列的 CSV）

---

## 排除项（YAGNI）

- 实时视频流（MJPEG）：需要独立 HTTP server，复杂度高
- 多摄像头同步视频：用户可调用多次 `export_video` 切换 camera 参数
- 二进制序列化格式导出：CSV 已足够，RL 框架通常自行管理格式
- 视频标注（叠加文字/力箭头）：可后处理，超出本次范围
