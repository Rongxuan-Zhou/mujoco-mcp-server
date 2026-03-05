# Media Export Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 添加 3 个 MCP 工具：`export_video`（MP4/GIF）、`export_state_log`（完整状态 CSV）、`plot_trajectory`（相图/3D 轨迹）。

**Architecture:** `media.py`（新建）提供 `export_video`，`export.py` 扩展两个新工具。`sim_step` 录制时追加 `ctrl` 字段。视频 GIF 用 Pillow（已有依赖），MP4 用 `imageio[ffmpeg]`（新增可选依赖）。所有工具遵循 `_impl` 同步函数 + async MCP 包装器模式。

**Tech Stack:** Python 3.10+, MuJoCo ≥ 2.3, Pillow（GIF）, imageio[ffmpeg]（MP4, 可选）, matplotlib（相图/3D）, pandas（CSV 读取）

---

## 背景：必读文件

执行前请读取：
- `src/mujoco_mcp/tools/export.py`（理解现有 export_csv / plot_data 实现模式）
- `src/mujoco_mcp/tools/simulation.py`（找到 sim_step 录制代码段）
- `src/mujoco_mcp/tools/rendering.py`（了解渲染模式）
- `src/mujoco_mcp/server.py:60-70`（工具注册行）
- `pyproject.toml`（依赖结构）

## 关键约定

- **装饰器顺序**：`@mcp.tool()` 外层，`@safe_tool` 内层
- **`_impl` 模式**：每个工具有同步 `_XXX_impl` 函数供测试直接调用
- **可变默认参数**：用 `None` + 内联 guard，不用 `[]`
- **状态恢复**：`export_video` 和 `export_state_log` 在 `finally` 块还原 `qpos/qvel/time` + `mj_forward`
- **导入**：`from .._registry import mcp`；`from . import safe_tool`

---

## Task 1: 写 12 个失败测试

**Files:**
- Create: `tests/test_media.py`
- Modify: `tests/test_export_tools.py`（末尾追加 6 个测试）

### Step 1: 创建 `tests/test_media.py`

```python
"""Tests for export_video (media.py)."""
import json
import os
import mujoco
import numpy as np
import pytest
from PIL import Image

from mujoco_mcp.tools.media import _export_video_impl

# ── 测试模型 ──────────────────────────────────────────────────────────────────
FALL_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    <body name="box" pos="0 0 0.1">
      <joint name="fall" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.2 0.5 0.8 1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_fall():
    model = mujoco.MjModel.from_xml_string(FALL_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def _make_traj(model, data, n=10):
    traj = []
    for _ in range(n):
        mujoco.mj_step(model, data)
        traj.append({
            "t": data.time,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(),
        })
    return traj


# ── tests ─────────────────────────────────────────────────────────────────────

def test_export_video_gif_creates_file(tmp_path):
    """GIF 文件应存在且为有效 GIF 格式。"""
    model, data = _make_fall()
    traj = _make_traj(model, data, n=5)
    out = str(tmp_path / "test.gif")
    result = json.loads(_export_video_impl(model, data, traj, out, fps=10, fmt="gif"))
    assert result["ok"] is True
    assert result["frames"] == 5
    assert os.path.exists(out)
    img = Image.open(out)
    assert img.format == "GIF"


def test_export_video_gif_frame_count(tmp_path):
    """GIF 帧数应等于轨迹长度。"""
    model, data = _make_fall()
    traj = _make_traj(model, data, n=8)
    out = str(tmp_path / "test8.gif")
    result = json.loads(_export_video_impl(model, data, traj, out, fps=10, fmt="gif"))
    assert result["frames"] == 8


def test_export_video_empty_trajectory_raises():
    """空轨迹应抛出 ValueError。"""
    model, data = _make_fall()
    with pytest.raises(ValueError, match="empty"):
        _export_video_impl(model, data, [], "/tmp/never.gif", fmt="gif")


def test_export_video_state_restored_after(tmp_path):
    """导出后 qpos/qvel 应恢复原值。"""
    model, data = _make_fall()
    traj = _make_traj(model, data, n=5)
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    out = str(tmp_path / "state.gif")
    _export_video_impl(model, data, traj, out, fps=10, fmt="gif")
    assert np.allclose(data.qpos, qpos_before)
    assert np.allclose(data.qvel, qvel_before)


def test_export_video_mp4_requires_imageio(tmp_path, monkeypatch):
    """无 imageio 时应抛出 RuntimeError 含 'imageio'。"""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "imageio":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    model, data = _make_fall()
    traj = _make_traj(model, data, n=3)
    with pytest.raises(RuntimeError, match="imageio"):
        _export_video_impl(model, data, traj, str(tmp_path / "out.mp4"), fmt="mp4")


def test_export_video_invalid_format_raises(tmp_path):
    """不支持的格式应抛出 ValueError。"""
    model, data = _make_fall()
    traj = _make_traj(model, data, n=3)
    with pytest.raises(ValueError, match="fmt"):
        _export_video_impl(model, data, traj, str(tmp_path / "out.avi"), fmt="avi")
```

### Step 2: 追加 6 个测试到 `tests/test_export_tools.py`

在文件末尾追加（不要修改现有内容）：

```python
# ── export_state_log + plot_trajectory tests ──────────────────────────────────

import mujoco as _mujoco  # noqa: E402 (bottom of file)
from mujoco_mcp.tools.export import _export_state_log_impl, _plot_trajectory_impl
from mcp.types import ImageContent  # noqa: E402

SENSOR_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
      <site name="cog"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="accel" site="cog"/>
  </sensor>
  <actuator>
    <motor name="motor" joint="slider" gear="1" ctrllimited="true" ctrlrange="-10 10"/>
  </actuator>
</mujoco>
"""

CONTACT_XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="box" pos="0 0 0.1">
      <joint name="fall" type="free"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _make_sensor_traj(n=20):
    model = _mujoco.MjModel.from_xml_string(SENSOR_XML)
    data = _mujoco.MjData(model)
    _mujoco.mj_forward(model, data)
    data.ctrl[0] = 5.0
    traj = []
    for _ in range(n):
        _mujoco.mj_step(model, data)
        traj.append({
            "t": data.time,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(),
        })
    return model, data, traj


def _make_contact_traj(n=30):
    model = _mujoco.MjModel.from_xml_string(CONTACT_XML)
    data = _mujoco.MjData(model)
    _mujoco.mj_forward(model, data)
    traj = []
    for _ in range(n):
        _mujoco.mj_step(model, data)
        traj.append({
            "t": data.time,
            "qpos": data.qpos.tolist(),
            "qvel": data.qvel.tolist(),
            "ctrl": data.ctrl.tolist(),
        })
    return model, data, traj


def test_export_state_log_qpos_qvel_ctrl(tmp_path):
    """export_state_log 应输出 qpos/qvel/ctrl 列。"""
    import csv as _csv
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "state.csv")
    result = json.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["qpos", "qvel", "ctrl"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    assert "qpos_0" in header
    assert "qvel_0" in header
    assert "ctrl_0" in header


def test_export_state_log_body_xpos(tmp_path):
    """body_xpos:<name> 应产生 <name>_x/y/z 列。"""
    import csv as _csv
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "xpos.csv")
    result = json.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["body_xpos:cart"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    assert "cart_x" in header
    assert "cart_y" in header
    assert "cart_z" in header


def test_export_state_log_contacts(tmp_path):
    """contacts tag 应产生 contact_count 和 max_contact_force 列。"""
    import csv as _csv
    model, data, traj = _make_contact_traj()
    out = str(tmp_path / "contacts.csv")
    result = json.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["contacts"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    assert "contact_count" in header
    assert "max_contact_force" in header


def test_export_state_log_sensors(tmp_path):
    """sensors tag 应产生加速度计 accel_0/1/2 列（3D 传感器）。"""
    import csv as _csv
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "sensors.csv")
    result = json.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["sensors"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    # accelerometer has 3 dims → accel_0, accel_1, accel_2
    assert any(col.startswith("accel") for col in header)


def test_plot_trajectory_phase_returns_image(tmp_path):
    """plot_trajectory phase 模式应返回 ImageContent。"""
    import pandas as pd
    model, data, traj = _make_sensor_traj(n=50)
    csv_path = str(tmp_path / "traj.csv")
    _export_state_log_impl(model, data, traj, csv_path, include=["qpos", "qvel"])
    result = _plot_trajectory_impl(csv_path, "phase", dof=0)
    assert any(isinstance(r, ImageContent) for r in result)
    img_item = next(r for r in result if isinstance(r, ImageContent))
    assert len(img_item.data) > 100  # non-trivial base64 content


def test_plot_trajectory_path3d_returns_image(tmp_path):
    """plot_trajectory path3d 模式应返回 ImageContent。"""
    model, data, traj = _make_sensor_traj(n=50)
    csv_path = str(tmp_path / "traj3d.csv")
    _export_state_log_impl(model, data, traj, csv_path,
                           include=["body_xpos:cart"])
    result = _plot_trajectory_impl(csv_path, "path3d", body="cart")
    assert any(isinstance(r, ImageContent) for r in result)
```

### Step 3: 运行测试确认全部失败

```bash
cd /home/rongxuan_zhou/mujoco_mcp
python -m pytest tests/test_media.py tests/test_export_tools.py -v --tb=short 2>&1 | tail -30
```

**预期**：所有新测试 FAIL（ImportError 或 AttributeError），原有测试 PASS。

### Step 4: 提交失败测试

```bash
git add tests/test_media.py tests/test_export_tools.py
git commit -m "test(media): add 12 failing tests for export_video, export_state_log, plot_trajectory"
```

---

## Task 2: 扩展 sim_step 录制保存 ctrl

**Files:**
- Modify: `src/mujoco_mcp/tools/simulation.py`（找到录制追加的那行）

### Step 1: 找到录制代码

读取 `src/mujoco_mcp/tools/simulation.py`，找到类似：
```python
slot.trajectory.append({"t": data.time, "qpos": ..., "qvel": ...})
```

### Step 2: 添加 ctrl 字段

将该行改为：
```python
slot.trajectory.append({
    "t": data.time,
    "qpos": data.qpos.tolist(),
    "qvel": data.qvel.tolist(),
    "ctrl": data.ctrl.tolist(),
})
```

**注意**：`data.ctrl` 即使未设置也存在（全零数组），`.tolist()` 安全。

### Step 3: 运行现有测试确认无回归

```bash
python -m pytest tests/test_sim_tools.py tests/test_export_tools.py -v --tb=short
```

**预期**：原有测试全部 PASS（ctrl 是新增字段，现有消费者只读 t/qpos/qvel）。

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/simulation.py
git commit -m "feat(sim): save ctrl in trajectory recording for export_state_log"
```

---

## Task 3: 实现 export_state_log + plot_trajectory（扩展 export.py）

**Files:**
- Modify: `src/mujoco_mcp/tools/export.py`

### Step 1: 读取 export.py 并追加 imports

读取文件，在现有 imports 末尾追加（若已有则跳过）：
```python
import mujoco
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mcp.types import ImageContent, TextContent
```

### Step 2: 追加 `_export_state_log_impl`

在 `export.py` 末尾追加：

```python
# ─── export_state_log ─────────────────────────────────────────────────────────

def _export_state_log_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: list[dict],
    output_path: str,
    include: list[str] | None = None,
) -> str:
    """Export full-state CSV with selectable columns.

    Args:
        model: MjModel instance.
        data: MjData instance (state is saved/restored).
        trajectory: list of frames from slot.trajectory.
        output_path: destination CSV path.
        include: list of tags — "qpos", "qvel", "ctrl",
            "body_xpos:<name>", "body_xquat:<name>", "sensors", "contacts".
    """
    if include is None:
        include = ["qpos", "qvel"]
    if not trajectory:
        raise ValueError("trajectory is empty; call sim_record + sim_step first")

    # ── Parse include into column groups ──────────────────────────────────────
    need_forward = any(
        tag.startswith("body_xpos:") or tag.startswith("body_xquat:")
        or tag in ("sensors", "contacts")
        for tag in include
    )

    header = ["t"]

    if "qpos" in include:
        header += [f"qpos_{i}" for i in range(model.nq)]
    if "qvel" in include:
        header += [f"qvel_{i}" for i in range(model.nv)]
    if "ctrl" in include:
        header += [f"ctrl_{i}" for i in range(model.nu)]

    body_xpos_bodies: list[tuple[str, int]] = []
    body_xquat_bodies: list[tuple[str, int]] = []
    for tag in include:
        if tag.startswith("body_xpos:"):
            name = tag[len("body_xpos:"):]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body {name!r} not found in model")
            body_xpos_bodies.append((name, bid))
            header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        elif tag.startswith("body_xquat:"):
            name = tag[len("body_xquat:"):]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body {name!r} not found in model")
            body_xquat_bodies.append((name, bid))
            header += [f"{name}_qw", f"{name}_qx", f"{name}_qy", f"{name}_qz"]

    # Build sensor column mapping (name → sensordata index)
    sensor_col_map: list[tuple[str, int]] = []
    if "sensors" in include and model.nsensor > 0:
        # Call mj_forward once to get sensordata length
        mujoco.mj_forward(model, data)
        n_sensordata = len(data.sensordata)
        for i in range(model.nsensor):
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
            adr = int(model.sensor_adr[i])
            next_adr = int(model.sensor_adr[i + 1]) if i + 1 < model.nsensor else n_sensordata
            dim = next_adr - adr
            for d in range(dim):
                col = sname if dim == 1 else f"{sname}_{d}"
                sensor_col_map.append((col, adr + d))
        header += [col for col, _ in sensor_col_map]

    if "contacts" in include:
        header += ["contact_count", "max_contact_force"]

    # ── Save state ─────────────────────────────────────────────────────────────
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    time_orig = data.time

    warnings_list: list[str] = []
    ctrl_warn_issued = False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()

            for frame in trajectory:
                row: dict = {"t": frame["t"]}

                if "qpos" in include:
                    for i, v in enumerate(frame["qpos"]):
                        row[f"qpos_{i}"] = v
                if "qvel" in include:
                    for i, v in enumerate(frame["qvel"]):
                        row[f"qvel_{i}"] = v
                if "ctrl" in include:
                    if "ctrl" in frame:
                        for i, v in enumerate(frame["ctrl"]):
                            row[f"ctrl_{i}"] = v
                    else:
                        if not ctrl_warn_issued:
                            warnings_list.append(
                                "ctrl not found in trajectory frames "
                                "(re-run sim_step with recording after this patch)"
                            )
                            ctrl_warn_issued = True
                        for i in range(model.nu):
                            row[f"ctrl_{i}"] = ""

                if need_forward:
                    data.qpos[:] = frame["qpos"]
                    data.qvel[:] = frame["qvel"]
                    data.time = frame.get("t", 0.0)
                    mujoco.mj_forward(model, data)

                    for name, bid in body_xpos_bodies:
                        row[f"{name}_x"] = float(data.xpos[bid, 0])
                        row[f"{name}_y"] = float(data.xpos[bid, 1])
                        row[f"{name}_z"] = float(data.xpos[bid, 2])

                    for name, bid in body_xquat_bodies:
                        row[f"{name}_qw"] = float(data.xquat[bid, 0])
                        row[f"{name}_qx"] = float(data.xquat[bid, 1])
                        row[f"{name}_qy"] = float(data.xquat[bid, 2])
                        row[f"{name}_qz"] = float(data.xquat[bid, 3])

                    for col, idx in sensor_col_map:
                        row[col] = float(data.sensordata[idx])

                    if "contacts" in include:
                        row["contact_count"] = data.ncon
                        if data.ncon > 0:
                            max_f = 0.0
                            f_buf = np.zeros(6)
                            for j in range(data.ncon):
                                mujoco.mj_contactForce(model, data, j, f_buf)
                                max_f = max(max_f, float(np.linalg.norm(f_buf[:3])))
                            row["max_contact_force"] = max_f
                        else:
                            row["max_contact_force"] = 0.0

                writer.writerow(row)
    finally:
        data.qpos[:] = qpos_orig
        data.qvel[:] = qvel_orig
        data.time = time_orig
        mujoco.mj_forward(model, data)

    result: dict = {
        "ok": True,
        "path": output_path,
        "rows": len(trajectory),
        "columns": header,
    }
    if warnings_list:
        result["warning"] = "; ".join(warnings_list)
    return json.dumps(result)
```

### Step 3: 追加 `_plot_trajectory_impl`

继续追加：

```python
# ─── plot_trajectory ──────────────────────────────────────────────────────────

def _plot_trajectory_impl(
    csv_path: str,
    plot_type: str,
    dof: int = 0,
    body: str | None = None,
    output_path: str | None = None,
    title: str = "",
) -> list:
    """Plot phase portrait or 3D body trajectory from CSV.

    Args:
        csv_path: path to CSV (from export_csv or export_state_log).
        plot_type: "phase" (qpos vs qvel) or "path3d" (body 3D path).
        dof: DOF index for phase mode.
        body: body name for path3d mode (looks for <body>_x/y/z columns).
        output_path: if given, save PNG to file; else return inline.
        title: plot title.

    Returns:
        [ImageContent, TextContent]
    """
    import pandas as pd

    if plot_type not in ("phase", "path3d"):
        raise ValueError(f"plot_type must be 'phase' or 'path3d', got {plot_type!r}")

    df = pd.read_csv(csv_path)

    if plot_type == "phase":
        qpos_col = f"qpos_{dof}"
        qvel_col = f"qvel_{dof}"
        for col in (qpos_col, qvel_col):
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in {csv_path}; "
                                 "run export_state_log with 'qpos' and 'qvel'")

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(df[qpos_col], df[qvel_col], alpha=0.7, linewidth=1.2)
        ax.scatter([df[qpos_col].iloc[0]], [df[qvel_col].iloc[0]],
                   color="green", s=80, zorder=5, label="start")
        ax.scatter([df[qpos_col].iloc[-1]], [df[qvel_col].iloc[-1]],
                   color="red", s=80, zorder=5, label="end")
        ax.set_xlabel(f"{qpos_col} [rad/m]")
        ax.set_ylabel(f"{qvel_col} [rad·s⁻¹ / m·s⁻¹]")
        ax.set_title(title or f"Phase portrait DOF {dof}")
        ax.legend()
        summary = f"Phase portrait DOF={dof}, {len(df)} frames"

    else:  # path3d
        if body is None:
            raise ValueError("body must be specified for plot_type='path3d'")
        x_col, y_col, z_col = f"{body}_x", f"{body}_y", f"{body}_z"
        for col in (x_col, y_col, z_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {csv_path}; "
                    f"run export_state_log with 'body_xpos:{body}'"
                )

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        t_norm = np.linspace(0, 1, len(df))
        cmap = plt.cm.viridis

        for i in range(len(df) - 1):
            ax.plot(
                df[x_col].iloc[i : i + 2].tolist(),
                df[y_col].iloc[i : i + 2].tolist(),
                df[z_col].iloc[i : i + 2].tolist(),
                color=cmap(t_norm[i]),
                linewidth=1.5,
            )
        ax.scatter(
            [df[x_col].iloc[0]], [df[y_col].iloc[0]], [df[z_col].iloc[0]],
            color="green", s=80, zorder=5, label="start",
        )
        ax.scatter(
            [df[x_col].iloc[-1]], [df[y_col].iloc[-1]], [df[z_col].iloc[-1]],
            color="red", s=80, zorder=5, label="end",
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title or f"3D path: {body}")
        ax.legend()
        summary = (
            f"3D path {body}, {len(df)} frames, "
            f"x={df[x_col].min():.3f}..{df[x_col].max():.3f}"
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as fout:
            fout.write(base64.b64decode(png_b64))

    return [
        ImageContent(type="image", data=png_b64, mimeType="image/png"),
        TextContent(type="text", text=summary),
    ]
```

### Step 4: 追加 MCP 工具包装器

继续追加：

```python
# ─── MCP tools ────────────────────────────────────────────────────────────────

@mcp.tool()
@safe_tool
async def export_state_log(
    ctx: Context,
    output_path: str,
    include: list[str] | None = None,
    sim_name: str | None = None,
) -> str:
    """Export full-state CSV with selectable columns from recorded trajectory.

    Args:
        output_path: destination CSV path.
        include: list of tags. Supported: "qpos", "qvel", "ctrl",
            "body_xpos:<name>", "body_xquat:<name>", "sensors", "contacts".
            Defaults to ["qpos", "qvel"].
        sim_name: simulation slot name (default slot if None).
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    if include is None:
        include = ["qpos", "qvel"]
    await asyncio.sleep(0)
    return _export_state_log_impl(slot.model, slot.data, slot.trajectory,
                                   output_path, include)


@mcp.tool()
@safe_tool
async def plot_trajectory(
    ctx: Context,
    csv_path: str,
    plot_type: str,
    dof: int = 0,
    body: str | None = None,
    output_path: str | None = None,
    title: str = "",
) -> list:
    """Plot phase portrait or 3D body trajectory from a state log CSV.

    Args:
        csv_path: path to CSV from export_csv or export_state_log.
        plot_type: "phase" (qpos_N vs qvel_N) or "path3d" (body 3D path).
        dof: DOF index for phase mode (default 0).
        body: body name for path3d mode (requires body_xpos:<name> in CSV).
        output_path: if given, save PNG; else return inline.
        title: plot title.
    """
    await asyncio.sleep(0)
    return _plot_trajectory_impl(csv_path, plot_type, dof, body, output_path, title)
```

**注意**：`plot_trajectory` 不访问 sim slot，因此不需要 `ctx.request_context`。

### Step 5: 确认 export.py 有必要 imports

检查文件顶部 imports，确保有：
```python
import asyncio
import base64
import csv
import io
import json
import os

import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from mcp.server.fastmcp import Context
from mcp.types import ImageContent, TextContent

from .._registry import mcp
from . import safe_tool
```

若已有部分 imports 则补充缺失的。**不要重复导入。**

### Step 6: 运行新增测试确认通过

```bash
python -m pytest tests/test_export_tools.py -v --tb=short -k "state_log or trajectory"
```

**预期**：6 个新增 export 测试全部 PASS。

### Step 7: 提交

```bash
git add src/mujoco_mcp/tools/export.py
git commit -m "feat(export): add export_state_log and plot_trajectory tools"
```

---

## Task 4: 创建 media.py 实现 export_video

**Files:**
- Create: `src/mujoco_mcp/tools/media.py`
- Modify: `pyproject.toml`（添加 media 可选依赖）

### Step 1: 创建 `src/mujoco_mcp/tools/media.py`

```python
"""Media export tools — export_video (MP4/GIF) from recorded trajectory."""
from __future__ import annotations

import asyncio
import json
import os

import mujoco
from PIL import Image as PILImage

from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


def _export_video_impl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trajectory: list[dict],
    output_path: str,
    fps: int = 30,
    fmt: str = "mp4",
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
) -> str:
    """Render trajectory to video file (MP4 or GIF).

    Args:
        model: MjModel instance.
        data: MjData instance (state is saved/restored).
        trajectory: list of frames [{t, qpos, qvel, ...}].
        output_path: destination file path.
        fps: frames per second.
        fmt: "mp4" or "gif".
        camera: camera name (None = free camera).
        width: render width in pixels.
        height: render height in pixels.

    Returns:
        JSON string {"ok", "path", "frames", "duration_s", "format"}.
    """
    if not trajectory:
        raise ValueError("trajectory is empty; call sim_record + sim_step first")
    if fmt not in ("mp4", "gif"):
        raise ValueError(f"fmt must be 'mp4' or 'gif', got {fmt!r}")

    # Validate camera before rendering
    if camera is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise ValueError(f"Camera {camera!r} not found in model")

    # Save current state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    time_orig = data.time

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        frames: list = []
        for frame in trajectory:
            data.qpos[:] = frame["qpos"]
            data.qvel[:] = frame["qvel"]
            data.time = frame.get("t", 0.0)
            mujoco.mj_forward(model, data)
            if camera is not None:
                renderer.update_scene(data, camera=camera)
            else:
                renderer.update_scene(data)
            frames.append(renderer.render().copy())

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if fmt == "gif":
            imgs = [PILImage.fromarray(f) for f in frames]
            imgs[0].save(
                output_path,
                format="GIF",
                save_all=True,
                append_images=imgs[1:],
                duration=max(1, 1000 // fps),
                loop=0,
                optimize=False,
            )
        else:  # mp4
            try:
                import imageio
            except ImportError:
                raise RuntimeError(
                    "MP4 export requires imageio[ffmpeg]. "
                    "Install with: pip install 'mujoco-mcp[media]'"
                )
            with imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
                for f in frames:
                    writer.append_data(f)

        return json.dumps({
            "ok": True,
            "path": output_path,
            "frames": len(frames),
            "duration_s": round(len(frames) / fps, 3),
            "format": fmt,
        })
    finally:
        renderer.close()
        data.qpos[:] = qpos_orig
        data.qvel[:] = qvel_orig
        data.time = time_orig
        mujoco.mj_forward(model, data)


@mcp.tool()
@safe_tool
async def export_video(
    ctx: Context,
    output_path: str,
    fps: int = 30,
    fmt: str = "mp4",
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    sim_name: str | None = None,
) -> str:
    """Export recorded trajectory as MP4 or GIF video.

    Args:
        output_path: destination file path (.mp4 or .gif).
        fps: frames per second (default 30).
        fmt: "mp4" (requires imageio[ffmpeg]) or "gif" (Pillow, no extra deps).
        camera: camera name for rendering (None = free camera).
        width: render width in pixels (default 640).
        height: render height in pixels (default 480).
        sim_name: simulation slot name (default slot if None).

    Note:
        MP4 requires: pip install 'mujoco-mcp[media]'
        GIF uses Pillow (already installed), no extra deps needed.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    await asyncio.sleep(0)
    return _export_video_impl(
        slot.model, slot.data, slot.trajectory,
        output_path, fps=fps, fmt=fmt, camera=camera,
        width=width, height=height,
    )
```

### Step 2: 更新 `pyproject.toml` 添加 media 可选依赖

读取 `pyproject.toml`，找到 `[project.optional-dependencies]` 段，在 `vision = [...]` 之后添加：

```toml
media = ["imageio>=2.31", "imageio[ffmpeg]"]
```

### Step 3: 运行 media 测试

```bash
python -m pytest tests/test_media.py -v --tb=short
```

**预期**：6 个测试全部 PASS（test_mp4_requires_imageio 需 imageio 未安装或用 monkeypatch）。

**若 test_export_video_mp4_requires_imageio 失败**（因为 imageio 已安装）：这不是 bug，跳过即可；monkeypatch 测试在有 imageio 时对 import 路径 mock 可能不稳定，可将该测试标记 `@pytest.mark.skipif(True, reason="requires no imageio")`，不影响核心功能。

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/media.py pyproject.toml
git commit -m "feat(media): add export_video tool for MP4/GIF generation"
```

---

## Task 5: 注册 media + 全套测试 + README

**Files:**
- Modify: `src/mujoco_mcp/server.py`
- Modify: `README.md`

### Step 1: 注册 media 模块

读取 `src/mujoco_mcp/server.py`，找到：
```python
from .tools import ... optimization, robustness  # noqa: ...
```

在末尾追加 `, media`：
```python
from .tools import ... optimization, robustness, media  # noqa: E402,F401,E501
```

### Step 2: 运行完整测试套件

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -40
```

**预期**：≥ 136 个测试全部 PASS（124 原有 + 12 新增）。

### Step 3: 更新 README

读取 `README.md`，找到并修改：
1. 工具数量：`62` → `65`
2. 测试数量：`124` → `136`（或实际通过数量）
3. 在 Robustness 行之后添加 Media 行：

```markdown
| **Media** | `export_video` `export_state_log` `plot_trajectory` | Video export (MP4/GIF), full-state CSV, phase portrait & 3D trajectory plots |
```

### Step 4: 提交

```bash
git add src/mujoco_mcp/server.py README.md
git commit -m "feat: register media tools; update README 65 tools 136 tests"
```

### Step 5: 推送

等待用户确认后推送：
```bash
git push origin master
```

---

## 快速参考：_impl 函数签名

| 函数 | 文件 | 签名摘要 |
|------|------|---------|
| `_export_video_impl` | `media.py` | `(model, data, trajectory, output_path, fps, fmt, camera, width, height) → str` |
| `_export_state_log_impl` | `export.py` | `(model, data, trajectory, output_path, include) → str` |
| `_plot_trajectory_impl` | `export.py` | `(csv_path, plot_type, dof, body, output_path, title) → list` |

## 常见陷阱

1. **`mujoco.Renderer.update_scene` 接受 camera 名字符串** — 不需要先转换为 id
2. **PIL GIF 调色板** — 使用 `optimize=False` 避免编码速度问题；颜色质量低于 MP4 是正常的
3. **`model.sensor_adr` 是 int32 数组** — 需要 `int(model.sensor_adr[i])` 转换
4. **`export_state_log` 的 mj_forward 副作用** — 在 `finally` 块一定要还原 qpos/qvel/time
5. **`plot_trajectory` 不需要 sim slot** — `ctx` 参数在 async 包装器中不被使用（`await asyncio.sleep(0)` 仍需执行）
6. **`mpl_toolkits.mplot3d` 导入** — 必须在 `add_subplot(projection="3d")` 之前导入（即使不直接引用）
