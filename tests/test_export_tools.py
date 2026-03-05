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
    """export_csv with empty trajectory returns error JSON."""
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
    from mcp.types import TextContent, ImageContent
    ctx = _make_ctx(trajectory=_fake_traj(10))
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        asyncio.run(export_csv(ctx, output_path=path))
        plot_ctx = MagicMock()  # plot_data doesn't use ctx
        result = asyncio.run(plot_data(plot_ctx, csv_path=path))
        assert len(result) == 2
        assert isinstance(result[0], TextContent)
        assert isinstance(result[1], ImageContent)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_plot_data_missing_file():
    """plot_data returns error TextContent if CSV file does not exist."""
    from mujoco_mcp.tools.export import plot_data
    from mcp.types import TextContent
    ctx = MagicMock()
    result = asyncio.run(plot_data(ctx, csv_path="/nonexistent/path/file.csv"))
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    d = json.loads(result[0].text)
    assert "error" in d


# ── export_state_log + plot_trajectory tests ──────────────────────────────────

import json as _json_state  # noqa: E402
import mujoco as _mujoco  # noqa: E402

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
    from mujoco_mcp.tools.export import _export_state_log_impl
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "state.csv")
    result = _json_state.loads(_export_state_log_impl(model, data, traj, out,
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
    from mujoco_mcp.tools.export import _export_state_log_impl
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "xpos.csv")
    result = _json_state.loads(_export_state_log_impl(model, data, traj, out,
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
    from mujoco_mcp.tools.export import _export_state_log_impl
    model, data, traj = _make_contact_traj()
    out = str(tmp_path / "contacts.csv")
    result = _json_state.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["contacts"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    assert "contact_count" in header
    assert "max_contact_force" in header


def test_export_state_log_sensors(tmp_path):
    """sensors tag 应产生加速度计 accel_0/1/2 列（3D 传感器）。"""
    import csv as _csv
    from mujoco_mcp.tools.export import _export_state_log_impl
    model, data, traj = _make_sensor_traj()
    out = str(tmp_path / "sensors.csv")
    result = _json_state.loads(_export_state_log_impl(model, data, traj, out,
                                               include=["sensors"]))
    assert result["ok"] is True
    with open(out) as f:
        header = next(_csv.reader(f))
    assert any(col.startswith("accel") for col in header)


def test_plot_trajectory_phase_returns_image(tmp_path):
    """plot_trajectory phase 模式应返回 ImageContent。"""
    from mujoco_mcp.tools.export import _export_state_log_impl, _plot_trajectory_impl
    from mcp.types import ImageContent
    model, data, traj = _make_sensor_traj(n=50)
    csv_path = str(tmp_path / "traj.csv")
    _export_state_log_impl(model, data, traj, csv_path, include=["qpos", "qvel"])
    result = _plot_trajectory_impl(csv_path, "phase", dof=0)
    assert any(isinstance(r, ImageContent) for r in result)
    img_item = next(r for r in result if isinstance(r, ImageContent))
    assert len(img_item.data) > 100


def test_plot_trajectory_path3d_returns_image(tmp_path):
    """plot_trajectory path3d 模式应返回 ImageContent。"""
    from mujoco_mcp.tools.export import _export_state_log_impl, _plot_trajectory_impl
    from mcp.types import ImageContent
    model, data, traj = _make_sensor_traj(n=50)
    csv_path = str(tmp_path / "traj3d.csv")
    _export_state_log_impl(model, data, traj, csv_path,
                           include=["body_xpos:cart"])
    result = _plot_trajectory_impl(csv_path, "path3d", body="cart")
    assert any(isinstance(r, ImageContent) for r in result)
