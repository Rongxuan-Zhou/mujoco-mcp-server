"""Tests for export_video (media.py)."""
import json
import os
import mujoco
import numpy as np
import pytest
from PIL import Image

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
    from mujoco_mcp.tools.media import _export_video_impl
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
    from mujoco_mcp.tools.media import _export_video_impl
    model, data = _make_fall()
    traj = _make_traj(model, data, n=8)
    out = str(tmp_path / "test8.gif")
    result = json.loads(_export_video_impl(model, data, traj, out, fps=10, fmt="gif"))
    assert result["frames"] == 8


def test_export_video_empty_trajectory_raises():
    """空轨迹应抛出 ValueError。"""
    from mujoco_mcp.tools.media import _export_video_impl
    model, data = _make_fall()
    with pytest.raises(ValueError, match="empty"):
        _export_video_impl(model, data, [], "/tmp/never.gif", fmt="gif")


def test_export_video_state_restored_after(tmp_path):
    """导出后 qpos/qvel 应与调用前一致（data 不应被 export 修改）。"""
    from mujoco_mcp.tools.media import _export_video_impl
    model, data = _make_fall()
    traj = _make_traj(model, data, n=5)
    # 在调用 _export_video_impl 之前快照（这是函数应当"保留"的状态）
    qpos_before = data.qpos.copy()
    qvel_before = data.qvel.copy()
    out = str(tmp_path / "state.gif")
    _export_video_impl(model, data, traj, out, fps=10, fmt="gif")
    assert np.allclose(data.qpos, qpos_before)
    assert np.allclose(data.qvel, qvel_before)


def test_export_video_mp4_requires_imageio(tmp_path):
    """无 imageio 时应抛出 RuntimeError 含 'imageio'。"""
    from unittest.mock import patch
    from mujoco_mcp.tools.media import _export_video_impl
    model, data = _make_fall()
    traj = _make_traj(model, data, n=3)
    # patch.dict 将 "imageio" 设为 None，使 import imageio 抛出 ImportError
    with patch.dict("sys.modules", {"imageio": None}):
        with pytest.raises(RuntimeError, match="imageio"):
            _export_video_impl(model, data, traj, str(tmp_path / "out.mp4"), fmt="mp4")


def test_export_video_invalid_format_raises(tmp_path):
    """不支持的格式应抛出 ValueError。"""
    from mujoco_mcp.tools.media import _export_video_impl
    model, data = _make_fall()
    traj = _make_traj(model, data, n=3)
    with pytest.raises(ValueError, match="fmt"):
        _export_video_impl(model, data, traj, str(tmp_path / "out.avi"), fmt="avi")
