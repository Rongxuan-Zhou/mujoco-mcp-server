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
