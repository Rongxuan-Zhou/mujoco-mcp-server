import pytest
import json
import numpy as np

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
    from mujoco_mcp.tools.spatial import _body_name_to_id
    model, _ = model_data
    bid = _body_name_to_id(model, "table")
    assert bid > 0

def test_body_name_to_id_not_found(model_data):
    from mujoco_mcp.tools.spatial import _body_name_to_id
    model, _ = model_data
    with pytest.raises(ValueError, match="not found"):
        _body_name_to_id(model, "nonexistent")

def test_collect_subtree_leaf(model_data):
    from mujoco_mcp.tools.spatial import _body_name_to_id, _collect_subtree
    model, _ = model_data
    table_id = _body_name_to_id(model, "table")
    ids = _collect_subtree(model, table_id)
    assert table_id in ids

def test_collect_subtree_with_children(model_data):
    from mujoco_mcp.tools.spatial import _body_name_to_id, _collect_subtree
    model, _ = model_data
    ball_id = _body_name_to_id(model, "ball")
    child_id = _body_name_to_id(model, "ball_child")
    ids = _collect_subtree(model, ball_id)
    assert ball_id in ids
    assert child_id in ids

def test_body_aabb_box(model_data):
    from mujoco_mcp.tools.spatial import _body_aabb_impl
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    # table geom: box size [1.0, 0.5, 0.4], body pos [0, 0, 0.5]
    # Expected: x in [-1, 1], y in [-0.5, 0.5], z in [0.1, 0.9]
    assert lo[2] == pytest.approx(0.1, abs=1e-4)
    assert hi[2] == pytest.approx(0.9, abs=1e-4)

def test_surface_anchor_top_center(model_data):
    from mujoco_mcp.tools.spatial import _body_aabb_impl, _surface_anchor_impl
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "center")
    assert pt[2] == pytest.approx(hi[2], abs=1e-6)
    assert pt[0] == pytest.approx((lo[0] + hi[0]) / 2, abs=1e-6)

def test_surface_anchor_top_plus_x(model_data):
    from mujoco_mcp.tools.spatial import _body_aabb_impl, _surface_anchor_impl
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", include_children=False)
    pt = _surface_anchor_impl(lo, hi, "top", "+x")
    assert pt[0] == pytest.approx(hi[0], abs=1e-6)
    assert pt[2] == pytest.approx(hi[2], abs=1e-6)

def test_surface_anchor_invalid_surface(model_data):
    from mujoco_mcp.tools.spatial import _body_aabb_impl, _surface_anchor_impl
    model, data = model_data
    lo, hi = _body_aabb_impl(model, data, "table", False)
    with pytest.raises(ValueError, match="surface must be one of"):
        _surface_anchor_impl(lo, hi, "invalid", "center")
