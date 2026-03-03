"""Tests for analyze_scene tool — mocks Gemini API, no real key needed."""
import json
import pytest
from unittest.mock import patch, MagicMock

import mujoco_mcp.tools.vision as _vision_mod


@pytest.fixture(autouse=True)
def _clear_gemini_cache():
    _vision_mod._gemini_client_cache.clear()
    yield
    _vision_mod._gemini_client_cache.clear()


def _make_ctx(has_renderer=True):
    import numpy as np
    import mujoco

    model = mujoco.MjModel.from_xml_string("""
    <mujoco>
      <worldbody>
        <body name="box"><geom type="box" size=".1 .1 .1"/></body>
      </worldbody>
    </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from mujoco_mcp.sim_manager import SimSlot
    slot = SimSlot(name="test", model=model, data=data)

    if has_renderer:
        renderer_mock = MagicMock()
        renderer_mock.render.return_value = np.zeros((480, 640, 3), dtype="uint8")
        slot.renderer = renderer_mock

    sm = MagicMock()
    sm.get.return_value = slot
    if has_renderer:
        sm.require_renderer.return_value = slot.renderer
    else:
        sm.require_renderer.side_effect = RuntimeError("no GL")

    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = sm
    return ctx


def test_analyze_scene_returns_json_with_analysis(monkeypatch):
    """Happy path: Gemini returns text, tool wraps it in JSON."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "The robot arm is fully extended upward."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    import asyncio
    from mujoco_mcp.tools.vision import analyze_scene
    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        ctx = _make_ctx(has_renderer=True)
        result = asyncio.run(analyze_scene(ctx, prompt="Where is the arm?"))

    data = json.loads(result)
    assert "analysis" in data
    assert data["analysis"] == "The robot arm is fully extended upward."
    assert data["image_sent"] is True
    assert data["model"] == "gemini-2.5-pro"
    mock_client.models.generate_content.assert_called_once()


def test_analyze_scene_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import asyncio
    from mujoco_mcp.tools.vision import analyze_scene
    ctx = _make_ctx()
    result = asyncio.run(analyze_scene(ctx, prompt="describe the scene"))
    data = json.loads(result)
    assert "error" in data
    assert data["error"] == "NO_API_KEY"
    assert "GEMINI_API_KEY" in data["message"]


def test_analyze_scene_no_renderer_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "Scene has 1 body based on metadata."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    import asyncio
    from mujoco_mcp.tools.vision import analyze_scene
    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        ctx = _make_ctx(has_renderer=False)
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert data["image_sent"] is False
    assert "analysis" in data


def test_analyze_scene_gemini_error_handled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API quota exceeded")

    import asyncio
    from mujoco_mcp.tools.vision import analyze_scene
    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        ctx = _make_ctx()
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert "error" in data
    assert data["error"] in ("QUOTA_EXCEEDED", "GEMINI_ERROR")
    assert "API quota exceeded" in data["message"]


# ---------------------------------------------------------------------------
# compare_scenes tests
# ---------------------------------------------------------------------------

def _make_ctx_two_slots(has_renderer=True):
    """Context with two named slots: 'alpha' and 'beta'."""
    import numpy as np
    import mujoco
    from mujoco_mcp.sim_manager import SimSlot

    def _make_slot(name):
        m = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <worldbody>
            <body name="box"><geom type="box" size=".1 .1 .1"/></body>
          </worldbody>
        </mujoco>
        """)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        slot = SimSlot(name=name, model=m, data=d)
        if has_renderer:
            r = MagicMock()
            r.render.return_value = np.zeros((480, 640, 3), dtype="uint8")
            slot.renderer = r
        return slot

    slot_alpha = _make_slot("alpha")
    slot_beta = _make_slot("beta")

    sm = MagicMock()
    sm.get.side_effect = lambda name: {"alpha": slot_alpha, "beta": slot_beta}[name]
    sm.snapshot_slots.return_value = {"alpha": slot_alpha, "beta": slot_beta}
    if has_renderer:
        sm.require_renderer.side_effect = lambda slot: slot.renderer
    else:
        sm.require_renderer.side_effect = RuntimeError("no GL")

    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = sm
    return ctx


def test_compare_scenes_happy_path(monkeypatch):
    """compare_scenes returns comparison JSON with two slots."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "Slot B has higher velocity."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        ctx = _make_ctx_two_slots(has_renderer=True)
        result = asyncio.run(compare_scenes(ctx, prompt="What changed?", slot_a="alpha", slot_b="beta"))

    data = json.loads(result)
    assert "comparison" in data
    assert data["comparison"] == "Slot B has higher velocity."
    assert data["slots"] == ["alpha", "beta"]
    assert data["images_sent"] == 2
    mock_client.models.generate_content.assert_called_once()


def test_compare_scenes_auto_slot_selection(monkeypatch):
    """compare_scenes auto-picks first two slots when both slot args are None."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "Both look similar."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        ctx = _make_ctx_two_slots(has_renderer=True)
        result = asyncio.run(compare_scenes(ctx, prompt="Compare"))

    data = json.loads(result)
    assert "comparison" in data
    assert len(data["slots"]) == 2


def test_compare_scenes_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    ctx = _make_ctx_two_slots()
    result = asyncio.run(compare_scenes(ctx, prompt="compare", slot_a="alpha", slot_b="beta"))
    data = json.loads(result)
    assert "error" in data
    assert data["error"] == "NO_API_KEY"
    assert "GEMINI_API_KEY" in data["message"]


def test_compare_scenes_only_slot_a_given(monkeypatch):
    """compare_scenes must reject when only slot_a is supplied."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    ctx = _make_ctx_two_slots()
    result = asyncio.run(compare_scenes(ctx, prompt="compare", slot_a="alpha"))
    data = json.loads(result)
    assert "error" in data
    assert data["error"] == "INVALID_ARGS"
    assert "slot_b" in data["message"]


def test_compare_scenes_only_slot_b_given(monkeypatch):
    """compare_scenes must reject when only slot_b is supplied."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    ctx = _make_ctx_two_slots()
    result = asyncio.run(compare_scenes(ctx, prompt="compare", slot_b="beta"))
    data = json.loads(result)
    assert "error" in data
    assert data["error"] == "INVALID_ARGS"
    assert "slot_a" in data["message"]


def test_compare_scenes_auto_too_few_slots(monkeypatch):
    """compare_scenes auto-select must fail when fewer than 2 slots are loaded."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    import asyncio
    import mujoco
    from mujoco_mcp.sim_manager import SimSlot
    from mujoco_mcp.tools.vision import compare_scenes

    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='box'><geom type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    slot_only = SimSlot(name="only", model=model, data=data)

    sm = MagicMock()
    sm.snapshot_slots.return_value = {"only": slot_only}
    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = sm

    result = asyncio.run(compare_scenes(ctx, prompt="compare"))
    data_out = json.loads(result)
    assert "error" in data_out
    assert data_out["error"] == "INVALID_ARGS"
    assert "2" in data_out["message"]


# ---------------------------------------------------------------------------
# track_object tests
# ---------------------------------------------------------------------------

def test_track_object_no_prompt_returns_trajectory():
    """track_object without a prompt returns trajectory data and no analysis."""
    import asyncio
    from mujoco_mcp.tools.vision import track_object

    ctx = _make_ctx(has_renderer=True)
    result = asyncio.run(track_object(ctx, body_name="box", n_steps=10, capture_every=0))
    data = json.loads(result)

    assert data["body"] == "box"
    assert data["n_steps"] == 10
    assert isinstance(data["trajectory"], list)
    assert len(data["trajectory"]) == 10
    assert all("t" in pt and "pos" in pt for pt in data["trajectory"])
    assert len(data["trajectory"][0]["pos"]) == 3
    assert data["keyframes_captured"] == 0
    assert data["analysis"] is None
    assert data["model"] is None


def test_track_object_unknown_body_returns_error():
    """track_object with a nonexistent body name returns a JSON error."""
    import asyncio
    from mujoco_mcp.tools.vision import track_object

    ctx = _make_ctx(has_renderer=False)
    result = asyncio.run(track_object(ctx, body_name="does_not_exist", n_steps=5))
    data = json.loads(result)

    assert "error" in data
    assert data["error"] == "BODY_NOT_FOUND"
    assert "does_not_exist" in data["message"]
    assert "Available bodies" in data["message"]


# ── Task 1 tests ──────────────────────────────────────────────────────────────

from mujoco_mcp.tools.vision import _make_error, _call_with_retry


def test_make_error_basic_fields():
    result = json.loads(_make_error("NO_API_KEY", "key missing"))
    assert result["error"] == "NO_API_KEY"
    assert result["message"] == "key missing"


def test_make_error_extra_kwargs():
    result = json.loads(_make_error("QUOTA_EXCEEDED", "too many", retry_after=2.0))
    assert result["retry_after"] == 2.0


def test_call_with_retry_succeeds_immediately():
    calls = []
    def fn():
        calls.append(1)
        return "ok"
    assert _call_with_retry(fn) == "ok"
    assert len(calls) == 1


def test_call_with_retry_retries_on_429():
    calls = []
    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("429 RESOURCE_EXHAUSTED quota")
        return "ok"
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        result = _call_with_retry(fn, max_retries=3, base_delay=0.001)
    assert result == "ok"
    assert len(calls) == 3


def test_call_with_retry_raises_non_rate_limit():
    def fn():
        raise ValueError("bad model")
    with pytest.raises(ValueError, match="bad model"):
        _call_with_retry(fn, max_retries=3)

# ── Task 2 tests ──────────────────────────────────────────────────────────────

from mujoco_mcp.tools.vision import _render_slot_image


def test_render_slot_image_small_uses_png(monkeypatch):
    """Images <= 512x512 pixels should use PNG."""
    monkeypatch.delenv("MUJOCO_MCP_VISION_JPEG_QUALITY", raising=False)
    import numpy as np
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = pixels
    mock_mgr = MagicMock()
    mock_mgr.require_renderer.return_value = mock_renderer
    mock_slot = MagicMock()

    result = _render_slot_image(mock_mgr, mock_slot, camera=None, width=100, height=100)
    assert result is not None
    img_bytes, mime_type = result
    assert mime_type == "image/png"


def test_render_slot_image_large_uses_jpeg(monkeypatch):
    """Images with pixel count > 262144 use JPEG (512x512=262144 still PNG, 640x640=409600 uses JPEG)."""
    monkeypatch.setenv("MUJOCO_MCP_VISION_JPEG_QUALITY", "85")
    import numpy as np
    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = pixels
    mock_mgr = MagicMock()
    mock_mgr.require_renderer.return_value = mock_renderer
    mock_slot = MagicMock()

    result = _render_slot_image(mock_mgr, mock_slot, camera=None, width=640, height=640)
    assert result is not None
    img_bytes, mime_type = result
    assert mime_type == "image/jpeg"


# ── Task 3 tests ──────────────────────────────────────────────────────────────

from mujoco_mcp.tools.vision import _detect_intent


def test_detect_intent_physics_keywords():
    assert _detect_intent("What contact forces are acting?") == "physics"
    assert _detect_intent("Is there a collision between the box and the floor?") == "physics"
    assert _detect_intent("How much friction force is there?") == "physics"


def test_detect_intent_kinematics_keywords():
    assert _detect_intent("What is the joint angle of the elbow?") == "kinematics"
    assert _detect_intent("Where is the end effector positioned?") == "kinematics"
    assert _detect_intent("Describe the robot pose.") == "kinematics"


def test_detect_intent_comparison_keywords():
    assert _detect_intent("What changed between the two states?") == "comparison"
    assert _detect_intent("Compare the before and after positions.") == "comparison"


def test_detect_intent_general_fallback():
    assert _detect_intent("Tell me about the scene.") == "general"
    assert _detect_intent("Describe what you see.") == "general"


# ── Task 4 tests ──────────────────────────────────────────────────────────────


def _make_ctx_with_trajectory():
    """Build a mock context whose slot has a recorded trajectory."""
    import asyncio
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body><joint type="slide"/><geom type="sphere" size=".1"/></body>'
        '</worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from mujoco_mcp.sim_manager import SimSlot
    slot = SimSlot(name="traj_test", model=model, data=data)
    # Add fake trajectory frames
    slot.trajectory = [
        {"t": float(i) * 0.01, "qpos": list(data.qpos), "qvel": list(data.qvel)}
        for i in range(10)
    ]

    mgr = MagicMock()
    mgr.get.return_value = slot
    mgr.require_renderer.side_effect = RuntimeError("no GL")
    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = mgr
    return ctx, slot


def test_render_figure_strip_no_trajectory():
    """Empty trajectory returns error TextContent."""
    import asyncio
    import mujoco
    from mujoco_mcp.sim_manager import SimSlot
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>')
    data = mujoco.MjData(model)
    slot = SimSlot(name="empty", model=model, data=data)
    slot.trajectory = []
    mgr = MagicMock()
    mgr.get.return_value = slot
    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = mgr

    from mujoco_mcp.tools.vision import render_figure_strip
    result = asyncio.run(render_figure_strip(ctx, timestamps=[0.0, 0.05]))
    # result is list; first element is TextContent with error
    assert len(result) >= 1
    text_content = result[0]
    data_out = json.loads(text_content.text)
    assert "error" in data_out
    assert data_out["error"] == "NO_TRAJECTORY"


def test_render_figure_strip_restores_state():
    """After rendering, qpos/qvel/time must be restored to original values."""
    import asyncio
    import numpy as np
    ctx, slot = _make_ctx_with_trajectory()
    original_qpos = slot.data.qpos.copy()
    original_qvel = slot.data.qvel.copy()
    original_time = float(slot.data.time)

    from mujoco_mcp.tools.vision import render_figure_strip
    asyncio.run(render_figure_strip(ctx, timestamps=[0.02, 0.07]))

    np.testing.assert_array_almost_equal(slot.data.qpos, original_qpos)
    np.testing.assert_array_almost_equal(slot.data.qvel, original_qvel)
    assert abs(slot.data.time - original_time) < 1e-9


def test_render_figure_strip_empty_timestamps():
    """Empty timestamps list returns INVALID_ARGS error."""
    import asyncio
    ctx, slot = _make_ctx_with_trajectory()
    from mujoco_mcp.tools.vision import render_figure_strip
    result = asyncio.run(render_figure_strip(ctx, timestamps=[]))
    assert len(result) >= 1
    data_out = json.loads(result[0].text)
    assert data_out["error"] == "INVALID_ARGS"
