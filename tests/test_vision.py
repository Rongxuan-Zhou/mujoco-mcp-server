"""Tests for analyze_scene tool — mocks Gemini API, no real key needed."""
import json
import pytest
from unittest.mock import patch, MagicMock


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

    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from mujoco_mcp.tools.vision import analyze_scene
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
    assert "GEMINI_API_KEY" in data["error"]


def test_analyze_scene_no_renderer_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "Scene has 1 body based on metadata."
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from mujoco_mcp.tools.vision import analyze_scene
        ctx = _make_ctx(has_renderer=False)
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert data["image_sent"] is False
    assert "analysis" in data


def test_analyze_scene_gemini_error_handled(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API quota exceeded")

    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from mujoco_mcp.tools.vision import analyze_scene
        ctx = _make_ctx()
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert "error" in data
    assert "API quota exceeded" in data["error"]


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

    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from mujoco_mcp.tools.vision import compare_scenes
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

    with patch("mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from mujoco_mcp.tools.vision import compare_scenes
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
    assert "GEMINI_API_KEY" in data["error"]


def test_compare_scenes_only_slot_a_given(monkeypatch):
    """compare_scenes must reject when only slot_a is supplied."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    import asyncio
    from mujoco_mcp.tools.vision import compare_scenes
    ctx = _make_ctx_two_slots()
    result = asyncio.run(compare_scenes(ctx, prompt="compare", slot_a="alpha"))
    data = json.loads(result)
    assert "error" in data
    assert "slot_b" in data["error"]
