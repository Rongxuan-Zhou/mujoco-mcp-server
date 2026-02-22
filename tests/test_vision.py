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
