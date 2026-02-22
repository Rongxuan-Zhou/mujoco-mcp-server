# Vision API (analyze_scene) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an `analyze_scene` MCP tool that renders the current simulation frame and sends it to Gemini 2.5 Pro for natural language scene understanding.

**Architecture:** Single new file `tools/vision.py` with one MCP tool. Reuses the existing renderer pipeline from `rendering.py`. Falls back to text-only (scene_map) when GL is unavailable. Reads `GEMINI_API_KEY` and optional `GEMINI_VISION_MODEL` from environment.

**Tech Stack:** `google-genai>=1.0` Python SDK, MuJoCo existing renderer (PIL → PNG bytes → Gemini inline image).

---

### Task 1: Add `google-genai` dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Open pyproject.toml and add the dependency**

In `pyproject.toml`, find the `dependencies` list and add one line:

```toml
[project]
dependencies = [
    "mcp>=1.26",
    "mujoco>=2.3.1",
    "numpy>=1.24",
    "Pillow>=10.0",
    "pandas>=2.0",
    "matplotlib>=3.7",
    "packaging>=21.0",
    "scipy>=1.9",
    "gymnasium>=0.29",
    "google-genai>=1.0",   # ← add this line
]
```

**Step 2: Install**

```bash
pip install -e .
```

Expected: installs without errors, `google-genai` appears in `pip list`.

**Step 3: Verify import works**

```bash
python -c "from google import genai; from google.genai import types; print('ok')"
```

Expected: prints `ok`.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat(vision): add google-genai dependency"
```

---

### Task 2: Create `tools/vision.py` (TDD)

**Files:**
- Create: `tests/test_vision.py`
- Create: `src/mujoco_mcp/tools/vision.py`

**Step 1: Write the failing tests**

Create `tests/test_vision.py`:

```python
"""Tests for analyze_scene tool — mocks Gemini API, no real key needed."""
import json
import pytest
from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_ctx(has_renderer=True, nq=2):
    """Build a minimal fake MCP context."""
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

    from src.mujoco_mcp.sim_manager import SimSlot
    slot = SimSlot(name="test", model=model, data=data)

    if has_renderer:
        renderer_mock = MagicMock()
        renderer_mock.render.return_value = np.zeros((480, 640, 3), dtype="uint8")
        slot.renderer = renderer_mock

    sm = MagicMock()
    sm.get.return_value = slot
    sm.require_renderer.side_effect = (
        (lambda s: s.renderer) if has_renderer
        else (lambda s: (_ for _ in ()).throw(RuntimeError("no GL")))
    )

    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = sm
    return ctx


# ── tests ────────────────────────────────────────────────────────────────────

def test_analyze_scene_returns_json_with_analysis(monkeypatch):
    """Happy path: Gemini returns text, tool wraps it in JSON."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "The robot arm is fully extended upward."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        from src.mujoco_mcp.tools.vision import _call_gemini
        result = _call_gemini(
            api_key="fake-key",
            model="gemini-2.5-pro-latest",
            system_prompt="You are a robotics analyst.",
            user_prompt="Where is the arm?",
            png_bytes=b"\x89PNG\r\n",
        )

    assert result == "The robot arm is fully extended upward."
    mock_client.models.generate_content.assert_called_once()


def test_analyze_scene_no_api_key(monkeypatch):
    """Missing GEMINI_API_KEY → JSON error, no crash."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import asyncio
    from src.mujoco_mcp.tools.vision import analyze_scene

    ctx = _make_ctx()
    result = asyncio.run(analyze_scene(ctx, prompt="describe the scene"))
    data = json.loads(result)

    assert "error" in data
    assert "GEMINI_API_KEY" in data["error"]


def test_analyze_scene_no_renderer_fallback(monkeypatch):
    """No GL renderer → image_sent=False, falls back to text-only analysis."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_response = MagicMock()
    mock_response.text = "Scene has 1 body based on metadata."

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from src.mujoco_mcp.tools.vision import analyze_scene

        ctx = _make_ctx(has_renderer=False)
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert data["image_sent"] is False
    assert "analysis" in data


def test_analyze_scene_gemini_error_handled(monkeypatch):
    """Gemini API exception → JSON error returned, no crash."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API quota exceeded")

    with patch("src.mujoco_mcp.tools.vision.genai.Client", return_value=mock_client):
        import asyncio
        from src.mujoco_mcp.tools.vision import analyze_scene

        ctx = _make_ctx()
        result = asyncio.run(analyze_scene(ctx, prompt="describe"))

    data = json.loads(result)
    assert "error" in data
    assert "API quota exceeded" in data["error"]
```

**Step 2: Run tests to confirm they all FAIL**

```bash
pytest tests/test_vision.py -v
```

Expected: 4 errors — `ModuleNotFoundError: cannot import 'analyze_scene'`

**Step 3: Implement `src/mujoco_mcp/tools/vision.py`**

```python
"""Phase 4g: Vision API tool — analyze_scene via Gemini 2.5 Pro."""

import json
import os
from io import BytesIO

from PIL import Image
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_camera, update_scene
from . import safe_tool

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ── Private helper ────────────────────────────────────────────────────────────

def _call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    png_bytes: bytes | None,
) -> str:
    """Send image + prompt to Gemini. Returns response text."""
    client = genai.Client(api_key=api_key)

    parts = []
    if png_bytes:
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(data=png_bytes, mime_type="image/png")
            )
        )
    parts.append(genai_types.Part(text=user_prompt))

    response = client.models.generate_content(
        model=model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    )
    return response.text


# ── MCP Tool ──────────────────────────────────────────────────────────────────

@mcp.tool()
@safe_tool
async def analyze_scene(
    ctx: Context,
    prompt: str,
    sim_name: str | None = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
) -> str:
    """Analyze the current simulation scene using Gemini 2.5 Pro vision.

    Renders the scene, sends the image + simulation metadata to Gemini, and
    returns a natural language analysis.

    Requires GEMINI_API_KEY environment variable.
    Optional: GEMINI_VISION_MODEL (default: gemini-2.5-pro-latest).

    Args:
        prompt: Natural language question about the scene.
        sim_name: Simulation slot (default: active slot).
        camera: Named camera for rendering (default: free camera).
        width: Render width in pixels.
        height: Render height in pixels.

    Returns:
        JSON: {"analysis": str, "model": str, "image_sent": bool, "sim_time": float}
    """
    if not _GENAI_AVAILABLE:
        return json.dumps({"error": "google-genai not installed. Run: pip install google-genai>=1.0"})

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return json.dumps({
            "error": "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key"
        })

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro-latest")

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    # Build system prompt with scene metadata
    import mujoco as _mj
    body_names = [
        _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
        for i in range(m.nbody)
    ]
    joint_names = [
        _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_JOINT, i) or f"<joint:{i}>"
        for i in range(m.njnt)
    ]
    qpos_str = ", ".join(f"{v:.3f}" for v in d.qpos)

    system_prompt = (
        "You are a robotics simulation analyst. The image shows a MuJoCo physics simulation.\n\n"
        f"Scene metadata:\n"
        f"- Bodies: {body_names}\n"
        f"- Joints: {joint_names} with current qpos: [{qpos_str}]\n"
        f"- Time: {d.time:.3f}s | Contacts: {d.ncon}\n\n"
        "Answer the user's question concisely and precisely, focusing on spatial "
        "relationships, robot pose, and physical state. Use metric units (meters, radians)."
    )

    # Try to render image
    png_bytes = None
    image_sent = False
    try:
        renderer = mgr.require_renderer(slot)
        cam_id = resolve_camera(m, camera)
        update_scene(renderer, d, camera_id=cam_id)
        pixels = renderer.render()
        img = Image.fromarray(pixels).resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        image_sent = True
    except RuntimeError:
        # No GL renderer — fall back to text-only
        system_prompt += "\n\n(No image available — answer based on metadata only.)"

    # Call Gemini
    analysis = _call_gemini(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=prompt,
        png_bytes=png_bytes,
    )

    return json.dumps({
        "analysis": analysis,
        "model": model,
        "image_sent": image_sent,
        "sim_time": float(d.time),
    }, ensure_ascii=False)
```

**Step 4: Run tests — expect 4 PASS**

```bash
pytest tests/test_vision.py -v
```

Expected output:
```
PASSED tests/test_vision.py::test_analyze_scene_returns_json_with_analysis
PASSED tests/test_vision.py::test_analyze_scene_no_api_key
PASSED tests/test_vision.py::test_analyze_scene_no_renderer_fallback
PASSED tests/test_vision.py::test_analyze_scene_gemini_error_handled
```

**Step 5: Commit**

```bash
git add src/mujoco_mcp/tools/vision.py tests/test_vision.py
git commit -m "feat(vision): add analyze_scene tool with Gemini 2.5 Pro"
```

---

### Task 3: Register tool in server.py

**Files:**
- Modify: `src/mujoco_mcp/server.py:65`

**Step 1: Add the import line**

In `server.py`, find the line:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env  # noqa: E402, F401
```

Append `vision` to it:
```python
from .tools import spatial, menagerie, control, sensor_fusion, coordination, rl_env, vision  # noqa: E402, F401
```

**Step 2: Run full test suite to confirm nothing broken**

```bash
pytest tests/ -v --ignore=tests/test_vision.py -x
```

Expected: all existing tests pass (23 passed, 2 skipped as before).

**Step 3: Run vision tests one more time**

```bash
pytest tests/test_vision.py -v
```

Expected: 4 passed.

**Step 4: Commit**

```bash
git add src/mujoco_mcp/server.py
git commit -m "feat(vision): register analyze_scene in server"
```

---

### Task 4: Manual smoke test (requires real GEMINI_API_KEY)

> Skip if no API key available — the mock tests cover correctness.

**Step 1: Export API key**

```bash
export GEMINI_API_KEY=your_google_ai_studio_key
```

Get a free key at: https://aistudio.google.com/app/apikey

**Step 2: Start server and call the tool**

```bash
python -c "
import asyncio, os
os.environ['GEMINI_API_KEY'] = os.environ['GEMINI_API_KEY']

# Quick functional check without MCP transport
from unittest.mock import MagicMock
import mujoco, numpy as np

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name=\"box\"><geom type=\"box\" size=\".1 .1 .1\"/></body></worldbody></mujoco>')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

from src.mujoco_mcp.sim_manager import SimSlot
slot = SimSlot(name='test', model=model, data=data)
sm = MagicMock()
sm.get.return_value = slot
sm.require_renderer.side_effect = RuntimeError('no GL')
ctx = MagicMock()
ctx.request_context.lifespan_context.sim_manager = sm

from src.mujoco_mcp.tools.vision import analyze_scene
result = asyncio.run(analyze_scene(ctx, prompt='What bodies are in this scene?'))
import json
print(json.dumps(json.loads(result), indent=2, ensure_ascii=False))
"
```

Expected: JSON with `"analysis"` field containing Gemini's description of the scene.

**Step 3: If tests pass and smoke test works, task complete — no commit needed.**
