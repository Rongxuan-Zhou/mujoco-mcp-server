# Vision Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在工程质量、性能、分析质量、功能四个维度各做最核心的一件改进，全面提升 MuJoCo MCP Vision API。

**Architecture:** 按依赖顺序分 4 个任务：先建工程基础（错误码+重试+客户端缓存），再优化性能（图像格式自适应），再提升分析质量（intent 路由），最后新增 render_figure_strip 工具。每个任务独立可测。

**Tech Stack:** Python 3.10+, google-genai>=1.0, Pillow, MuJoCo 2.3+, pytest, asyncio

---

## Task 1: 工程基础 — 错误码标准化 + 重试 + 客户端缓存

**Files:**
- Modify: `src/mujoco_mcp/tools/vision.py:1-25`（imports + 模块级常量）
- Test: `tests/test_vision.py`（追加新测试）

### Step 1: 写失败测试

在 `tests/test_vision.py` 末尾追加：

```python
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
    # patch time.sleep to avoid actual waiting
    import unittest.mock as mock
    with mock.patch("time.sleep"):
        result = _call_with_retry(fn, max_retries=3, base_delay=0.001)
    assert result == "ok"
    assert len(calls) == 3


def test_call_with_retry_raises_non_rate_limit():
    def fn():
        raise ValueError("bad model")
    import pytest
    with pytest.raises(ValueError, match="bad model"):
        _call_with_retry(fn, max_retries=3)
```

运行：`pytest tests/test_vision.py::test_make_error_basic_fields -v`
期望：**FAIL**（`_make_error` 未定义）

### Step 2: 实现 `_make_error`, `_call_with_retry`, `_get_client`

在 `vision.py` 的 `logger = ...` 行之后（大约第 18 行），`try: from google...` 之前，插入：

```python
import time

# ── Error helpers ─────────────────────────────────────────────────────────────

def _make_error(code: str, message: str, **kwargs) -> str:
    """Return a JSON error string with standardised fields."""
    return json.dumps({"error": code, "message": message, **kwargs}, ensure_ascii=False)


# ── Gemini client cache ───────────────────────────────────────────────────────

_gemini_client_cache: "dict[str, genai.Client]" = {}


def _get_client(api_key: str) -> "genai.Client":
    """Return a cached Gemini client for *api_key*."""
    if api_key not in _gemini_client_cache:
        _gemini_client_cache[api_key] = genai.Client(api_key=api_key)
    return _gemini_client_cache[api_key]


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _call_with_retry(fn, max_retries: int = 3, base_delay: float = 2.0):
    """Call *fn()* with exponential-backoff retry on 429 / RESOURCE_EXHAUSTED."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise
```

然后，把 `_call_gemini` 和 `_call_gemini_multi_image` 内的 `genai.Client(api_key=api_key)` 改为 `_get_client(api_key)`。

把现有所有直接 `return json.dumps({"error": "..."})` 的地方统一改为 `return _make_error(CODE, message)` 格式：

| 位置 | 旧代码 | 新代码 |
|---|---|---|
| `_GENAI_AVAILABLE` 检查 | `json.dumps({"error": "google-genai not installed..."})` | `_make_error("NO_GENAI", "google-genai not installed. Run: pip install 'mujoco-mcp-server[vision]'")` |
| `GEMINI_API_KEY` 检查 | `json.dumps({"error": "GEMINI_API_KEY not set..."})` | `_make_error("NO_API_KEY", "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key")` |
| Gemini 调用异常 | `json.dumps({"error": f"{type(e).__name__}: {e}"})` | `_make_error("QUOTA_EXCEEDED" if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else "GEMINI_ERROR", str(e))` |
| `body_name` 不存在 | `json.dumps({"error": f"Body '{body_name}' not found..."})` | `_make_error("BODY_NOT_FOUND", f"Body '{body_name}' not found. Available bodies: {available}")` |
| `compare_scenes` slot_b 缺失 | `json.dumps({"error": "slot_b must be..."})` | `_make_error("INVALID_ARGS", "slot_b must be provided when slot_a is specified...")` |

同时把所有 Gemini 调用套上 `_call_with_retry`：

```python
# _call_gemini 内部，response 获取前：
response = _call_with_retry(lambda: client.models.generate_content(...))
```

### Step 3: 运行测试

```bash
pytest tests/test_vision.py -v --tb=short
```

期望：全部原有测试 + 5 个新测试 = **15 passed**

注意：原有测试里检查 `data["error"]` 的断言可能需要更新为检查 `data["message"]`（因为 `error` 字段现在是错误码）。例如：
- `test_analyze_scene_no_api_key`：改为 `assert data["error"] == "NO_API_KEY"`
- `test_analyze_scene_gemini_error_handled`：改为 `assert "API quota" in data["message"]`（若测试验证 quota 错误）

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/vision.py tests/test_vision.py
git commit -m "feat(vision): add _make_error, _call_with_retry, _get_client — engineering foundations"
```

---

## Task 2: 性能 — 图像格式自适应（PNG vs JPEG）

**Files:**
- Modify: `src/mujoco_mcp/tools/vision.py`（`_render_slot_png` → `_render_slot_image`）
- Test: `tests/test_vision.py`（追加测试）

### Step 1: 写失败测试

```python
# ── Task 2 tests ──────────────────────────────────────────────────────────────

from mujoco_mcp.tools.vision import _render_slot_image


def test_render_slot_image_small_uses_png(monkeypatch):
    """Images <= 512×512 pixels should use PNG."""
    monkeypatch.delenv("MUJOCO_MCP_VISION_JPEG_QUALITY", raising=False)
    # mock mgr and slot — renderer returns a 100×100 pixel array
    import numpy as np
    from unittest.mock import MagicMock
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = pixels
    mock_mgr = MagicMock()
    mock_mgr.require_renderer.return_value = mock_renderer
    mock_slot = MagicMock()
    mock_slot.model = MagicMock()
    mock_slot.data = MagicMock()

    result = _render_slot_image(mock_mgr, mock_slot, camera=None, width=100, height=100)
    assert result is not None
    img_bytes, mime_type = result
    assert mime_type == "image/png"


def test_render_slot_image_large_uses_jpeg(monkeypatch):
    """Images > 512×512 pixels should use JPEG."""
    monkeypatch.setenv("MUJOCO_MCP_VISION_JPEG_QUALITY", "85")
    import numpy as np
    from unittest.mock import MagicMock
    pixels = np.zeros((640, 640, 3), dtype=np.uint8)
    mock_renderer = MagicMock()
    mock_renderer.render.return_value = pixels
    mock_mgr = MagicMock()
    mock_mgr.require_renderer.return_value = mock_renderer
    mock_slot = MagicMock()
    mock_slot.model = MagicMock()
    mock_slot.data = MagicMock()

    result = _render_slot_image(mock_mgr, mock_slot, camera=None, width=640, height=640)
    assert result is not None
    img_bytes, mime_type = result
    assert mime_type == "image/jpeg"
```

运行：`pytest tests/test_vision.py::test_render_slot_image_small_uses_png -v`
期望：**FAIL**（`_render_slot_image` 未定义）

### Step 2: 实现 `_render_slot_image`

把 `_render_slot_png` 函数替换为：

```python
def _render_slot_image(
    mgr,
    slot,
    camera: str | None,
    width: int,
    height: int,
) -> "tuple[bytes, str] | None":
    """Render *slot* to image bytes. Returns (bytes, mime_type) or None on failure.

    Uses JPEG for large images (width*height > 262144) to reduce token usage,
    PNG otherwise. JPEG quality controlled by MUJOCO_MCP_VISION_JPEG_QUALITY env var (default 85).
    """
    try:
        renderer = mgr.require_renderer(slot)
        cam_id = resolve_camera(slot.model, camera)
        update_scene(renderer, slot.data, camera_id=cam_id)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        if img.width != width or img.height != height:
            img = img.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        use_jpeg = (width * height) > 262144
        if use_jpeg:
            quality = int(os.environ.get("MUJOCO_MCP_VISION_JPEG_QUALITY", "85"))
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            mime_type = "image/jpeg"
        else:
            img.save(buf, format="PNG", optimize=True)
            mime_type = "image/png"
        return buf.getvalue(), mime_type
    except Exception as e:
        logger.warning("_render_slot_image failed for slot %r: %s", getattr(slot, "name", "?"), e)
        return None
```

然后更新所有调用 `_render_slot_png` 的地方：
- `analyze_scene`：把 `_render_slot_png(...)` 改为 `_render_slot_image(...)`，解包 `(png_bytes, mime_type)` 元组（None 检查不变）
- `compare_scenes`：同上，`png_bytes_a` → `(img_bytes_a, mime_a)` 或 `None`
- `_call_gemini` 和 `_call_gemini_multi_image`：`mime_type` 参数改用实际值（不再硬编码 `"image/png"`），函数签名增加 `mime_type: str = "image/png"` 参数
- `track_object`：`_render_slot_png` 改为 `_render_slot_image`，处理返回值变化

### Step 3: 运行测试

```bash
pytest tests/test_vision.py -v --tb=short
```

期望：所有现有测试 + 2 个新测试 = **17 passed**

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/vision.py tests/test_vision.py
git commit -m "perf(vision): adaptive image format (JPEG for large images), cache Gemini client"
```

---

## Task 3: 分析质量 — prompt 意图路由

**Files:**
- Modify: `src/mujoco_mcp/tools/vision.py`（`_build_system_prompt` + `_detect_intent`）
- Test: `tests/test_vision.py`（追加测试）

### Step 1: 写失败测试

```python
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
```

运行：`pytest tests/test_vision.py::test_detect_intent_general_fallback -v`
期望：**FAIL**（`_detect_intent` 未定义）

### Step 2: 实现 `_detect_intent` 和更新 `_build_system_prompt`

在 `_build_system_prompt` 函数之前插入：

```python
# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "physics": ["contact", "touch", "force", "collision", "friction", "penetration", "impact"],
    "kinematics": ["pose", "position", "joint", "angle", "where", "extend", "reach", "end-effector", "link"],
    "comparison": ["compare", "difference", "change", "before", "after", "versus", "vs", "between"],
}


def _detect_intent(prompt: str) -> str:
    """Classify *prompt* into one of: physics, kinematics, comparison, general."""
    lower = prompt.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent
    return "general"
```

更新 `_build_system_prompt` 签名和末尾的 Response Format 段：

```python
def _build_system_prompt(m: "mujoco.MjModel", d: "mujoco.MjData", intent: str = "general") -> str:
    ...
    # 在 parts 末尾，根据 intent 追加专项引导（替换现有通用格式说明）
    _INTENT_GUIDANCE = {
        "physics": (
            "**Physics Focus**: Quantify contact normal forces (estimated from penetration depth and stiffness). "
            "Name the geom pairs in contact. Report penetration depth in millimeters. "
            "Describe friction direction if visible."
        ),
        "kinematics": (
            "**Kinematics Focus**: List each joint angle in radians. "
            "Describe the end-effector world position in (x, y, z) meters. "
            "Note any joint near its limit (within 10% of range)."
        ),
        "comparison": (
            "**Comparison Focus**: For each body/joint that changed, report Δposition and Δangle. "
            "Use format: 'body_name: before → after (Δ = value)'. "
            "Summarise the most significant change first."
        ),
        "general": "",
    }

    guidance = _INTENT_GUIDANCE.get(intent, "")
    parts += [
        "",
        "## Response Format",
        "Structure your answer with these sections (omit sections not relevant to the question):",
        "**Pose**: Describe body/joint positions and orientations.",
        "**Contacts**: Describe what is touching what.",
        "**Dynamics**: Velocity or energy observations.",
        "**Answer**: Direct answer to the user's question.",
    ]
    if guidance:
        parts += ["", guidance]
    parts += ["", "Be concise and precise. Use metric units (meters, radians). Prefer numbers over vague terms."]

    return "\n".join(parts)
```

在 `analyze_scene` 中更新调用：

```python
intent = _detect_intent(prompt)
system_prompt = _build_system_prompt(m, d, intent=intent)
```

在 `compare_scenes` 中（对 comparison 意图固定使用 `"comparison"`）：

```python
prompt_a = _build_system_prompt(slot_obj_a.model, slot_obj_a.data, intent="comparison")
prompt_b = _build_system_prompt(slot_obj_b.model, slot_obj_b.data, intent="comparison")
```

### Step 3: 运行测试

```bash
pytest tests/test_vision.py -v --tb=short
```

期望：所有现有测试 + 4 个新测试 = **21 passed**

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/vision.py tests/test_vision.py
git commit -m "feat(vision): add _detect_intent for prompt-aware system prompt routing"
```

---

## Task 4: 功能扩展 — `render_figure_strip` 工具

**Files:**
- Modify: `src/mujoco_mcp/tools/vision.py`（新增工具）
- Test: `tests/test_vision.py`（追加测试）

轨迹格式参考（来自 `sim_record` + `sim_step`）：
```python
slot.trajectory = [
    {"t": 0.001, "qpos": [...], "qvel": [...]},
    {"t": 0.002, "qpos": [...], "qvel": [...]},
    ...
]
```

### Step 1: 写失败测试

```python
# ── Task 4 tests ──────────────────────────────────────────────────────────────
import asyncio
from unittest.mock import MagicMock, patch
import numpy as np


def _make_ctx_with_trajectory(qpos_len=2):
    """Build a mock context whose slot has a recorded trajectory."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>'
        '<body><joint type="slide"/><geom type="sphere" size=".1"/></body>'
        '</worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    from src.mujoco_mcp.sim_manager import SimSlot
    slot = SimSlot(name="traj_test", model=model, data=data)
    # Add fake trajectory frames
    slot.trajectory = [
        {"t": float(i) * 0.01, "qpos": [float(i) * 0.01], "qvel": [0.0]}
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
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody></worldbody></mujoco>')
    data = mujoco.MjData(model)
    from src.mujoco_mcp.sim_manager import SimSlot
    slot = SimSlot(name="empty", model=model, data=data)
    slot.trajectory = []
    mgr = MagicMock()
    mgr.get.return_value = slot
    ctx = MagicMock()
    ctx.request_context.lifespan_context.sim_manager = mgr

    from src.mujoco_mcp.tools.vision import render_figure_strip
    result = asyncio.run(render_figure_strip(ctx, timestamps=[0.0, 0.05]))
    # result is list; first element is TextContent with error
    assert len(result) >= 1
    text_content = result[0]
    data_out = json.loads(text_content.text)
    assert "error" in data_out
    assert data_out["error"] == "NO_TRAJECTORY"


def test_render_figure_strip_restores_state():
    """After rendering, qpos must be restored to its original value."""
    ctx, slot = _make_ctx_with_trajectory()
    original_qpos = slot.data.qpos.copy()

    from src.mujoco_mcp.tools.vision import render_figure_strip
    asyncio.run(render_figure_strip(ctx, timestamps=[0.02, 0.07]))

    import numpy as np
    np.testing.assert_array_almost_equal(slot.data.qpos, original_qpos)
```

运行：`pytest tests/test_vision.py::test_render_figure_strip_no_trajectory -v`
期望：**FAIL**（`render_figure_strip` 未定义）

### Step 2: 实现 `render_figure_strip`

在 `vision.py` 末尾（`track_object` 之后）追加：

```python
# ---------------------------------------------------------------------------
# render_figure_strip tool
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def render_figure_strip(
    ctx: Context,
    timestamps: list[float],
    sim_name: str | None = None,
    camera: str | None = None,
) -> list:
    """Render frames at specified timestamps from a recorded trajectory.

    Sets simulation state to the nearest recorded frame for each timestamp
    and renders a snapshot. Useful for generating paper figures.

    Requires a recorded trajectory (use sim_record + sim_step first).

    Args:
        timestamps: List of time values (seconds) to render.
        sim_name: Simulation slot name.
        camera: Named camera for rendering.

    Returns:
        [TextContent(summary JSON), ImageContent, ImageContent, ...]
    """
    from mcp.types import TextContent, ImageContent
    import base64
    import bisect

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    traj = slot.trajectory
    if not traj:
        return [TextContent(
            type="text",
            text=_make_error("NO_TRAJECTORY",
                             "No trajectory recorded. Use sim_record(action='start') then sim_step."),
        )]

    # Save current state to restore afterwards
    saved_qpos = d.qpos.copy()
    saved_qvel = d.qvel.copy()
    saved_time = float(d.time)

    # Build sorted list of timestamps from trajectory for binary search
    traj_times = [frame["t"] for frame in traj]

    timestamps_rendered = []
    nearest_frames = []
    images: list = []

    for ts in timestamps:
        # Binary search for nearest frame
        idx = bisect.bisect_left(traj_times, ts)
        if idx == 0:
            nearest_idx = 0
        elif idx >= len(traj):
            nearest_idx = len(traj) - 1
        else:
            # Pick whichever neighbour is closer
            before = traj_times[idx - 1]
            after = traj_times[idx]
            nearest_idx = idx - 1 if abs(ts - before) <= abs(ts - after) else idx

        frame = traj[nearest_idx]
        actual_t = frame["t"]

        # Restore state to this frame
        qpos = frame["qpos"]
        qvel = frame["qvel"]
        if len(qpos) == m.nq:
            d.qpos[:] = qpos
        if len(qvel) == m.nv:
            d.qvel[:] = qvel
        d.time = actual_t
        mujoco.mj_forward(m, d)

        # Render
        result = _render_slot_image(mgr, slot, camera, width=640, height=480)
        if result is not None:
            img_bytes, mime_type = result
            images.append(ImageContent(
                type="image",
                data=base64.b64encode(img_bytes).decode(),
                mimeType=mime_type,
            ))

        timestamps_rendered.append(ts)
        nearest_frames.append(actual_t)

    # Restore original state
    d.qpos[:] = saved_qpos
    d.qvel[:] = saved_qvel
    d.time = saved_time
    mujoco.mj_forward(m, d)

    summary = TextContent(
        type="text",
        text=json.dumps({
            "timestamps_requested": timestamps,
            "timestamps_rendered": timestamps_rendered,
            "nearest_frames": nearest_frames,
            "images_rendered": len(images),
        }, ensure_ascii=False),
    )

    return [summary] + images
```

### Step 3: 运行测试

```bash
pytest tests/test_vision.py -v --tb=short
```

期望：所有现有测试 + 2 个新测试 = **23 passed**

### Step 4: 提交

```bash
git add src/mujoco_mcp/tools/vision.py tests/test_vision.py
git commit -m "feat(vision): add render_figure_strip tool for trajectory keyframe rendering"
```

---

## 最终验证

```bash
# 全量测试（排除已知网络失败的 menagerie 测试）
pytest tests/ --ignore=tests/test_menagerie.py -v --tb=short
```

期望：**26+ passed, 0 failed**

---

## 快速参考

| Task | 新增函数 | 关键测试 |
|---|---|---|
| 1 | `_make_error`, `_call_with_retry`, `_get_client` | `test_call_with_retry_retries_on_429` |
| 2 | `_render_slot_image` (replaces `_render_slot_png`) | `test_render_slot_image_large_uses_jpeg` |
| 3 | `_detect_intent`, `_build_system_prompt(intent=)` | `test_detect_intent_physics_keywords` |
| 4 | `render_figure_strip` | `test_render_figure_strip_restores_state` |
