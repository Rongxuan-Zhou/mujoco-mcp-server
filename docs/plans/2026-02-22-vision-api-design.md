# Vision API Integration Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to create the implementation plan.

**Goal:** Add a `analyze_scene` MCP tool that sends rendered simulation frames to Gemini 2.5 Pro for high-quality natural language scene understanding.

**Architecture:** New tool in `tools/vision.py`. On each call: render current scene → send PNG + metadata + user prompt to Gemini 2.5 Pro API → return structured JSON analysis. No changes to existing tools.

**Tech Stack:** `google-genai>=1.0` Python SDK, `GEMINI_API_KEY` env var, MuJoCo existing renderer.

---

## Tool Interface

```python
analyze_scene(
    prompt: str,           # Natural language question about the scene
    sim_name: str | None,  # Simulation slot (default: active)
    camera: str | None,    # Camera for rendering (default: free camera)
    width: int = 640,      # Render resolution width
    height: int = 480,     # Render resolution height
) -> str                   # JSON response
```

**Returns:**
```json
{
  "analysis": "The Franka arm end-effector is at approximately (0.4, 0.1, 1.6)m ...",
  "model": "gemini-2.5-pro-latest",
  "image_sent": true,
  "sim_time": 0.0
}
```

## Data Flow

```
User prompt
    ↓
render scene → PNG bytes  (reuse existing renderer, same as render_snapshot)
    ↓
Build Gemini request:
  - System context: body names, joint names, qpos, sim time, contact count
  - Image: PNG bytes (inline, not URL)
  - User prompt
    ↓
google.genai SDK  →  Gemini 2.5 Pro API
    ↓
Parse text response → return JSON
```

## System Prompt Template

```
You are a robotics simulation analyst. The image shows a MuJoCo physics simulation.

Scene metadata:
- Bodies: {body_names}
- Joints: {joint_names} with current qpos: {qpos}
- Time: {t:.3f}s | Contacts: {n_contacts}

Answer the user's question concisely and precisely, focusing on spatial
relationships, robot pose, and physical state. Use metric units (meters, radians).
```

## Configuration

| Env var | Required | Default | Description |
|---------|----------|---------|-------------|
| `GEMINI_API_KEY` | Yes | — | Google AI Studio API key |
| `GEMINI_VISION_MODEL` | No | `gemini-2.5-pro-latest` | Model name override |

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| `GEMINI_API_KEY` not set | Return JSON error with setup instructions |
| GL renderer unavailable | Send scene_map text only, `image_sent: false` |
| Gemini API timeout / error | `@safe_tool` catches, returns JSON error |
| Invalid model name | Same, hint to check `GEMINI_VISION_MODEL` |

## Files Changed

| File | Change |
|------|--------|
| `src/mujoco_mcp/tools/vision.py` | **Create** — `analyze_scene` tool |
| `src/mujoco_mcp/server.py` | **Modify** — import `tools.vision` |
| `pyproject.toml` | **Modify** — add `google-genai>=1.0` |
| `tests/test_vision.py` | **Create** — unit tests (mock API) |

## Out of Scope

- Streaming responses (single-shot is sufficient)
- Caching API responses
- Supporting multiple vision providers (only Gemini 2.5 Pro)
- Auto-triggering vision analysis on every render
