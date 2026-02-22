"""Phase 4g: Vision API tool — analyze_scene via Gemini 2.5 Pro."""

import asyncio
import json
import os
from io import BytesIO

import mujoco
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
    text = response.text
    if text is None:
        finish = getattr(response, "finish_reason", "unknown")
        raise ValueError(f"Gemini returned no text (finish_reason={finish})")
    return text


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
        return json.dumps({"error": "google-genai not installed. Run: pip install 'mujoco-mcp-server[vision]'"})

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return json.dumps({
            "error": "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key"
        })

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro-latest")

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    body_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
        for i in range(m.nbody)
    ]
    joint_names = [
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<joint:{i}>"
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

    png_bytes = None
    image_sent = False
    if slot.renderer is not None:
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
        except Exception as e:
            system_prompt += f"\n\n(Image capture failed: {e} — answering from metadata only.)"
    else:
        system_prompt += "\n\n(No image available — answer based on metadata only.)"

    try:
        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(None, lambda: _call_gemini(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=prompt,
            png_bytes=png_bytes,
        ))
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    return json.dumps({
        "analysis": analysis,
        "model": model,
        "image_sent": image_sent,
        "sim_time": float(d.time),
    }, ensure_ascii=False)
