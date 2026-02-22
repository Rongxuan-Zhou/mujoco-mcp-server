"""Phase 4g: Vision API tool — analyze_scene via Gemini 2.5 Pro."""

import asyncio
import json
import logging
import os
from io import BytesIO

import mujoco
from PIL import Image
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_camera, update_scene
from . import safe_tool

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


def _build_system_prompt(m: "mujoco.MjModel", d: "mujoco.MjData") -> str:
    """Build a rich system prompt with full scene metadata for Gemini."""
    # Body names and world positions
    body_info = []
    for i in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
        pos = d.xpos[i]
        body_info.append(f"  {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})m")

    # Joint state: name, qpos, qvel
    joint_lines = []
    qpos_idx = 0
    for i in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f"<joint:{i}>"
        jtype = m.jnt_type[i]
        # Free joint has 7 qpos, hinge/slide has 1
        nq = 7 if jtype == mujoco.mjtJoint.mjJNT_FREE else 1
        qp = d.qpos[qpos_idx:qpos_idx + nq]
        qv = d.qvel[qpos_idx:qpos_idx + nq] if qpos_idx + nq <= len(d.qvel) else []
        qpos_str = ", ".join(f"{v:.3f}" for v in qp)
        qvel_str = ", ".join(f"{v:.3f}" for v in qv)
        joint_lines.append(f"  {name}: qpos=[{qpos_str}] qvel=[{qvel_str}] rad/s")
        qpos_idx += nq

    # Actuator commands
    actuator_lines = []
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"<act:{i}>"
        actuator_lines.append(f"  {name}: ctrl={d.ctrl[i]:.3f}")

    # Contact summary (top 5 by force magnitude)
    contact_lines = []
    for c in range(min(d.ncon, 5)):
        con = d.contact[c]
        geom1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or f"geom{con.geom1}"
        geom2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or f"geom{con.geom2}"
        contact_lines.append(
            f"  {geom1} <-> {geom2} at ({con.pos[0]:.3f},{con.pos[1]:.3f},{con.pos[2]:.3f})m"
        )

    parts = [
        "You are an expert robotics simulation analyst examining a MuJoCo physics simulation.",
        "",
        "## Scene State",
        f"Time: {d.time:.4f}s | Active contacts: {d.ncon}",
        "",
        "### Body Positions (world frame)",
        "\n".join(body_info),
        "",
        "### Joint State (position & velocity)",
        "\n".join(joint_lines) if joint_lines else "  (no joints)",
    ]

    if actuator_lines:
        parts += ["", "### Actuator Commands", "\n".join(actuator_lines)]

    if contact_lines:
        parts += ["", f"### Active Contacts ({d.ncon} total, showing up to 5)", "\n".join(contact_lines)]

    parts += [
        "",
        "## Response Format",
        "Structure your answer with these sections (omit sections not relevant to the question):",
        "**Pose**: Describe body/joint positions and orientations.",
        "**Contacts**: Describe what is touching what and estimated forces.",
        "**Dynamics**: Velocity, acceleration, or energy observations.",
        "**Answer**: Direct answer to the user's question.",
        "",
        "Be concise and precise. Use metric units (meters, radians, N). Prefer numbers over vague terms.",
    ]

    return "\n".join(parts)


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
    Optional: GEMINI_VISION_MODEL (default: gemini-2.5-pro).

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

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro")

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    system_prompt = _build_system_prompt(m, d)

    png_bytes = None
    image_sent = False
    try:
        renderer = mgr.require_renderer(slot)  # raises RuntimeError when no GL renderer available
        cam_id = resolve_camera(m, camera)
        update_scene(renderer, d, camera_id=cam_id)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        if img.width != width or img.height != height:  # only resize when dimensions differ (fix I-2)
            img = img.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        png_bytes = buf.getvalue()
        image_sent = True
    except RuntimeError:
        # require_renderer raises RuntimeError when no GL renderer available
        system_prompt += "\n\n*Note: No rendered image available — analysis based on metadata only.*"
    except Exception as e:
        # Renderer present but image capture failed
        system_prompt += f"\n\n(Image capture failed: {e} — answering from metadata only.)"
        logger.warning(f"Vision tool: image capture failed: {e}")

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


# ---------------------------------------------------------------------------
# Private helpers shared by analyze_scene and compare_scenes
# ---------------------------------------------------------------------------

def _render_slot_png(
    mgr,
    slot,
    camera: str | None,
    width: int,
    height: int,
) -> "bytes | None":
    """Render *slot* to a PNG byte-string.  Returns None when rendering fails."""
    try:
        renderer = mgr.require_renderer(slot)
        cam_id = resolve_camera(slot.model, camera)
        update_scene(renderer, slot.data, camera_id=cam_id)
        pixels = renderer.render()
        img = Image.fromarray(pixels)
        if img.width != width or img.height != height:
            img = img.resize((width, height), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    except Exception as e:
        logger.warning("_render_slot_png failed for slot %r: %s", getattr(slot, "name", "?"), e)
        return None


def _call_gemini_multi_image(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: "list[bytes | None]",
) -> str:
    """Send multiple images + prompt to Gemini.  Returns response text.

    Each image is preceded by a text label ("Image 1:", "Image 2:", …).
    None entries are silently skipped (the label is still omitted).
    """
    client = genai.Client(api_key=api_key)

    parts: list = []
    image_index = 0
    for img_bytes in images:
        image_index += 1
        if img_bytes is None:
            continue
        parts.append(genai_types.Part(text=f"Image {image_index}:"))
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(data=img_bytes, mime_type="image/png")
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


# ---------------------------------------------------------------------------
# compare_scenes tool
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def compare_scenes(
    ctx: Context,
    prompt: str,
    slot_a: str | None = None,
    slot_b: str | None = None,
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
) -> str:
    """Compare two simulation scenes using Gemini vision.

    Renders both scenes side-by-side and asks Gemini to analyze differences.
    If slot_b is None, compares slot_a at current time vs after re-rendering
    (useful for comparing same scene from different cameras).
    If both are None, requires at least two loaded slots.

    Requires GEMINI_API_KEY environment variable.

    Args:
        prompt: What to compare (e.g. "What changed?", "Which robot is more extended?").
        slot_a: First simulation slot name (default: first available slot).
        slot_b: Second simulation slot name (default: second available slot).
        camera: Named camera for rendering both scenes.
        width: Render width per image.
        height: Render height per image.

    Returns:
        JSON: {"comparison": str, "model": str, "images_sent": int, "slots": [str, str]}
    """
    if not _GENAI_AVAILABLE:
        return json.dumps({
            "error": "google-genai not installed. Run: pip install 'mujoco-mcp-server[vision]'"
        })

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return json.dumps({
            "error": "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key"
        })

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro")

    mgr = ctx.request_context.lifespan_context.sim_manager

    # ------------------------------------------------------------------
    # Resolve slot names
    # ------------------------------------------------------------------
    if slot_a is None and slot_b is None:
        all_slots = list(mgr.snapshot_slots().keys())
        if len(all_slots) < 2:
            return json.dumps({
                "error": (
                    f"compare_scenes needs at least 2 loaded slots, "
                    f"but only {len(all_slots)} found: {all_slots}"
                )
            })
        name_a, name_b = all_slots[0], all_slots[1]
    elif slot_a is not None and slot_b is None:
        return json.dumps({
            "error": "slot_b must be provided when slot_a is specified. "
                     "Pass both slot names, or omit both to auto-select."
        })
    elif slot_a is None and slot_b is not None:
        return json.dumps({
            "error": "slot_a must be provided when slot_b is specified. "
                     "Pass both slot names, or omit both to auto-select."
        })
    else:
        name_a, name_b = slot_a, slot_b  # type: ignore[assignment]

    slot_obj_a = mgr.get(name_a)
    slot_obj_b = mgr.get(name_b)

    # ------------------------------------------------------------------
    # Render each slot
    # ------------------------------------------------------------------
    png_bytes_a = _render_slot_png(mgr, slot_obj_a, camera, width, height)
    png_bytes_b = _render_slot_png(mgr, slot_obj_b, camera, width, height)

    # ------------------------------------------------------------------
    # Build combined system prompt
    # ------------------------------------------------------------------
    prompt_a = _build_system_prompt(slot_obj_a.model, slot_obj_a.data)
    prompt_b = _build_system_prompt(slot_obj_b.model, slot_obj_b.data)

    system_prompt = "\n\n".join([
        f"## Slot A — '{name_a}'",
        prompt_a,
        f"## Slot B — '{name_b}'",
        prompt_b,
        "## Task",
        "You will receive two rendered images (Image 1 = Slot A, Image 2 = Slot B). "
        "Compare them carefully and answer the user's question.",
    ])

    if png_bytes_a is None and png_bytes_b is None:
        system_prompt += (
            "\n\n*Note: Neither scene could be rendered — "
            "comparison is based on metadata only.*"
        )
    elif png_bytes_a is None:
        system_prompt += f"\n\n*Note: Image for Slot A ('{name_a}') unavailable — using metadata only.*"
    elif png_bytes_b is None:
        system_prompt += f"\n\n*Note: Image for Slot B ('{name_b}') unavailable — using metadata only.*"

    # ------------------------------------------------------------------
    # Call Gemini with both images
    # ------------------------------------------------------------------
    try:
        loop = asyncio.get_running_loop()
        analysis = await loop.run_in_executor(
            None,
            lambda: _call_gemini_multi_image(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=prompt,
                images=[png_bytes_a, png_bytes_b],
            ),
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})

    images_sent = sum(1 for b in [png_bytes_a, png_bytes_b] if b is not None)
    return json.dumps(
        {
            "comparison": analysis,
            "model": model,
            "images_sent": images_sent,
            "slots": [name_a, name_b],
        },
        ensure_ascii=False,
    )
