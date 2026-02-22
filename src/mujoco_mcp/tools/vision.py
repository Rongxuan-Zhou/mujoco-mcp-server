"""Phase 4g: Vision API tool — analyze_scene via Gemini 2.5 Pro."""

import asyncio
import json
import logging
import os
import threading
import time
from io import BytesIO

import mujoco
from PIL import Image
from mcp.server.fastmcp import Context

from .._registry import mcp
from ..compat import resolve_camera, update_scene
from . import safe_tool

logger = logging.getLogger(__name__)

# ── Optional Gemini dependency ────────────────────────────────────────────────

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
    try:
        from google.api_core import exceptions as _gapi_exc
        _RATE_LIMIT_EXC = (_gapi_exc.ResourceExhausted, _gapi_exc.TooManyRequests)
    except ImportError:
        _RATE_LIMIT_EXC = ()
except ImportError:
    _GENAI_AVAILABLE = False
    _RATE_LIMIT_EXC = ()

# ── Error helpers ─────────────────────────────────────────────────────────────

def _make_error(code: str, message: str, **kwargs) -> str:
    """Return a JSON error string with standardised fields."""
    return json.dumps({"error": code, "message": message, **kwargs}, ensure_ascii=False)


# ── Gemini client cache ───────────────────────────────────────────────────────

_gemini_client_cache: dict = {}
_gemini_client_lock = threading.Lock()


def _get_client(api_key: str):
    """Return a cached Gemini client for *api_key*."""
    with _gemini_client_lock:
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
            is_rate_limit = (
                isinstance(e, _RATE_LIMIT_EXC)
                or "429" in str(e)
                or "RESOURCE_EXHAUSTED" in str(e)
            )
            if is_rate_limit and attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise


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
    mime_type: str = "image/png",
) -> str:
    """Send image + prompt to Gemini. Returns response text."""
    client = _get_client(api_key)

    parts = []
    if png_bytes:
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(data=png_bytes, mime_type=mime_type)
            )
        )
    parts.append(genai_types.Part(text=user_prompt))

    response = _call_with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    ))
    text = response.text
    if text is None:
        finish = getattr(response, "finish_reason", "unknown")
        raise ValueError(f"Gemini returned no text (finish_reason={finish})")
    return text


# ---------------------------------------------------------------------------
# Private helpers shared by analyze_scene, compare_scenes, track_object
# ---------------------------------------------------------------------------

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


def _call_gemini_multi_image(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: "list[tuple[bytes, str] | None]",
) -> str:
    """Send multiple images + prompt to Gemini.  Returns response text.

    Each image is preceded by a text label ("Image 1:", "Image 2:", ...).
    None entries are silently skipped (the label is still omitted).
    Each entry is a (bytes, mime_type) tuple or None.
    """
    client = _get_client(api_key)

    parts: list = []
    image_index = 0
    for img_entry in images:
        image_index += 1
        if img_entry is None:
            continue
        img_bytes, img_mime = img_entry
        parts.append(genai_types.Part(text=f"Image {image_index}:"))
        parts.append(
            genai_types.Part(
                inline_data=genai_types.Blob(data=img_bytes, mime_type=img_mime)
            )
        )

    parts.append(genai_types.Part(text=user_prompt))

    response = _call_with_retry(lambda: client.models.generate_content(
        model=model,
        contents=[genai_types.Content(role="user", parts=parts)],
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
        ),
    ))
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
        return _make_error("NO_GENAI", "google-genai not installed. Run: pip install 'mujoco-mcp-server[vision]'")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _make_error("NO_API_KEY", "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key")

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro")

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m, d = slot.model, slot.data

    system_prompt = _build_system_prompt(m, d)

    png_bytes = None
    img_mime_type = "image/png"
    image_sent = False
    try:
        result = _render_slot_image(mgr, slot, camera, width, height)
        if result is not None:
            png_bytes, img_mime_type = result
            image_sent = True
        else:
            # _render_slot_image returned None (renderer raised or image failed)
            system_prompt += "\n\n*Note: No rendered image available — analysis based on metadata only.*"
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
            mime_type=img_mime_type if image_sent else "image/png",
        ))
    except Exception as e:
        return _make_error(
            "QUOTA_EXCEEDED" if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else "GEMINI_ERROR",
            str(e),
        )

    return json.dumps({
        "analysis": analysis,
        "model": model,
        "image_sent": image_sent,
        "sim_time": float(d.time),
    }, ensure_ascii=False)


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
        return _make_error("NO_GENAI", "google-genai not installed. Run: pip install 'mujoco-mcp-server[vision]'")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return _make_error("NO_API_KEY", "GEMINI_API_KEY not set. Export it: export GEMINI_API_KEY=your_key")

    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro")

    mgr = ctx.request_context.lifespan_context.sim_manager

    # ------------------------------------------------------------------
    # Resolve slot names
    # ------------------------------------------------------------------
    if slot_a is None and slot_b is None:
        all_slots = list(mgr.snapshot_slots().keys())
        if len(all_slots) < 2:
            return _make_error(
                "INVALID_ARGS",
                f"compare_scenes needs at least 2 loaded slots, but only {len(all_slots)} found: {all_slots}",
            )
        name_a, name_b = all_slots[0], all_slots[1]
    elif slot_a is not None and slot_b is None:
        return _make_error(
            "INVALID_ARGS",
            "slot_b must be provided when slot_a is specified. Pass both slot names, or omit both to auto-select.",
        )
    elif slot_a is None and slot_b is not None:
        return _make_error(
            "INVALID_ARGS",
            "slot_a must be provided when slot_b is specified. Pass both slot names, or omit both to auto-select.",
        )
    else:
        name_a, name_b = slot_a, slot_b  # type: ignore[assignment]

    slot_obj_a = mgr.get(name_a)
    slot_obj_b = mgr.get(name_b)

    # ------------------------------------------------------------------
    # Render each slot — returns (bytes, mime_type) tuple or None
    # ------------------------------------------------------------------
    result_a = _render_slot_image(mgr, slot_obj_a, camera, width, height)
    result_b = _render_slot_image(mgr, slot_obj_b, camera, width, height)
    img_bytes_a, mime_a = result_a if result_a else (None, "image/png")
    img_bytes_b, mime_b = result_b if result_b else (None, "image/png")

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

    if img_bytes_a is None and img_bytes_b is None:
        system_prompt += (
            "\n\n*Note: Neither scene could be rendered — "
            "comparison is based on metadata only.*"
        )
    elif img_bytes_a is None:
        system_prompt += f"\n\n*Note: Image for Slot A ('{name_a}') unavailable — using metadata only.*"
    elif img_bytes_b is None:
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
                images=[
                    (img_bytes_a, mime_a) if img_bytes_a else None,
                    (img_bytes_b, mime_b) if img_bytes_b else None,
                ],
            ),
        )
    except Exception as e:
        return _make_error(
            "QUOTA_EXCEEDED" if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else "GEMINI_ERROR",
            str(e),
        )

    images_sent = sum(1 for b in [img_bytes_a, img_bytes_b] if b is not None)
    return json.dumps(
        {
            "comparison": analysis,
            "model": model,
            "images_sent": images_sent,
            "slots": [name_a, name_b],
        },
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# track_object tool
# ---------------------------------------------------------------------------

@mcp.tool()
@safe_tool
async def track_object(
    ctx: Context,
    body_name: str,
    n_steps: int = 200,
    capture_every: int = 50,
    prompt: str | None = None,
    sim_name: str | None = None,
    camera: str | None = None,
    width: int = 320,
    height: int = 240,
) -> str:
    """Track a body's trajectory and analyze its motion using Gemini vision.

    Steps the simulation forward, records the body's world position at each step,
    and captures rendered keyframes at intervals. Optionally sends keyframes to
    Gemini for motion analysis.

    Args:
        body_name: Name of the body to track (use scene_map to find names).
        n_steps: Number of simulation steps to run (max 5000).
        capture_every: Render a keyframe every N steps (0 = no visual capture).
        prompt: Optional question for Gemini vision analysis of the trajectory.
                If None or no API key, returns trajectory data only.
        sim_name: Simulation slot name.
        camera: Named camera for keyframes.
        width: Keyframe render width (smaller = faster).
        height: Keyframe render height.

    Returns:
        JSON: {
            "body": str,
            "n_steps": int,
            "trajectory": [{"t": float, "pos": [x, y, z]}],
            "keyframes_captured": int,
            "analysis": str | null,
            "model": str | null,
        }
    """
    # ------------------------------------------------------------------
    # Clamp n_steps
    # ------------------------------------------------------------------
    n_steps = min(n_steps, 5000)

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    m = slot.model
    d = slot.data

    # ------------------------------------------------------------------
    # Resolve body_id
    # ------------------------------------------------------------------
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        available = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"<body:{i}>"
            for i in range(m.nbody)
        ]
        return _make_error("BODY_NOT_FOUND", f"Body '{body_name}' not found. Available bodies: {available}")

    # ------------------------------------------------------------------
    # Step simulation, record trajectory, capture keyframes
    # ------------------------------------------------------------------
    full_trajectory: list[dict] = []
    keyframe_images: list[tuple[bytes, str] | None] = []

    for step in range(n_steps):
        mujoco.mj_step(m, d)
        full_trajectory.append({
            "t": float(d.time),
            "pos": d.xpos[body_id].tolist(),
        })
        if capture_every > 0 and step % capture_every == 0:
            img_result = _render_slot_image(mgr, slot, camera, width, height)
            keyframe_images.append(img_result)

    # ------------------------------------------------------------------
    # Downsample trajectory if too large (keep at most 1000 points)
    # ------------------------------------------------------------------
    if len(full_trajectory) > 1000:
        stride = len(full_trajectory) // 1000 + 1
        trajectory = full_trajectory[::stride]
    else:
        trajectory = full_trajectory

    keyframes_captured = sum(1 for img in keyframe_images if img is not None)

    # ------------------------------------------------------------------
    # Optional Gemini analysis
    # ------------------------------------------------------------------
    analysis: str | None = None
    gemini_model: str | None = None

    if prompt is not None and keyframe_images:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key and _GENAI_AVAILABLE:
            gemini_model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-pro")

            # Build a concise trajectory summary for the system prompt
            start_pos = full_trajectory[0]["pos"]
            end_pos = full_trajectory[-1]["pos"]
            import math
            displacement = math.sqrt(
                sum((end_pos[i] - start_pos[i]) ** 2 for i in range(3))
            )
            total_time = full_trajectory[-1]["t"] - full_trajectory[0]["t"]

            system_prompt = "\n".join([
                "You are an expert robotics simulation analyst examining keyframes "
                "of a MuJoCo physics simulation.",
                "",
                f"## Tracked Body: '{body_name}'",
                f"Steps run: {n_steps} | Total simulated time: {total_time:.4f}s",
                f"Start position: ({start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}) m",
                f"End position:   ({end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}) m",
                f"Displacement:   {displacement:.4f} m",
                "",
                f"Keyframes sent: {keyframes_captured} "
                f"(one every {capture_every} steps)",
                "",
                "## Response Format",
                "Describe the motion you observe across the keyframes. "
                "Reference positions/orientations where relevant. "
                "Answer the user's question precisely using metric units.",
            ])

            try:
                loop = asyncio.get_running_loop()
                analysis = await loop.run_in_executor(
                    None,
                    lambda: _call_gemini_multi_image(
                        api_key=api_key,
                        model=gemini_model,
                        system_prompt=system_prompt,
                        user_prompt=prompt,
                        images=keyframe_images,
                    ),
                )
            except Exception as e:
                analysis = f"Gemini error: {type(e).__name__}: {e}"

    return json.dumps(
        {
            "body": body_name,
            "n_steps": n_steps,
            "trajectory": trajectory,
            "keyframes_captured": keyframes_captured,
            "analysis": analysis,
            "model": gemini_model,
        },
        ensure_ascii=False,
    )
