"""Media export tools — export_video (MP4/GIF) from recorded trajectory."""
from __future__ import annotations

import asyncio
import json
import os

import mujoco
from PIL import Image as PILImage

# Async yield interval for the render loop (frames). Rendering is more
# CPU-intensive per iteration than sim steps, so a shorter interval is used
# compared to ASYNC_YIELD_INTERVAL (1_000) in constants.py.
_RENDER_YIELD_INTERVAL = 50

from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


def _encode_frames(frames: list, output_path: str, fps: int, fmt: str) -> None:
    """Encode a list of uint8 RGB arrays to GIF or MP4.

    Args:
        frames: list of numpy uint8 arrays [H, W, 3].
        output_path: destination file path.
        fps: frames per second.
        fmt: "gif" or "mp4".
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if fmt == "gif":
        imgs = [PILImage.fromarray(f) for f in frames]
        imgs[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=max(1, 1000 // fps),
            loop=0,
            optimize=False,
        )
    else:  # mp4
        try:
            import imageio as _imageio  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "MP4 export requires imageio[ffmpeg]. "
                "Install with: pip install 'mujoco-mcp[media]'"
            ) from exc
        with _imageio.get_writer(output_path, fps=fps, macro_block_size=None) as writer:
            for f in frames:
                writer.append_data(f)


def _export_video_impl(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    trajectory: list,
    output_path: str,
    fps: int = 30,
    fmt: str = "mp4",
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
) -> str:
    """Render trajectory to video file (MP4 or GIF).

    Args:
        model: MjModel instance.
        data: MjData instance (state is saved/restored).
        trajectory: list of frames [{t, qpos, qvel, ...}].
        output_path: destination file path.
        fps: frames per second.
        fmt: "mp4" or "gif".
        camera: camera name (None = free camera).
        width: render width in pixels.
        height: render height in pixels.

    Returns:
        JSON string {"ok", "path", "frames", "duration_s", "format"}.
    """
    if not trajectory:
        raise ValueError("trajectory is empty; call sim_record + sim_step first")
    if fmt not in ("mp4", "gif"):
        raise ValueError(f"fmt must be 'mp4' or 'gif', got {fmt!r}")

    # Validate camera before starting renderer
    if camera is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise ValueError(f"Camera {camera!r} not found in model")

    # Use a scratch MjData copy for rendering so the caller's data is never
    # mutated.
    render_data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        frames: list = []
        for frame in trajectory:
            render_data.qpos[:] = frame["qpos"]
            render_data.qvel[:] = frame["qvel"]
            render_data.time = frame.get("t", 0.0)
            mujoco.mj_forward(model, render_data)
            if camera is not None:
                renderer.update_scene(render_data, camera=camera)
            else:
                renderer.update_scene(render_data)
            frames.append(renderer.render().copy())

        _encode_frames(frames, output_path, fps, fmt)

        return json.dumps({
            "ok": True,
            "path": output_path,
            "frames": len(frames),
            "duration_s": round(len(frames) / fps, 3),
            "format": fmt,
        })
    finally:
        renderer.close()


@mcp.tool()
@safe_tool
async def export_video(
    ctx: Context,
    output_path: str,
    fps: int = 30,
    fmt: str = "mp4",
    camera: str | None = None,
    width: int = 640,
    height: int = 480,
    sim_name: str | None = None,
) -> str:
    """Export recorded trajectory as MP4 or GIF video.

    Args:
        output_path: destination file path (.mp4 or .gif).
        fps: frames per second (default 30).
        fmt: "mp4" (requires imageio[ffmpeg]) or "gif" (Pillow, no extra deps).
        camera: camera name for rendering (None = free camera).
        width: render width in pixels (default 640).
        height: render height in pixels (default 480).
        sim_name: simulation slot name (default slot if None).

    Note:
        MP4 requires: pip install 'mujoco-mcp[media]'
        GIF uses Pillow (already installed), no extra deps needed.
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    model = slot.model
    trajectory = slot.trajectory

    if not trajectory:
        raise ValueError("trajectory is empty; call sim_record + sim_step first")
    if fmt not in ("mp4", "gif"):
        raise ValueError(f"fmt must be 'mp4' or 'gif', got {fmt!r}")
    if camera is not None:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if cam_id < 0:
            raise ValueError(f"Camera {camera!r} not found in model")

    # Rendering loop is inline here (not delegated to _export_video_impl) so
    # that we can yield the event loop every _RENDER_YIELD_INTERVAL frames.
    # _export_video_impl remains the sync, test-friendly entry point.
    render_data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames: list = []
    try:
        for i, frame in enumerate(trajectory):
            if i % _RENDER_YIELD_INTERVAL == 0:
                await asyncio.sleep(0)
            render_data.qpos[:] = frame["qpos"]
            render_data.qvel[:] = frame["qvel"]
            render_data.time = frame.get("t", 0.0)
            mujoco.mj_forward(model, render_data)
            if camera is not None:
                renderer.update_scene(render_data, camera=camera)
            else:
                renderer.update_scene(render_data)
            frames.append(renderer.render().copy())
    finally:
        renderer.close()

    _encode_frames(frames, output_path, fps, fmt)
    return json.dumps({
        "ok": True,
        "path": output_path,
        "frames": len(frames),
        "duration_s": round(len(frames) / fps, 3),
        "format": fmt,
    })
