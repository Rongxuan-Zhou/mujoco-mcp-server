"""GL backend detection for Linux headless rendering.
MUST be called BEFORE import mujoco anywhere in the process.
Sets MUJOCO_GL environment variable so MuJoCo reads it at first import."""

import os
import sys
import subprocess
import logging

logger = logging.getLogger(__name__)


def detect_and_set_gl_backend() -> str:
    """Probe EGL then OSMesa in subprocesses. Set env var for main process.

    Returns the backend name: 'egl', 'osmesa', or 'none'.
    If MUJOCO_MCP_NO_RENDER=1, skips probing and returns 'none'.
    If MUJOCO_GL is already set, respects it.
    """
    if os.environ.get("MUJOCO_MCP_NO_RENDER", "0") == "1":
        logger.info("MUJOCO_MCP_NO_RENDER=1: skipping GL init")
        return "none"

    if "MUJOCO_GL" in os.environ:
        backend = os.environ["MUJOCO_GL"]
        logger.info(f"GL backend from env: {backend}")
        return backend

    for backend in ("egl", "osmesa"):
        if _probe_backend(backend):
            os.environ["MUJOCO_GL"] = backend
            logger.info(f"Auto-detected GL backend: {backend}")
            return backend

    logger.warning("No GL backend found. Rendering will be disabled.")
    return "none"


def _probe_backend(backend: str) -> bool:
    code = (
        f"import os; os.environ['MUJOCO_GL']='{backend}'; "
        "import mujoco; "
        "ctx=mujoco.GLContext(64,64); ctx.make_current(); ctx.free(); "
        "print('OK')"
    )
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip() == "OK"
    except Exception:
        return False


def get_gl_diagnostics() -> dict:
    return {
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "unset"),
        "egl_probe": _probe_backend("egl"),
        "osmesa_probe": _probe_backend("osmesa"),
        "nvidia_gpu": _nvidia_gpu(),
    }


def _nvidia_gpu() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "not found"
    except FileNotFoundError:
        return "nvidia-smi unavailable"
