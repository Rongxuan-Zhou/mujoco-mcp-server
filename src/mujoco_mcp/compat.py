"""Runtime feature detection for MuJoCo version compatibility.
Import once, use everywhere. Must be imported AFTER gl_setup sets MUJOCO_GL."""

import mujoco
import numpy as np
from packaging.version import Version
from typing import Optional
import logging

logger = logging.getLogger(__name__)

MUJOCO_VERSION = Version(mujoco.__version__.split(".post")[0])

# Feature flags (evaluated once at import)
# Note: Renderer is NEVER used as a context manager — this is a hard rule, not a flag.
HAS_CAMERA_STRING_NAME = MUJOCO_VERSION >= Version("3.0.0")
HAS_SCENE_OPTION_KWARG = MUJOCO_VERSION >= Version("2.3.3")
# MuJoCo 3.x introduced contact.geom[] array; 2.x uses contact.geom1/geom2
HAS_GEOM_ARRAY = MUJOCO_VERSION >= Version("3.0.0")

logger.info(
    f"MuJoCo {MUJOCO_VERSION} | "
    f"cam_str={HAS_CAMERA_STRING_NAME} scene_opt={HAS_SCENE_OPTION_KWARG} "
    f"geom_arr={HAS_GEOM_ARRAY}"
)


def get_version_info() -> dict:
    return {
        "mujoco_version": str(MUJOCO_VERSION),
        "camera_string_name": HAS_CAMERA_STRING_NAME,
        "scene_option_kwarg": HAS_SCENE_OPTION_KWARG,
        "geom_array": HAS_GEOM_ARRAY,
    }


# --- Contact geometry access (geom1/geom2 deprecated in 3.x) ---

def contact_geoms(contact) -> tuple[int, int]:
    """Return (geom1_id, geom2_id) compatible across MuJoCo 2.x and 3.x.

    MuJoCo 3.x deprecates contact.geom1/geom2 in favour of contact.geom[0/1].
    This helper picks the right accessor based on the installed version.
    """
    if HAS_GEOM_ARRAY:
        return int(contact.geom[0]), int(contact.geom[1])
    return int(contact.geom1), int(contact.geom2)


# --- Energy computation helpers ---

def ensure_energy(model: mujoco.MjModel) -> bool:
    """Enable mjENBL_ENERGY on model if not already set. Returns previous state."""
    flag = int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    was_enabled = bool(model.opt.enableflags & flag)
    if not was_enabled:
        model.opt.enableflags |= flag
    return was_enabled


def restore_energy(model: mujoco.MjModel, was_enabled: bool) -> None:
    """Restore mjENBL_ENERGY to its original state."""
    if not was_enabled:
        model.opt.enableflags &= ~int(mujoco.mjtEnableBit.mjENBL_ENERGY)


# --- Renderer lifecycle (NEVER use `with`) ---

def create_renderer(model: mujoco.MjModel,
                    width: int = 640, height: int = 480) -> mujoco.Renderer:
    """Version-safe renderer creation. Caller must call .close() explicitly."""
    return mujoco.Renderer(model, height=height, width=width)


# --- Camera resolution (always int ID) ---

def resolve_camera(model: mujoco.MjModel, camera: Optional[str]) -> int:
    """Name → int ID. Returns -1 for free camera."""
    if camera is None:
        return -1
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        available = list_named(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        raise ValueError(f"Camera '{camera}' not found. Available: {available}")
    return cam_id


# --- Scene update (conditional kwargs) ---

def update_scene(renderer: mujoco.Renderer,
                 data: mujoco.MjData,
                 camera_id: int = -1,
                 scene_option: Optional[mujoco.MjvOption] = None):
    kw = {}
    if camera_id >= 0:
        kw["camera"] = camera_id
    if scene_option is not None and HAS_SCENE_OPTION_KWARG:
        kw["scene_option"] = scene_option
    renderer.update_scene(data, **kw)


# --- Name listing ---

def list_named(model: mujoco.MjModel, obj_type: int, count: int) -> list[str]:
    out = []
    for i in range(count):
        n = mujoco.mj_id2name(model, obj_type, i)
        if n:
            out.append(n)
    return out


def resolve_name(model: mujoco.MjModel, obj_type: int, name: str,
                 type_label: str) -> int:
    """Name → int ID with descriptive error on miss."""
    oid = mujoco.mj_name2id(model, obj_type, name)
    if oid < 0:
        count_map = {
            mujoco.mjtObj.mjOBJ_BODY:     model.nbody,
            mujoco.mjtObj.mjOBJ_GEOM:     model.ngeom,
            mujoco.mjtObj.mjOBJ_JOINT:    model.njnt,
            mujoco.mjtObj.mjOBJ_SITE:     model.nsite,
            mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
            mujoco.mjtObj.mjOBJ_SENSOR:   model.nsensor,
            mujoco.mjtObj.mjOBJ_CAMERA:   model.ncam,
        }
        available = list_named(model, obj_type, count_map.get(obj_type, 0))
        raise ValueError(f"{type_label} '{name}' not found. Available: {available}")
    return oid
