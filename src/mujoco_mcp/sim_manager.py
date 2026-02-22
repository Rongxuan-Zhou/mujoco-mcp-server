"""Multi-slot simulation manager. Thread-safe lifecycle for MjModel/MjData/Renderer."""

import mujoco
import threading
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from .compat import create_renderer, list_named, MUJOCO_VERSION

logger = logging.getLogger(__name__)


@dataclass
class SimSlot:
    name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    renderer: Optional[mujoco.Renderer] = None
    xml_path: Optional[str] = None
    xml_string: Optional[str] = None   # set only when loaded from string; for reload/batch
    trajectory: list = field(default_factory=list)
    recording: bool = False
    passive_viewer: Optional[object] = None  # mujoco.viewer Handle (launch_passive)
    controller: Optional[object] = None       # Phase 4c: RobotController
    sensor_manager: Optional[object] = None   # Phase 4d: SensorManager
    rl_env: Optional[object] = None           # Phase 4f: MuJoCoRLEnvironment


class SimManager:
    def __init__(self, enable_rendering: bool = True,
                 render_width: int = 640, render_height: int = 480):
        self.slots: dict[str, SimSlot] = {}
        self.active_slot: Optional[str] = None
        self.coordinator: Optional[object] = None  # Phase 4e: MultiRobotCoordinator
        self.render_width = render_width
        self.render_height = render_height
        self._rendering_enabled = enable_rendering
        self._rendering_available = False
        self._gl_context: Optional[mujoco.GLContext] = None
        self._lock = threading.Lock()

    # ---- lifecycle ----

    def init_rendering(self):
        if not self._rendering_enabled:
            return
        try:
            self._gl_context = mujoco.GLContext(
                self.render_width, self.render_height)
            self._gl_context.make_current()
            self._rendering_available = True
            logger.info("GL rendering ready")
        except Exception as e:
            self._rendering_available = False
            logger.warning(f"Rendering unavailable: {e}")

    @property
    def can_render(self) -> bool:
        return self._rendering_available

    def cleanup(self):
        with self._lock:
            slots_snapshot = list(self.slots.values())
            self.slots.clear()
            self.active_slot = None
        for s in slots_snapshot:
            if s.renderer:
                try:
                    s.renderer.close()
                except Exception:
                    pass
            if s.passive_viewer:
                try:
                    s.passive_viewer.close()
                except Exception:
                    pass
        if self._gl_context:
            try:
                self._gl_context.free()
            except Exception:
                pass
            self._gl_context = None

    # ---- slot management ----

    def load(self, name: str, *,
             xml_path: str | None = None,
             xml_string: str | None = None) -> dict:
        """Load MJCF into a named slot. Closes existing slot with same name."""
        # Do heavy work (MjModel parse, renderer creation) outside the lock
        if xml_path:
            p = Path(xml_path)
            if not p.exists():
                raise FileNotFoundError(f"MJCF not found: {xml_path}")
            model = mujoco.MjModel.from_xml_path(str(p))
            stored_xml_path = xml_path
            stored_xml_string = None          # M-4: don't read file into memory
        elif xml_string:
            model = mujoco.MjModel.from_xml_string(xml_string)
            stored_xml_path = None
            stored_xml_string = xml_string    # keep for reload/batch
        else:
            raise ValueError("Provide xml_path or xml_string")

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        renderer = None
        if self._rendering_available:
            try:
                renderer = create_renderer(model, self.render_width, self.render_height)
            except Exception as e:
                logger.warning(f"Renderer failed for '{name}': {e}")

        slot = SimSlot(
            name=name, model=model, data=data,
            renderer=renderer,
            xml_path=stored_xml_path,
            xml_string=stored_xml_string,
        )

        with self._lock:
            old = self.slots.pop(name, None)
            self.slots[name] = slot
            self.active_slot = name

        # Close old renderer and viewer outside the lock (may block briefly)
        if old:
            if old.renderer:
                try:
                    old.renderer.close()
                except Exception:
                    pass
            if old.passive_viewer:
                try:
                    old.passive_viewer.close()
                except Exception:
                    pass

        return self._summary(slot)

    def get(self, name: str | None = None) -> SimSlot:
        """Retrieve slot by name (or active slot if name is None). Thread-safe."""
        with self._lock:
            target = name or self.active_slot
            if target is None:
                raise ValueError("No simulation loaded. Call sim_load first.")
            slot = self.slots.get(target)
            if slot is None:
                available = list(self.slots.keys())
                raise ValueError(
                    f"Slot '{target}' not found. Available: {available}")
            return slot  # returning reference is safe; slot fields are not replaced

    def snapshot_slots(self) -> dict[str, SimSlot]:
        """Return a shallow copy of the slots dict. Safe for iteration."""
        with self._lock:
            return dict(self.slots)

    def require_renderer(self, slot: SimSlot) -> mujoco.Renderer:
        """Return renderer or raise a clear error."""
        if slot.renderer is None:
            raise RuntimeError(
                "Rendering unavailable on this node. "
                "Run server_diagnostics for GL backend details.")
        return slot.renderer

    def _summary(self, s: SimSlot) -> dict:
        m = s.model
        obj = mujoco.mjtObj
        return {
            "name": s.name,
            "mujoco_version": str(MUJOCO_VERSION),
            "nq": m.nq, "nv": m.nv, "nu": m.nu,
            "nbody": m.nbody, "ngeom": m.ngeom, "njnt": m.njnt,
            "nsite": m.nsite, "nsensor": m.nsensor, "ncam": m.ncam,
            "timestep": m.opt.timestep,
            "has_renderer": s.renderer is not None,
            "bodies":    list_named(m, obj.mjOBJ_BODY,     m.nbody),
            "joints":    list_named(m, obj.mjOBJ_JOINT,    m.njnt),
            "actuators": list_named(m, obj.mjOBJ_ACTUATOR, m.nu),
            "sensors":   list_named(m, obj.mjOBJ_SENSOR,   m.nsensor),
            "cameras":   list_named(m, obj.mjOBJ_CAMERA,   m.ncam),
        }
