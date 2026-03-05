"""pytest configuration: set GL backend before any test module imports mujoco."""
import os

# Must be set before any 'import mujoco' in test files.
# OSMesa provides software rendering without a display server.
os.environ.setdefault("MUJOCO_GL", "osmesa")
