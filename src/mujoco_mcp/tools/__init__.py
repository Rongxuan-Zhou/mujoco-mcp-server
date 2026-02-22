"""Tool registration package. Importing each submodule registers its @mcp.tool() functions."""

import functools
import traceback
import json
import mujoco


def safe_tool(func):
    """Decorator: catch all MuJoCo / IO / value errors and return JSON error objects."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except mujoco.FatalError as e:
            return json.dumps({"error": "mujoco_fatal", "message": str(e)})
        except FileNotFoundError as e:
            return json.dumps({"error": "file_not_found", "message": str(e)})
        except ValueError as e:
            return json.dumps({"error": "invalid_input", "message": str(e)})
        except RuntimeError as e:
            return json.dumps({"error": "runtime", "message": str(e)})
        except Exception as e:
            return json.dumps({
                "error": type(e).__name__,
                "message": str(e),
                "tb": traceback.format_exc()[-400:],
            })
    return wrapper


def _viewer_sync(slot) -> None:
    """Sync passive viewer if one is open for this slot. Silent no-op if none.

    Called automatically at the end of every state-modifying tool so the
    MuJoCo viewer window stays in sync without any extra user action.
    """
    v = getattr(slot, "passive_viewer", None)
    if v is None:
        return
    try:
        if v.is_running():
            v.sync()
        else:
            # Window was closed by the user — detach cleanly
            slot.passive_viewer = None
    except Exception:
        slot.passive_viewer = None
