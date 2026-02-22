"""Phase 4b: MuJoCo Menagerie model loader MCP tools.

Enables loading 39+ robot models by name without managing XML paths manually.
Models are downloaded from google-deepmind/mujoco_menagerie and cached locally.
"""

import json
from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool
from ..menagerie_loader import MenagerieLoader

_loader = MenagerieLoader()  # Singleton with local file cache


@mcp.tool()
@safe_tool
async def list_menagerie_models(
    ctx: Context,
    category: str | None = None,
) -> str:
    """List available MuJoCo Menagerie robot models by category.

    Args:
        category: Filter by category:
                  "arms"|"quadrupeds"|"humanoids"|"grippers"|
                  "mobile_manipulators"|"drones". None = all categories.

    Returns:
        JSON: {"categories": {category: [model_name, ...]}} or {category: [...]}
    """
    models = _loader.get_available_models()
    if category:
        if category not in models:
            raise ValueError(f"Unknown category {category!r}. Available: {list(models)}")
        return json.dumps({category: models[category]}, indent=2)
    return json.dumps({"categories": models}, indent=2)


@mcp.tool()
@safe_tool
async def validate_menagerie_model(ctx: Context, model_name: str) -> str:
    """Download and validate a Menagerie model. Returns body/joint/actuator counts.

    Args:
        model_name: Menagerie model name, e.g. "franka_emika_panda", "unitree_go2".

    Returns:
        JSON: {"valid": bool, "n_bodies": int, "n_joints": int, "n_actuators": int,
               "xml_size": int}
    """
    result = _loader.validate_model(model_name)
    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def load_menagerie_model(
    ctx: Context,
    model_name: str,
    sim_name: str = "default",
    scene_name: str | None = None,
) -> str:
    """Download, resolve XML includes, and load a Menagerie model into a sim slot.

    Models are cached in /tmp/mujoco_menagerie/ after the first download.

    Args:
        model_name: Menagerie model name, e.g. "franka_emika_panda", "unitree_go2".
        sim_name: Target simulation slot (created if not exists).
        scene_name: Optional label for the generated scene XML.

    Returns:
        JSON: sim_load summary (nq, nv, nu, nbody, bodies, joints, actuators, ...)
    """
    sm = ctx.request_context.lifespan_context.sim_manager
    xml = _loader.create_scene_xml(model_name, scene_name)
    result = sm.load(sim_name, xml_string=xml)
    result["menagerie_model"] = model_name
    return json.dumps(result, indent=2)
