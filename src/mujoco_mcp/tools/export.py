"""Export tools (tools 25–26): export_csv, plot_data."""

import asyncio
import base64
import io
import json
import os

from mcp.server.fastmcp import Context

from .._registry import mcp
from . import safe_tool


@mcp.tool()
@safe_tool
async def export_csv(
    ctx: Context,
    output_path: str,
    sim_name: str | None = None,
    include_energy: bool = False,
) -> str:
    """Export recorded trajectory to a CSV file.

    Columns: t, qpos_0 … qpos_{nq-1}, qvel_0 … qvel_{nv-1}
    If include_energy=True and the slot has energy data, adds kinetic/potential columns.

    Requires sim_record to have been started before sim_step calls.
    Returns: path written, row count, column names.
    """
    import csv

    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)

    traj = slot.trajectory
    if not traj:
        return json.dumps({"error": "No trajectory recorded. Use sim_record(action='start') then sim_step."})

    # Infer nq / nv from first frame
    first = traj[0]
    qpos_len = len(first.get("qpos", []))
    qvel_len = len(first.get("qvel", []))

    # Detect energy columns in trajectory frames
    has_energy = include_energy and ("E_pot" in first or "E_kin" in first)

    fieldnames = ["t"]
    fieldnames += [f"qpos_{i}" for i in range(qpos_len)]
    fieldnames += [f"qvel_{i}" for i in range(qvel_len)]
    if has_energy:
        fieldnames += ["E_pot", "E_kin"]

    rows_written = 0
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for frame in traj:
                row: dict = {"t": frame.get("t", 0.0)}
                qpos = frame.get("qpos", [])
                for i, v in enumerate(qpos):
                    row[f"qpos_{i}"] = v
                qvel = frame.get("qvel", [])
                for i, v in enumerate(qvel):
                    row[f"qvel_{i}"] = v
                if has_energy:
                    row["E_pot"] = frame.get("E_pot", "")
                    row["E_kin"] = frame.get("E_kin", "")
                writer.writerow(row)
                rows_written += 1
    except OSError as e:
        return json.dumps({"error": f"Cannot write CSV: {e}"})

    result: dict = {
        "ok":      True,
        "path":    output_path,
        "rows":    rows_written,
        "columns": fieldnames,
    }
    if include_energy and not has_energy:
        result["warning"] = (
            "include_energy=True but no energy data found in trajectory frames. "
            "Record with run_and_analyze(track=['energy']) to capture E_pot/E_kin."
        )
    return json.dumps(result, indent=2)


@mcp.tool()
@safe_tool
async def plot_data(
    ctx: Context,
    csv_path: str,
    x_col: str = "t",
    y_cols: list[str] | None = None,
    title: str = "",
    output_path: str | None = None,
) -> list:
    """Read a CSV file and return a matplotlib plot as an inline PNG image.

    Args:
        csv_path:    Path to CSV file (from export_csv or run_sweep)
        x_col:       Column name for x-axis (default "t")
        y_cols:      Column names to plot on y-axis; if None, plots all numeric columns except x_col
        title:       Plot title
        output_path: Optional path to save PNG; if omitted, returned inline only

    Returns inline PNG image (and JSON summary).
    """
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")   # headless — no display needed
    import matplotlib.pyplot as plt
    from mcp.types import TextContent, ImageContent

    if not os.path.exists(csv_path):
        return [TextContent(type="text",
                            text=json.dumps({"error": f"CSV not found: {csv_path}"}))]

    loop = asyncio.get_running_loop()
    df = await loop.run_in_executor(None, pd.read_csv, csv_path)

    if x_col not in df.columns:
        return [TextContent(type="text",
                            text=json.dumps({"error": f"x_col '{x_col}' not in CSV columns: {list(df.columns)}"}))]

    if y_cols is None:
        y_cols = [c for c in df.columns if c != x_col and pd.api.types.is_numeric_dtype(df[c])]

    missing = [c for c in y_cols if c not in df.columns]
    if missing:
        return [TextContent(type="text",
                            text=json.dumps({"error": f"y_cols not found: {missing}"}))]

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in y_cols:
        ax.plot(df[x_col], df[col], label=col, linewidth=1.2)

    ax.set_xlabel(x_col)
    ax.set_ylabel("value")
    ax.set_title(title or f"{', '.join(y_cols)} vs {x_col}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()

    if output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(b64))
        except OSError:
            pass  # non-fatal; image still returned inline

    summary = {
        "ok":      True,
        "csv":     csv_path,
        "x_col":   x_col,
        "y_cols":  y_cols,
        "n_rows":  len(df),
        "saved":   output_path,
    }
    return [
        TextContent(type="text", text=json.dumps(summary, indent=2)),
        ImageContent(type="image", data=b64, mimeType="image/png"),
    ]
