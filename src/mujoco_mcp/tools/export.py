"""Export tools (tools 25–26): export_csv, plot_data."""

import asyncio
import base64
import csv
import io
import json
import os
import sys

import mujoco
import numpy as np
from mcp.server.fastmcp import Context
from mcp.types import ImageContent, TextContent

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


# ─── export_state_log ─────────────────────────────────────────────────────────

def _export_state_log_impl(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    trajectory: list,
    output_path: str,
    include: list | None = None,
) -> str:
    """Export full-state CSV with selectable columns.

    Args:
        model: MjModel instance.
        data: MjData instance (state is saved/restored in finally block).
        trajectory: list of frames from slot.trajectory.
        output_path: destination CSV path.
        include: list of tags — "qpos", "qvel", "ctrl",
            "body_xpos:<name>", "body_xquat:<name>", "sensors", "contacts".
    """
    if include is None:
        include = ["qpos", "qvel"]
    if not trajectory:
        raise ValueError("trajectory is empty; call sim_record + sim_step first")

    need_forward = any(
        tag.startswith("body_xpos:") or tag.startswith("body_xquat:")
        or tag in ("sensors", "contacts")
        for tag in include
    )

    header = ["t"]

    if "qpos" in include:
        header += [f"qpos_{i}" for i in range(model.nq)]
    if "qvel" in include:
        header += [f"qvel_{i}" for i in range(model.nv)]
    if "ctrl" in include:
        header += [f"ctrl_{i}" for i in range(model.nu)]

    body_xpos_bodies: list = []
    body_xquat_bodies: list = []
    for tag in include:
        if tag.startswith("body_xpos:"):
            name = tag[len("body_xpos:"):]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body {name!r} not found in model")
            body_xpos_bodies.append((name, bid))
            header += [f"{name}_x", f"{name}_y", f"{name}_z"]
        elif tag.startswith("body_xquat:"):
            name = tag[len("body_xquat:"):]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body {name!r} not found in model")
            body_xquat_bodies.append((name, bid))
            header += [f"{name}_qw", f"{name}_qx", f"{name}_qy", f"{name}_qz"]

    # Build sensor column mapping after an initial mj_forward
    sensor_col_map: list = []
    if "sensors" in include and model.nsensor > 0:
        mujoco.mj_forward(model, data)
        n_sensordata = len(data.sensordata)
        for i in range(model.nsensor):
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) or f"sensor_{i}"
            adr = int(model.sensor_adr[i])
            next_adr = int(model.sensor_adr[i + 1]) if i + 1 < model.nsensor else n_sensordata
            dim = next_adr - adr
            for d in range(dim):
                col = sname if dim == 1 else f"{sname}_{d}"
                sensor_col_map.append((col, adr + d))
        header += [col for col, _ in sensor_col_map]

    if "contacts" in include:
        header += ["contact_count", "max_contact_force"]

    # Save state
    qpos_orig = data.qpos.copy()
    qvel_orig = data.qvel.copy()
    time_orig = data.time

    warnings_list: list = []
    ctrl_warn_issued = False

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()

            for frame in trajectory:
                row: dict = {"t": frame["t"]}

                if "qpos" in include:
                    for i, v in enumerate(frame["qpos"]):
                        row[f"qpos_{i}"] = v
                if "qvel" in include:
                    for i, v in enumerate(frame["qvel"]):
                        row[f"qvel_{i}"] = v
                if "ctrl" in include:
                    if "ctrl" in frame:
                        for i, v in enumerate(frame["ctrl"]):
                            row[f"ctrl_{i}"] = v
                    else:
                        if not ctrl_warn_issued:
                            warnings_list.append(
                                "ctrl not found in trajectory frames "
                                "(re-record after sim_step ctrl patch)"
                            )
                            ctrl_warn_issued = True
                        for i in range(model.nu):
                            row[f"ctrl_{i}"] = ""

                if need_forward:
                    data.qpos[:] = frame["qpos"]
                    data.qvel[:] = frame["qvel"]
                    data.time = frame.get("t", 0.0)
                    mujoco.mj_forward(model, data)

                    for name, bid in body_xpos_bodies:
                        row[f"{name}_x"] = float(data.xpos[bid, 0])
                        row[f"{name}_y"] = float(data.xpos[bid, 1])
                        row[f"{name}_z"] = float(data.xpos[bid, 2])

                    for name, bid in body_xquat_bodies:
                        row[f"{name}_qw"] = float(data.xquat[bid, 0])
                        row[f"{name}_qx"] = float(data.xquat[bid, 1])
                        row[f"{name}_qy"] = float(data.xquat[bid, 2])
                        row[f"{name}_qz"] = float(data.xquat[bid, 3])

                    for col, idx in sensor_col_map:
                        row[col] = float(data.sensordata[idx])

                    if "contacts" in include:
                        row["contact_count"] = data.ncon
                        if data.ncon > 0:
                            max_f = 0.0
                            f_buf = np.zeros(6)
                            for j in range(data.ncon):
                                mujoco.mj_contactForce(model, data, j, f_buf)
                                max_f = max(max_f, float(np.linalg.norm(f_buf[:3])))
                            row["max_contact_force"] = max_f
                        else:
                            row["max_contact_force"] = 0.0

                writer.writerow(row)
    finally:
        data.qpos[:] = qpos_orig
        data.qvel[:] = qvel_orig
        data.time = time_orig
        mujoco.mj_forward(model, data)

    result: dict = {
        "ok": True,
        "path": output_path,
        "rows": len(trajectory),
        "columns": header,
    }
    if warnings_list:
        result["warning"] = "; ".join(warnings_list)
    return json.dumps(result)


# ─── plot_trajectory ──────────────────────────────────────────────────────────

def _plot_trajectory_impl(
    csv_path: str,
    plot_type: str,
    dof: int = 0,
    body: str | None = None,
    output_path: str | None = None,
    title: str = "",
) -> list:
    """Plot phase portrait or 3D body trajectory from CSV.

    Args:
        csv_path: path to CSV from export_csv or export_state_log.
        plot_type: "phase" (qpos vs qvel) or "path3d" (body 3D path).
        dof: DOF index for phase mode.
        body: body name for path3d mode.
        output_path: if given, save PNG; else return inline.
        title: plot title.
    """
    if plot_type not in ("phase", "path3d"):
        raise ValueError(f"plot_type must be 'phase' or 'path3d', got {plot_type!r}")

    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)

    if plot_type == "phase":
        qpos_col = f"qpos_{dof}"
        qvel_col = f"qvel_{dof}"
        for col in (qpos_col, qvel_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {csv_path}; "
                    "run export_state_log with 'qpos' and 'qvel'"
                )

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(df[qpos_col], df[qvel_col], alpha=0.7, linewidth=1.2)
        ax.scatter(
            [df[qpos_col].iloc[0]], [df[qvel_col].iloc[0]],
            color="green", s=80, zorder=5, label="start",
        )
        ax.scatter(
            [df[qpos_col].iloc[-1]], [df[qvel_col].iloc[-1]],
            color="red", s=80, zorder=5, label="end",
        )
        ax.set_xlabel(f"{qpos_col} [rad/m]")
        ax.set_ylabel(f"{qvel_col} [rad·s⁻¹ / m·s⁻¹]")
        ax.set_title(title or f"Phase portrait DOF {dof}")
        ax.legend()
        summary = f"Phase portrait DOF={dof}, {len(df)} frames"

    else:  # path3d
        # On systems where a system mpl_toolkits shadows the venv version,
        # locate the correct mpl_toolkits next to the active matplotlib package
        # and prepend it to the namespace-package search path before importing.
        import mpl_toolkits as _mpl_toolkits
        from matplotlib.projections import get_projection_names, register_projection
        _mpl_sibling = os.path.normpath(
            os.path.join(os.path.dirname(matplotlib.__file__), "..", "mpl_toolkits")
        )
        if os.path.isdir(_mpl_sibling) and _mpl_sibling not in _mpl_toolkits.__path__:
            _mpl_toolkits.__path__.insert(0, _mpl_sibling)
        if "3d" not in get_projection_names():
            from mpl_toolkits.mplot3d import Axes3D as _Axes3D
            register_projection(_Axes3D)
        if body is None:
            raise ValueError("body must be specified for plot_type='path3d'")
        x_col, y_col, z_col = f"{body}_x", f"{body}_y", f"{body}_z"
        for col in (x_col, y_col, z_col):
            if col not in df.columns:
                raise ValueError(
                    f"Column {col!r} not found in {csv_path}; "
                    f"run export_state_log with 'body_xpos:{body}'"
                )

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        t_norm = np.linspace(0, 1, len(df))
        cmap = plt.cm.viridis

        for i in range(len(df) - 1):
            ax.plot(
                df[x_col].iloc[i : i + 2].tolist(),
                df[y_col].iloc[i : i + 2].tolist(),
                df[z_col].iloc[i : i + 2].tolist(),
                color=cmap(t_norm[i]),
                linewidth=1.5,
            )
        ax.scatter(
            [df[x_col].iloc[0]], [df[y_col].iloc[0]], [df[z_col].iloc[0]],
            color="green", s=80, zorder=5, label="start",
        )
        ax.scatter(
            [df[x_col].iloc[-1]], [df[y_col].iloc[-1]], [df[z_col].iloc[-1]],
            color="red", s=80, zorder=5, label="end",
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title(title or f"3D path: {body}")
        ax.legend()
        summary = (
            f"3D path {body}, {len(df)} frames, "
            f"x={df[x_col].min():.3f}..{df[x_col].max():.3f}"
        )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buf.seek(0)
    png_b64 = base64.b64encode(buf.read()).decode()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as fout:
            fout.write(base64.b64decode(png_b64))

    return [
        ImageContent(type="image", data=png_b64, mimeType="image/png"),
        TextContent(type="text", text=summary),
    ]


@mcp.tool()
@safe_tool
async def export_state_log(
    ctx: Context,
    output_path: str,
    include: list | None = None,
    sim_name: str | None = None,
) -> str:
    """Export full-state CSV with selectable columns from recorded trajectory.

    Args:
        output_path: destination CSV path.
        include: list of tags. Supported: "qpos", "qvel", "ctrl",
            "body_xpos:<name>", "body_xquat:<name>", "sensors", "contacts".
            Defaults to ["qpos", "qvel"].
        sim_name: simulation slot name (default slot if None).
    """
    mgr = ctx.request_context.lifespan_context.sim_manager
    slot = mgr.get(sim_name)
    if include is None:
        include = ["qpos", "qvel"]
    await asyncio.sleep(0)
    return _export_state_log_impl(slot.model, slot.data, slot.trajectory,
                                   output_path, include)


@mcp.tool()
@safe_tool
async def plot_trajectory(
    ctx: Context,
    csv_path: str,
    plot_type: str,
    dof: int = 0,
    body: str | None = None,
    output_path: str | None = None,
    title: str = "",
) -> list:
    """Plot phase portrait or 3D body trajectory from a state log CSV.

    Args:
        csv_path: path to CSV from export_csv or export_state_log.
        plot_type: "phase" (qpos_N vs qvel_N) or "path3d" (body 3D path).
        dof: DOF index for phase mode (default 0).
        body: body name for path3d mode (requires body_xpos:<name> in CSV).
        output_path: if given, save PNG; else return inline.
        title: plot title.
    """
    await asyncio.sleep(0)
    return _plot_trajectory_impl(csv_path, plot_type, dof, body, output_path, title)
