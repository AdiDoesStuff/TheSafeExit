"""
heatmap_sequence.py — Time-Evolving Bottleneck Heatmap Visualizations (Phase 3)

Provides functions to analyze, visualize, and animate the dynamics of Markov
fundamental matrix bottlenecks N_t = (I - Q_t)^(-1) as panic erodes rationality.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional, Sequence, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter

from src.floor_plan import CELL_WALL, CELL_WALKABLE, CELL_EXIT


def global_colorscale(snapshots: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Computes a single consistent (vmin, vmax) color scale across a sequence of
    2D bottleneck heatmaps.

    Parameters:
        snapshots: Sequence of snapshot dictionaries containing 'heatmap_2d' arrays.

    Returns:
        (vmin, vmax): Global minimum and maximum expected evacuation times.
    """
    if not snapshots:
        return (0.0, 1.0)

    vmins = []
    vmaxs = []

    for snap in snapshots:
        h = snap.get("heatmap_2d")
        if h is not None:
            valid = h[~np.isnan(h)]
            if len(valid) > 0:
                vmins.append(float(np.min(valid)))
                vmaxs.append(float(np.max(valid)))

    if not vmins or not vmaxs:
        return (0.0, 1.0)

    vmin = min(vmins)
    vmax = max(vmaxs)

    if vmax <= vmin:
        vmax = vmin + 1.0

    return (vmin, vmax)


def track_bottleneck_migration(
    snapshots: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Tracks the trajectory of the primary (#1) bottleneck cell across all snapshots.

    Parameters:
        snapshots: Sequence of snapshot dictionaries.

    Returns:
        List of dicts: [{'t': int, 'cell': (r, c), 'value': float}, ...]
    """
    migration = []
    for snap in snapshots:
        t = snap["t"]
        top3 = snap.get("top3_cells", [])
        if top3:
            r, c, val = top3[0]
            migration.append({"t": t, "cell": (int(r), int(c)), "value": float(val)})
        else:
            h = snap.get("heatmap_2d")
            if h is not None:
                valid_coords = np.argwhere(~np.isnan(h))
                if len(valid_coords) > 0:
                    vals = h[valid_coords[:, 0], valid_coords[:, 1]]
                    peak_idx = int(np.argmax(vals))
                    r, c = valid_coords[peak_idx]
                    migration.append({"t": t, "cell": (int(r), int(c)), "value": float(vals[peak_idx])})
    return migration


def plot_heatmap_grid(
    snapshots: Sequence[Dict[str, Any]],
    floor_plan: np.ndarray,
    ncols: int = 4,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Renders a static multi-panel grid of bottleneck heatmaps across time steps,
    using a shared global color scale.

    Parameters:
        snapshots: Sequence of snapshot dictionaries.
        floor_plan: 2D numpy array with cell encodings.
        ncols: Number of columns in subplot grid.
        save_path: Optional file path to save the figure.

    Returns:
        (fig, axes): Matplotlib Figure and array of Axes.
    """
    n_snaps = len(snapshots)
    if n_snaps == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No Snapshots Available", ha="center", va="center")
        return fig, np.array([ax])

    nrows = math.ceil(n_snaps / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 3.8, nrows * 3.4),
        squeeze=False,
    )

    vmin, vmax = global_colorscale(snapshots)
    cmap = plt.cm.plasma.copy()
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad="#1e1e24")
    else:
        cmap.set_bad(color="#1e1e24")

    exit_mask = (floor_plan == CELL_EXIT)
    exit_r, exit_c = np.where(exit_mask) if np.any(exit_mask) else ([], [])

    last_im = None
    for idx, snap in enumerate(snapshots):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        h = snap["heatmap_2d"]
        masked_h = np.ma.masked_invalid(h)
        im = ax.imshow(masked_h, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        last_im = im

        # Exits
        if len(exit_r) > 0:
            ax.scatter(exit_c, exit_r, color="#2ec4b6", marker="s", s=25, zorder=4)

        # Top 3 bottleneck cells
        top3 = snap.get("top3_cells", [])
        for rank, (br, bc, bval) in enumerate(top3, 1):
            color = "#ff0054" if rank == 1 else "#ffbe0b"
            ax.scatter(
                bc,
                br,
                color=color,
                marker="o",
                s=40 if rank == 1 else 25,
                edgecolors="white",
                linewidth=0.8,
                zorder=5,
            )

        mean_p = snap.get("mean_panic", 0.0)
        active = snap.get("active_agents", 0)
        ax.set_title(
            f"t = {snap['t']} | Panic: {mean_p:.2f} | Act: {active}",
            fontsize=9,
            fontweight="bold",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for idx in range(n_snaps, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle(
        r"Time-Evolving Bottleneck Heatmap Sequence $N_t = (I - Q_t)^{-1}$",
        fontsize=14,
        fontweight="bold",
        y=1.002,
    )
    plt.tight_layout()

    if last_im is not None:
        fig.subplots_adjust(right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(last_im, cax=cbar_ax)
        cbar.set_label("Expected Evacuation Time Steps", fontsize=10, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, axes


def animate_heatmap_sequence(
    snapshots: Sequence[Dict[str, Any]],
    floor_plan: np.ndarray,
    history: Optional[Sequence[Dict[str, Any]]] = None,
    save_path: Optional[str] = None,
    interval: int = 200,
    fps: int = 5,
) -> FuncAnimation:
    """
    Creates a synchronized 2-panel animation of the evolving bottleneck heatmap
    alongside panic level and active agent trajectories.

    Parameters:
        snapshots: Sequence of snapshot dictionaries.
        floor_plan: 2D numpy array floor plan.
        history: Optional full simulation history log.
        save_path: Optional file path (.gif) to save animation.
        interval: Delay between frames in milliseconds.
        fps: Frames per second for output video/gif.

    Returns:
        anim: Matplotlib FuncAnimation object.
    """
    if not snapshots:
        raise ValueError("Cannot animate an empty snapshot list.")

    fig, (ax_map, ax_curve) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    vmin, vmax = global_colorscale(snapshots)
    cmap = plt.cm.plasma.copy()
    if hasattr(cmap, "with_extremes"):
        cmap = cmap.with_extremes(bad="#1e1e24")
    else:
        cmap.set_bad(color="#1e1e24")

    # Time series curve data
    if history:
        steps_curve = [h["step"] for h in history]
        panic_curve = [h["mean_panic"] for h in history]
        active_curve = [h["active_agents"] for h in history]
    else:
        steps_curve = [s["t"] for s in snapshots]
        panic_curve = [s["mean_panic"] for s in snapshots]
        active_curve = [s["active_agents"] for s in snapshots]

    # Initialize right panel (time series)
    ax_curve.set_xlabel("Simulation Timestep (t)", fontsize=10, fontweight="bold")
    ax_curve.set_ylabel("Mean Panic Level $\\bar{u}_t$", color="#e63946", fontsize=10, fontweight="bold")
    line_panic, = ax_curve.plot(
        steps_curve, panic_curve, color="#e63946", lw=2, label="Mean Panic"
    )
    ax_curve.tick_params(axis="y", labelcolor="#e63946")
    ax_curve.grid(True, linestyle=":", alpha=0.5)

    ax_agents = ax_curve.twinx()
    ax_agents.set_ylabel("Active Agents", color="#3a86ef", fontsize=10, fontweight="bold")
    line_active, = ax_agents.plot(
        steps_curve, active_curve, color="#3a86ef", lw=2, linestyle="--", label="Active Agents"
    )
    ax_agents.tick_params(axis="y", labelcolor="#3a86ef")

    # Dynamic cursor line indicating current frame
    cursor = ax_curve.axvline(
        x=snapshots[0]["t"], color="#ffbe0b", linestyle="-", lw=2, alpha=0.9
    )

    # Initialize left panel (heatmap)
    first_snap = snapshots[0]
    masked_first = np.ma.masked_invalid(first_snap["heatmap_2d"])
    im = ax_map.imshow(masked_first, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)

    # Exits
    exit_mask = (floor_plan == CELL_EXIT)
    if np.any(exit_mask):
        exit_r, exit_c = np.where(exit_mask)
        ax_map.scatter(exit_c, exit_r, color="#2ec4b6", marker="s", s=50, label="Exit", zorder=4)

    # Scatter plot placeholder for top-3 bottlenecks
    scatter_bottlenecks = ax_map.scatter([], [], color="#ff0054", marker="o", s=70, edgecolors="white", lw=1.2, zorder=5)

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
    cbar.set_label("Expected Evacuation Steps", fontsize=9, fontweight="bold")

    title_text = ax_map.set_title("", fontsize=11, fontweight="bold", pad=8)
    ax_map.set_xticks([])
    ax_map.set_yticks([])

    fig.suptitle("Phase 3: Dynamic Bottleneck Migration Under Panic Feedback", fontsize=13, fontweight="bold")
    plt.tight_layout()

    def update(frame_idx: int):
        snap = snapshots[frame_idx]
        t = snap["t"]

        # Update heatmap
        masked_h = np.ma.masked_invalid(snap["heatmap_2d"])
        im.set_data(masked_h)

        # Update top-3 scatter
        top3 = snap.get("top3_cells", [])
        if top3:
            pts = np.array([[c, r] for r, c, _ in top3])
            scatter_bottlenecks.set_offsets(pts)
        else:
            scatter_bottlenecks.set_offsets(np.empty((0, 2)))

        # Update title
        mean_p = snap.get("mean_panic", 0.0)
        active = snap.get("active_agents", 0)
        title_text.set_text(f"t = {t} | Mean Panic: {mean_p:.2f} | Active: {active}")

        # Update cursor line
        cursor.set_xdata([t, t])

        return im, scatter_bottlenecks, title_text, cursor

    anim = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=interval,
        blit=False,
    )

    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close(fig)

    return anim
