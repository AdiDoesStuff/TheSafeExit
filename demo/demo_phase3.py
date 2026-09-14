"""
demo_phase3.py — Phase 3 Demo: Time-Evolving Bottleneck Heatmaps Under Panic Dynamics

Visualizes how the absorbing Markov chain fundamental matrix bottlenecks N_t = (I - Q_t)^(-1)
evolve and migrate over time as crowd panic erodes rational movement toward exits.

Run from repository root:
    .venv\\Scripts\\python demo\\demo_phase3.py

Produces three artifacts in demo/figures/:
  phase3_01_heatmap_grid.png          — 30-panel static time-series grid of N_t heatmaps
  phase3_02_heatmap_animation.gif     — Synchronized 2-panel animation (Heatmap + Panic/Agent dynamics)
  phase3_03_bottleneck_migration.png  — Trajectory map tracking spatial migration of the #1 bottleneck cell
"""

import os
import sys

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.floor_plan import get_simple_room, CELL_WALL, CELL_WALKABLE, CELL_EXIT, plot_floor_plan
from src.simulation import EvacuationSimulation
from src.heatmap_sequence import (
    global_colorscale,
    plot_heatmap_grid,
    animate_heatmap_sequence,
    track_bottleneck_migration,
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def main():
    print("\n" + "=" * 75)
    print("  PHASE 3: TIME-EVOLVING BOTTLENECK HEATMAPS & N_t DYNAMICS")
    print("=" * 75)

    # 1. Initialize Simulation
    grid_size = 20
    exit_width = 3
    n_agents = 80
    seed = 7
    snapshot_every = 20
    max_snapshots = 30
    max_steps = 600

    floor_plan = get_simple_room(grid_size, grid_size, exit_width=exit_width)
    print(f"\n[1/4] Initializing EvacuationSimulation on {grid_size}x{grid_size} grid...")
    print(f"      Agents: {n_agents} | Beta (Panic coupling): 0.4 | Seed: {seed}")
    print(f"      Snapshot cadence: every {snapshot_every} steps (max {max_snapshots} frames)")

    sim = EvacuationSimulation(
        floor_plan=floor_plan,
        n_agents=n_agents,
        D=0.2,
        k=0.5,
        beta=0.4,
        seed=seed,
    )

    # 2. Run Simulation with Reduced Heatmap Snapshots
    print(f"\n[2/4] Running simulation for up to {max_steps} timesteps...")
    results = sim.run(
        max_steps=max_steps,
        snapshot_every=snapshot_every,
        max_snapshots=max_snapshots,
    )

    total_steps = results["total_steps"]
    evac_rate = results["evacuation_rate"]
    mean_evac = results["mean_evacuation_time"]
    snapshots = results["snapshots"]
    history = results["history"]

    print(f"      Simulation completed in {total_steps} steps.")
    print(f"      Evacuation rate : {evac_rate * 100:.1f}% ({results['evacuated_agents']}/{n_agents} agents)")
    print(f"      Mean evac time  : {mean_evac:.1f} steps")
    print(f"      Snapshots taken : {len(snapshots)}")

    vmin, vmax = global_colorscale(snapshots)
    print(f"      Global heatmap range: [{vmin:.1f}, {vmax:.1f}] expected steps")

    # 3. Generate Static Heatmap Grid
    print(f"\n[3/4] Generating static multi-panel heatmap grid...")
    grid_path = os.path.join(FIGURES_DIR, "phase3_01_heatmap_grid.png")
    fig_grid, _ = plot_heatmap_grid(
        snapshots=snapshots,
        floor_plan=floor_plan,
        ncols=5,
        save_path=grid_path,
    )
    plt.close(fig_grid)
    print(f"  Saved -> {grid_path}")

    # 4. Generate Animated Heatmap Sequence
    print(f"\n[4/4] Generating synchronized 2-panel animation (GIF)...")
    anim_path = os.path.join(FIGURES_DIR, "phase3_02_heatmap_animation.gif")
    animate_heatmap_sequence(
        snapshots=snapshots,
        floor_plan=floor_plan,
        history=history,
        save_path=anim_path,
        interval=200,
        fps=5,
    )
    print(f"  Saved -> {anim_path}")

    # 5. Bottleneck Migration Trajectory Map
    print(f"\n[5/5] Visualizing spatial migration path of primary (#1) bottleneck...")
    migration = track_bottleneck_migration(snapshots)

    fig, ax = plt.subplots(figsize=(8, 7.5))
    plot_floor_plan(floor_plan, title="Primary Bottleneck Migration Trajectory Over Time", ax=ax)

    # Plot migration path
    if len(migration) > 0:
        times = [m["t"] for m in migration]
        coords = [m["cell"] for m in migration]
        vals = [m["value"] for m in migration]

        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]

        # Draw line path connecting sequential bottleneck centers
        ax.plot(cols, rows, color="#ff0054", linestyle="--", linewidth=1.5, alpha=0.7, zorder=6)

        # Draw scatter points colored by time step
        norm = mcolors.Normalize(vmin=min(times), vmax=max(times))
        scatter = ax.scatter(
            cols,
            rows,
            c=times,
            cmap="autumn_r",
            norm=norm,
            s=100,
            edgecolors="white",
            linewidth=1.2,
            zorder=7,
            label="Bottleneck #1 (by time)",
        )

        # Draw directional arrows along trajectory
        for i in range(len(coords) - 1):
            r1, c1 = coords[i]
            r2, c2 = coords[i + 1]
            if (r1, c1) != (r2, c2):
                ax.annotate(
                    "",
                    xy=(c2, r2),
                    xytext=(c1, r1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#ffbe0b",
                        lw=1.5,
                        mutation_scale=12,
                    ),
                    zorder=8,
                )

        # Annotate start and finish of bottleneck trail
        r_start, c_start = coords[0]
        r_end, c_end = coords[-1]
        ax.annotate(
            f"Start (t={times[0]})\n{vals[0]:.1f} steps",
            xy=(c_start, r_start),
            xytext=(c_start - 3.5, r_start - 1.5),
            arrowprops=dict(arrowstyle="->", color="#3a86ef", lw=1.5),
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e24", alpha=0.9, edgecolor="#3a86ef"),
            zorder=9,
        )
        ax.annotate(
            f"End (t={times[-1]})\n{vals[-1]:.1f} steps",
            xy=(c_end, r_end),
            xytext=(c_end + 1.5, r_end + 1.5),
            arrowprops=dict(arrowstyle="->", color="#ff0054", lw=1.5),
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e1e24", alpha=0.9, edgecolor="#ff0054"),
            zorder=9,
        )

        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label("Timestep (t)", fontsize=10, fontweight="bold")

    save(fig, "phase3_03_bottleneck_migration.png")

    # Summary
    print("\n" + "=" * 75)
    print("  PHASE 3 DEMO COMPLETE")
    print("=" * 75)
    print("  Deliverables:")
    print("  [x] Reduced N_t snapshot capture in simulation loop (src/simulation.py)")
    print("  [x] Global color scaling across temporal frames (src/heatmap_sequence.py)")
    print("  [x] Multi-panel static heatmap grid (demo/figures/phase3_01_heatmap_grid.png)")
    print("  [x] Synchronized 2-panel GIF animation (demo/figures/phase3_02_heatmap_animation.gif)")
    print("  [x] Primary bottleneck spatial migration tracking (demo/figures/phase3_03_bottleneck_migration.png)")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
