"""
demo_phase3b.py — Phase 3b: Generalization study across floor-plan geometry

Question: are the Phase 2/3 findings (O(L) -> O(L^2) collapse under panic
saturation, bottleneck migration to the geometrically farthest point)
properties of the MODEL, or artifacts of one specific room shape?

Method: run the identical simulation (D, k, beta, n_agents, seed all held
fixed) across three floor plans. Only geometry varies. If the qualitative
pattern holds across all three, it's a model property. If it doesn't, that's
equally worth knowing before this goes in the report as a general claim.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np

from src.floor_plan import (
    CELL_WALKABLE,
    get_room_with_pillar,
    get_simple_room,
    get_two_exit_room,
    compute_exit_distance_field,
    plot_floor_plan,
    validate_reachability,
)
from src.heatmap import fundamental_to_heatmap
from src.heatmap_sequence import track_bottleneck_migration
from src.markov import build_rational_QR, build_state_map, build_uniform_QR
from src.simulation import EvacuationSimulation
from src.solvers import fundamental_matrix_lu

FIGURES_DIR = repo_root / "demo" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Fixed across every geometry. Only the floor plan changes between runs —
# this is the whole point of the experiment.
ROWS, COLS = 20, 20
N_AGENTS = 80
SEED = 7
D = 0.2
K = 0.5
BETA = 0.4
MAX_STEPS = 150  # saturation locks in by ~t=60 per Phase 3 — no need for 600
SNAPSHOT_EVERY = 10
MAX_SNAPSHOTS = 15

GEOMETRIES = {
    "baseline": get_simple_room(ROWS, COLS, exit_width=2),
    "two_exit": get_two_exit_room(ROWS, COLS, exit_width=2),
    "pillar": get_room_with_pillar(ROWS, COLS, exit_width=2, pillar_size=4),
}


def static_rational_vs_uniform(grid: np.ndarray, name: str) -> dict:
    """
    Pure Q_rational vs Q_uniform comparison — no panic, no simulation. This
    is the theoretical ceiling each geometry's coupled simulation should
    approach once panic fully saturates (lambda -> 1 everywhere).
    """
    validate_reachability(grid)
    dist = compute_exit_distance_field(grid)
    state_map, idx_map, n_t, n_a = build_state_map(grid)

    Q_rat, R_rat = build_rational_QR(grid, dist, state_map, n_t, n_a)
    N_rat = fundamental_matrix_lu(Q_rat)
    h_rat = fundamental_to_heatmap(N_rat, grid, state_map, n_t)

    Q_unif, R_unif = build_uniform_QR(grid, state_map, n_t, n_a)
    N_unif = fundamental_matrix_lu(Q_unif)
    h_unif = fundamental_to_heatmap(N_unif, grid, state_map, n_t)

    walkable = grid == CELL_WALKABLE
    rational_mean = float(np.nanmean(h_rat[walkable]))
    uniform_mean = float(np.nanmean(h_unif[walkable]))

    return {
        "geometry": name,
        "rational_mean": rational_mean,
        "rational_max": float(np.nanmax(h_rat[walkable])),
        "uniform_mean": uniform_mean,
        "uniform_max": float(np.nanmax(h_unif[walkable])),
        "ratio": uniform_mean / rational_mean,
    }


def run_coupled_simulation(grid: np.ndarray, name: str) -> dict:
    """Full panic-coupled EvacuationSimulation run for one geometry."""
    sim = EvacuationSimulation(grid, n_agents=N_AGENTS, seed=SEED, D=D, k=K, beta=BETA)
    result = sim.run(
        max_steps=MAX_STEPS,
        snapshot_every=SNAPSHOT_EVERY,
        max_snapshots=MAX_SNAPSHOTS,
    )
    snapshots = result["snapshots"]
    final = snapshots[-1]

    # First snapshot at which mean panic crosses the full-rationality-loss
    # threshold u >= 1/beta (i.e. lambda would hit 1 for an average cell).
    threshold = 1.0 / BETA
    saturation_t = next((s["t"] for s in snapshots if s["mean_panic"] >= threshold), None)

    return {
        "geometry": name,
        "sim_result": result,
        "final_active_agents": final["active_agents"],
        "evac_rate": 1.0 - final["active_agents"] / N_AGENTS,
        "saturation_t": saturation_t,
    }


def exit_choice_consistency(sim_result: dict, saturation_t: Optional[int] = None) -> None:
    """
    Analyzes exit choice switching for the two-exit geometry, correctly split by phase:
    1. Pre-saturation (t < saturation_t): Rational / gradient-directed regime.
    2. Post-saturation (t >= saturation_t): Panic-saturated random walk / Brownian crossing regime.
    Properly filters out absorbed agent states (-1, -1) to prevent false switching artifacts.
    """
    history = sim_result.get("agent_position_history")
    if history is None:
        print("  [skipped] agent_position_history not found in sim_result.")
        return

    history = np.asarray(history)  # shape (steps, n_agents, 2)
    n_steps, n_agents, _ = history.shape
    midpoint_col = COLS / 2.0
    active_mask = (history[:, :, 0] >= 0)

    def count_switches(step_start: int, step_end: int) -> int:
        sub_history = history[step_start:step_end]
        sub_active = active_mask[step_start:step_end]
        switched = 0
        for a in range(n_agents):
            agent_act = sub_active[:, a]
            act_idx = np.where(agent_act)[0]
            if len(act_idx) > 1:
                sides = (sub_history[act_idx, a, 1] >= midpoint_col).astype(int)
                if np.any(np.diff(sides) != 0):
                    switched += 1
        return switched

    total_switched = count_switches(0, n_steps)

    print("\n  [two_exit] Exit Choice Consistency Analysis (Phase-Split):")
    if saturation_t is not None and 0 < saturation_t < n_steps:
        pre_switched = count_switches(0, saturation_t)
        post_switched = count_switches(saturation_t, n_steps)
        print(f"    - Pre-saturation  (t < {saturation_t:2d}, rational regime)     : {pre_switched}/{n_agents} ({100*pre_switched/n_agents:.1f}%)")
        print(f"    - Post-saturation (t >= {saturation_t:2d}, random walk regime): {post_switched}/{n_agents} ({100*post_switched/n_agents:.1f}%)")
        print(f"    - Overall lifetime switching (active steps only) : {total_switched}/{n_agents} ({100*total_switched/n_agents:.1f}%)")
    else:
        print(f"    - Overall lifetime switching (active steps only) : {total_switched}/{n_agents} ({100*total_switched/n_agents:.1f}%)")


def main():
    print(f"{'='*70}\nPhase 3b -- Generalization study across floor-plan geometry\n{'='*70}")

    static_rows = []
    for name, grid in GEOMETRIES.items():
        row = static_rational_vs_uniform(grid, name)
        static_rows.append(row)
        print(f"\n[{name}] Static comparison (no panic, no simulation):")
        print(f"  Rational: mean={row['rational_mean']:.1f}  max={row['rational_max']:.1f}")
        print(f"  Uniform:  mean={row['uniform_mean']:.1f}  max={row['uniform_max']:.1f}")
        print(f"  Ratio (uniform/rational): {row['ratio']:.1f}x")

    sim_rows = []
    for name, grid in GEOMETRIES.items():
        print(f"\nRunning coupled simulation: {name} ...")
        row = run_coupled_simulation(grid, name)
        sim_rows.append(row)
        print(f"  Evac rate @ t={MAX_STEPS}: {row['evac_rate']*100:.1f}%")
        print(f"  Saturation reached at t={row['saturation_t']}")
        if name == "two_exit":
            exit_choice_consistency(row["sim_result"], saturation_t=row["saturation_t"])

    print(f"\n{'='*70}\nComparison table\n{'='*70}")
    header = f"{'Geometry':<12}{'Rational':>10}{'Uniform':>10}{'Ratio':>8}{'Evac%':>8}{'Sat.t':>8}"
    print(header)
    print("-" * len(header))
    for s_row, sim_row in zip(static_rows, sim_rows):
        print(
            f"{s_row['geometry']:<12}"
            f"{s_row['rational_mean']:>10.1f}"
            f"{s_row['uniform_mean']:>10.1f}"
            f"{s_row['ratio']:>7.1f}x"
            f"{sim_row['evac_rate']*100:>7.1f}%"
            f"{str(sim_row['saturation_t']):>8}"
        )

    # --- Figure 1: the three floor plans, side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, grid) in zip(axes, GEOMETRIES.items()):
        plot_floor_plan(grid, title=name.replace("_", " ").title(), ax=ax)
    fig.suptitle("Phase 3b: Floor-plan geometries under comparison", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase3b_01_geometries.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: bottleneck migration overlaid on each geometry ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, grid) in zip(axes, GEOMETRIES.items()):
        sim_row = next(r for r in sim_rows if r["geometry"] == name)
        migration = track_bottleneck_migration(sim_row["sim_result"]["snapshots"])
        plot_floor_plan(grid, title=f"{name}: bottleneck migration", ax=ax)
        rows_ = [m["cell"][0] for m in migration]
        cols_ = [m["cell"][1] for m in migration]
        ts_ = [m["t"] for m in migration]
        ax.scatter(cols_, rows_, c=ts_, cmap="autumn_r", s=60, zorder=5, edgecolors="black")
        ax.plot(cols_, rows_, "--", color="orange", alpha=0.6, zorder=4)
    fig.suptitle("Phase 3b: Where does the bottleneck end up, per geometry?", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "phase3b_02_migration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figures to {FIGURES_DIR}/")
    print("  phase3b_01_geometries.png")
    print("  phase3b_02_migration_comparison.png")

    return static_rows, sim_rows


if __name__ == "__main__":
    main()
