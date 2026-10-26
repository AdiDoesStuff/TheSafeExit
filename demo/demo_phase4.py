"""
demo_phase4.py — Phase 4 Demo: Monte Carlo Validation Against Closed-Form Fundamental Matrix

Validates the discrete stochastic simulation against the analytical absorbing Markov chain
fundamental matrix N = (I - Q)^(-1).

Run from repository root:
    .venv\\Scripts\\python demo\\demo_phase4.py

Produces:
    demo/figures/phase4_01_monte_carlo_convergence.png
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np

from src.floor_plan import (
    CELL_WALKABLE,
    compute_exit_distance_field,
    get_simple_room,
)
from src.markov import (
    build_rational_QR,
    build_uniform_QR,
    build_mixed_QR,
    build_state_map,
)
from src.solvers import (
    fundamental_matrix_lu,
    run_monte_carlo,
)

FIGURES_DIR = repo_root / "demo" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ROWS, COLS = 20, 20
EXIT_WIDTH = 2
M_VALUES = [10, 50, 100, 500, 1000, 5000]
SEED = 42


def run_phase4_validation():
    print("=" * 75)
    print("  PHASE 4: MONTE CARLO VALIDATION AGAINST CLOSED-FORM FUNDAMENTAL MATRIX")
    print("=" * 75)

    # 1. Floor plan and Markov transition setup
    grid = get_simple_room(ROWS, COLS, exit_width=EXIT_WIDTH)
    dist_field = compute_exit_distance_field(grid)
    state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)

    # Transition matrices
    Q_rational, R_rational = build_rational_QR(
        grid, dist_field, state_map, n_transient, n_absorbing
    )
    N_rational = fundamental_matrix_lu(Q_rational)

    Q_uniform, R_uniform = build_uniform_QR(
        grid, state_map, n_transient, n_absorbing
    )
    N_uniform = fundamental_matrix_lu(Q_uniform)

    # 2. Select start cell: Farthest walkable cell from the exit (maximum geodesic distance)
    walkable_mask = (grid == CELL_WALKABLE)
    walkable_coords = list(zip(*np.where(walkable_mask)))
    distances = [dist_field[r, c] for r, c in walkable_coords]
    max_dist_idx = int(np.argmax(distances))
    start_cell = walkable_coords[max_dist_idx]
    start_dist = distances[max_dist_idx]
    start_state_idx = state_map[start_cell]

    # Closed-form expected evacuation times
    target_rational = float(np.sum(N_rational[start_state_idx, :]))
    target_uniform = float(np.sum(N_uniform[start_state_idx, :]))

    print(f"  Floor plan dimensions   : {ROWS}x{COLS} with {EXIT_WIDTH}-cell exit on right wall")
    print(f"  Transient states (N)    : {n_transient}")
    print(f"  Chosen start cell       : {start_cell} (farthest cell from exit, geodesic dist = {start_dist})")
    print(f"  Rational closed-form E[T] : {target_rational:.4f} steps (1^T N_rational)")
    print(f"  Uniform closed-form E[T]  : {target_uniform:.4f} steps (1^T N_uniform)")
    print("-" * 75)
    print("  1. Rational Chain (Steepest Descent) Validation:")
    print(f"  {'M (Trials)':<12}{'MC Mean':>12}{'Std Error':>12}{'Abs Error':>12}{'Rel Error (%)':>15}")
    print("-" * 75)

    mc_means_rat = []
    mc_se_rat = []
    abs_err_rat = []
    rel_err_rat = []

    rng = np.random.default_rng(SEED)

    for M in M_VALUES:
        res = run_monte_carlo(
            Q_rational, R_rational, start_state_idx=start_state_idx, M=M, rng=rng
        )
        mean_val = res["mean"]
        se_val = res["std_error"]
        abs_err = abs(mean_val - target_rational)
        rel_err = (abs_err / target_rational) * 100.0

        mc_means_rat.append(mean_val)
        mc_se_rat.append(se_val)
        abs_err_rat.append(abs_err)
        rel_err_rat.append(rel_err)

        print(
            f"  {M:<12}{mean_val:>12.4f}{se_val:>12.4f}{abs_err:>12.4f}{rel_err:>14.2f}%"
        )

    print("\n" + "-" * 75)
    print("  2. Stochastic Uniform Random Walk Chain Validation (Empirical Variance > 0):")
    print(f"  {'M (Trials)':<12}{'MC Mean':>12}{'Std Error':>12}{'Abs Error':>12}{'Rel Error (%)':>15}")
    print("-" * 75)

    mc_means_unif = []
    mc_se_unif = []
    abs_err_unif = []
    rel_err_unif = []

    for M in M_VALUES:
        res = run_monte_carlo(
            Q_uniform, R_uniform, start_state_idx=start_state_idx, M=M, rng=rng
        )
        mean_val = res["mean"]
        se_val = res["std_error"]
        abs_err = abs(mean_val - target_uniform)
        rel_err = (abs_err / target_uniform) * 100.0

        mc_means_unif.append(mean_val)
        mc_se_unif.append(se_val)
        abs_err_unif.append(abs_err)
        rel_err_unif.append(rel_err)

        print(
            f"  {M:<12}{mean_val:>12.4f}{se_val:>12.4f}{abs_err:>12.4f}{rel_err:>14.2f}%"
        )

    print("=" * 75)
    print(f"  Final Validation Summary (M = {M_VALUES[-1]}):")
    print(f"    [Rational] Target: {target_rational:.4f} | MC: {mc_means_rat[-1]:.4f} +/- {mc_se_rat[-1]:.4f} | Rel Err: {rel_err_rat[-1]:.3f}%")
    print(f"    [Uniform]  Target: {target_uniform:.4f} | MC: {mc_means_unif[-1]:.4f} +/- {mc_se_unif[-1]:.4f} | Rel Err: {rel_err_unif[-1]:.3f}%")
    print("=" * 75 + "\n")

    # 3. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    m_arr = np.array(M_VALUES)
    means_unif_arr = np.array(mc_means_unif)
    se_unif_arr = np.array(mc_se_unif)

    # Panel 1: Convergence of estimate with +/- 1 std error band
    ax1.set_xscale("log")
    ax1.axhline(
        target_uniform,
        color="#2ec4b6",
        linestyle="--",
        linewidth=2.2,
        label=f"Closed-form $1^T N_{{unif}} = {target_uniform:.1f}$",
        zorder=3,
    )
    ax1.fill_between(
        m_arr,
        means_unif_arr - se_unif_arr,
        means_unif_arr + se_unif_arr,
        color="#e63946",
        alpha=0.22,
        label=r"$\pm 1\,\mathrm{SE}$ Confidence Band",
    )
    ax1.plot(
        m_arr,
        means_unif_arr,
        marker="o",
        markersize=6,
        color="#e63946",
        linewidth=1.8,
        label=r"Monte Carlo Mean $\hat{T}_M$",
        zorder=4,
    )

    ax1.set_title("Monte Carlo Estimate vs. Closed-Form Target", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Simulation Trials $M$ (log scale)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Expected Evacuation Time (Steps)", fontsize=10, fontweight="bold")
    ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax1.legend(loc="best", frameon=True, fontsize=9)

    # Panel 2: Log-log error convergence rate vs 1/sqrt(M)
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Theoretical C / sqrt(M) reference line normalized at first empirical point
    c_theory = se_unif_arr[0] * np.sqrt(m_arr[0])
    theory_line = c_theory / np.sqrt(m_arr)

    ax2.plot(
        m_arr,
        se_unif_arr,
        marker="s",
        markersize=6,
        color="#3a86ef",
        linewidth=1.8,
        label=r"Empirical SE $(\sigma / \sqrt{M})$",
        zorder=4,
    )
    ax2.plot(
        m_arr,
        theory_line,
        linestyle="--",
        color="#ff0054",
        linewidth=1.8,
        label=r"Theoretical $O(1/\sqrt{M})$ Rate",
        zorder=3,
    )

    ax2.set_title(r"Standard Error Convergence Rate $\propto 1/\sqrt{M}$", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Simulation Trials $M$ (log scale)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Standard Error (log scale)", fontsize=10, fontweight="bold")
    ax2.grid(True, which="both", linestyle=":", alpha=0.5)
    ax2.legend(loc="best", frameon=True, fontsize=9)

    fig.suptitle(
        f"Phase 4: Monte Carlo Convergence Validation (Start Cell {start_cell}, Geodesic Dist = {start_dist})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()

    save_path = FIGURES_DIR / "phase4_01_monte_carlo_convergence.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure to {save_path}\n")

    return {
        "target_rational": target_rational,
        "target_uniform": target_uniform,
        "m_values": M_VALUES,
        "mc_means_rat": mc_means_rat,
        "mc_se_rat": mc_se_rat,
        "mc_means_unif": mc_means_unif,
        "mc_se_unif": mc_se_unif,
    }


if __name__ == "__main__":
    run_phase4_validation()
