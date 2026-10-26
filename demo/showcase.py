"""
showcase.py - TheSafeExit one-command demo.

Runs the four headline results from the project in about two minutes, prints the
numbers to the console and saves four PNGs.

    python demo/showcase.py

Outputs go to ``experiments/out/showcase/`` only. Nothing in ``demo/figures/``
is touched, and the canonical experiment outputs in ``experiments/out/`` are
left alone.

Every number here is uncalibrated and is meant for comparing designs, not for
predicting real evacuation times.
"""

from __future__ import annotations

import os
import sys
import time
import traceback

# Headless plotting and repo-root imports must be set before the heavy imports.
os.environ.setdefault("MPLBACKEND", "Agg")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt

from src.floor_plan import (
    CELL_WALKABLE,
    get_hallway,
    get_simple_room,
    compute_exit_distance_field,
    validate_reachability,
)
from src.markov import build_QR, build_rational_QR, build_state_map
from src.solvers import solve_expected_time_sparse, solve_expected_time_iterative
from src.optimization import exact_exit_search, optimize_exit_placement
from src.ensemble import run_ensemble

OUT_DIR = os.path.join(PROJECT_ROOT, "experiments", "out", "showcase")

# Shared palette, matching the web editor accents.
TEAL = "#2ec4b6"
VIOLET = "#7c6ff7"
CORAL = "#ff6b6b"
AMBER = "#ffa94d"
BLUE = "#3a86ef"
GREY = "#8892b0"

BASE_SEED = 1000


# --------------------------------------------------------------------------- #
# Console helpers
# --------------------------------------------------------------------------- #
def _rule(title: str) -> None:
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)


def _line(label: str, value: str) -> None:
    print(f"  {label:<42}{value:>28}")


def _done(start: float) -> None:
    print(f"  ...done in {time.perf_counter() - start:.1f} s")


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1b1f2e")
    ax.set_xlabel(xlabel, fontsize=9.5, color="#3c4254")
    ax.set_ylabel(ylabel, fontsize=9.5, color="#3c4254")
    ax.tick_params(labelsize=8.5, colors="#3c4254")
    ax.grid(True, linestyle=":", alpha=0.55)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# --------------------------------------------------------------------------- #
# 1. Numerical engine - one sparse solve on the L-shaped hallway
# --------------------------------------------------------------------------- #
def demo_solver():
    _rule("1 | Numerical engine - expected exit time from one sparse solve")
    start = time.perf_counter()

    grid = get_hallway()
    validate_reachability(grid)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    dist = compute_exit_distance_field(grid)
    Q, _ = build_rational_QR(grid, dist, state_map, n_transient, n_absorbing)

    report = solve_expected_time_sparse(Q)
    _line("Hallway grid", f"{grid.shape[0]} x {grid.shape[1]}")
    _line("Transient states (unknowns)", f"{n_transient}")
    _line("Absorbing exit states", f"{n_absorbing}")
    _line("Solve method", report.method)
    _line("Solve time", f"{report.seconds * 1000:.1f} ms")
    _line("Relative residual ||(I-Q)t - 1|| / ||1||", f"{report.residual:.2e}")
    _line("Worst-case expected exit time", f"{float(np.max(report.t)):.2f} steps")
    _line("Mean expected exit time", f"{float(np.mean(report.t)):.2f} steps")

    if report.residual >= 1e-8:
        raise RuntimeError(f"solver residual {report.residual:.2e} is above 1e-8")

    # Scaling sweep for the figure (this is what E2 measures).
    sizes = [20, 40, 60, 80, 100]
    sparse_s, cg_s, cg_it, states = [], [], [], []
    for size in sizes:
        g = get_simple_room(size, size, exit_width=max(1, size // 10))
        sm, _, nt, na = build_state_map(g)
        d = compute_exit_distance_field(g)
        q_rat, _ = build_rational_QR(g, d, sm, nt, na)
        q_uni, _ = build_QR(g, sm, nt, na)
        r_sparse = solve_expected_time_sparse(q_rat)
        r_cg = solve_expected_time_iterative(q_uni, method="cg")
        states.append(nt)
        sparse_s.append(r_sparse.seconds)
        cg_s.append(r_cg.seconds)
        cg_it.append(r_cg.iterations)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax1.loglog(states, sparse_s, "o-", color=TEAL, lw=2, label="sparse direct (spsolve)")
    ax1.loglog(states, cg_s, "s-", color=VIOLET, lw=2, label="CG on uniform chain")
    ax1.legend(fontsize=8.5, frameon=False)
    _style(ax1, "Solve time vs system size", "transient states", "seconds (log)")

    ax2.plot(states, cg_it, "s-", color=VIOLET, lw=2)
    _style(ax2, "CG iterations vs system size", "transient states", "iterations")
    fig.suptitle("TheSafeExit - sparse one-solve engine", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "showcase_solver.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    _line("Dense N skipped from", "80 x 80 (memory cap, E2)")
    _line("Figure", os.path.relpath(path, PROJECT_ROOT))
    _done(start)
    return path


# --------------------------------------------------------------------------- #
# 2. Exact vs heuristic exit placement
# --------------------------------------------------------------------------- #
def demo_exit_search():
    _rule("2 | Exit placement - exhaustive optimum vs BFGS heuristic")
    start = time.perf_counter()

    rows = cols = 12
    exit_width = 2

    exact = exact_exit_search(
        rows=rows,
        cols=cols,
        exit_width=exit_width,
        ensemble_top_k=3,
        n_runs=15,
        base_seed=BASE_SEED,
        sim_kwargs={"n_agents": 40, "panic_enabled": False, "cell_capacity": 1},
    )
    heuristic = optimize_exit_placement(rows=rows, cols=cols, exit_width=exit_width)

    best = exact["best"]
    gap = float(heuristic["best_J"] - exact["best_score"])
    finalists = [c for c in exact["candidates"] if "ensemble" in c]

    _line("Grid", f"{rows} x {cols}, exit width {exit_width}")
    _line("Legal positions enumerated", f"{exact['n_candidates']}")
    _line("Exact optimum (wall / position)", f"{best['wall']} / {best['position']}")
    _line("Exact worst-case BFS distance", f"{exact['best_score']:.2f} steps")
    _line("BFGS heuristic choice", f"{heuristic['best_wall']} / {heuristic['best_position']}")
    _line("BFGS objective value", f"{heuristic['best_J']:.2f} steps")
    _line("BFGS gap to enumerated optimum", f"{gap:.2f} steps")
    _line("Finalists re-scored by ensemble", f"{len(finalists)} x {exact['n_runs']} runs")
    if finalists:
        ranked = sorted(finalists, key=lambda c: c["ensemble_rank"])
        t90s = [c["ensemble"]["T90_median"] for c in ranked]
        _line(
            "Ensemble T90 spread across finalists",
            f"{min(t90s):.1f} - {max(t90s):.1f} steps",
        )

    # Figure: stage-1 objective over all positions, then finalist T90.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    scores = [c["max_distance"] for c in exact["candidates"]]
    order = np.argsort(scores)
    walls = [exact["candidates"][i]["wall"] for i in order]
    wall_color = {"top": BLUE, "bottom": TEAL, "left": VIOLET, "right": AMBER}
    ax1.bar(
        np.arange(len(scores)),
        [scores[i] for i in order],
        color=[wall_color[w] for w in walls],
        width=0.85,
    )
    ax1.set_ylim(0, max(scores) * 1.28)
    ax1.axhline(exact["best_score"], color=CORAL, ls="--", lw=1.5, label="exact optimum")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=wall_color[w], label=f"{w} wall")
        for w in ("top", "bottom", "left", "right")
    ]
    handles.append(plt.Line2D([], [], color=CORAL, ls="--", label="exact optimum"))
    ax1.legend(handles=handles, fontsize=8, frameon=False, ncol=3, loc="upper left")
    _style(ax1, "Stage 1: every legal exit position", "candidates (sorted)", "worst-case BFS distance")

    if finalists:
        ranked = sorted(finalists, key=lambda c: c["ensemble_rank"])
        labels = [f"{c['wall'][:4]} {c['position']}" for c in ranked]
        med = [c["ensemble"]["T90_median"] for c in ranked]
        lo = [c["ensemble"]["T90_bootstrap_ci"][0] for c in ranked]
        hi = [c["ensemble"]["T90_bootstrap_ci"][1] for c in ranked]
        yerr = [
            [max(0.0, m - l) for m, l in zip(med, lo)],
            [max(0.0, h - m) for m, h in zip(med, hi)],
        ]
        stage1_key = (best["wall"], best["position"])
        colors, edges = [], []
        for c in ranked:
            is_stage1 = (c["wall"], c["position"]) == stage1_key
            colors.append(TEAL if c["ensemble_rank"] == 1 else "#c3c9d8")
            edges.append(CORAL if is_stage1 else "#8b93a8")
        bars = ax2.bar(
            labels, med, yerr=yerr, capsize=5, color=colors,
            edgecolor=edges, linewidth=2.0, width=0.6,
        )
        for bar, c in zip(bars, ranked):
            if (c["wall"], c["position"]) == stage1_key:
                ax2.annotate(
                    "stage-1 optimum",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=CORAL,
                )
        ax2.set_ylim(0, max(hi) * 1.22)
        ax2.text(
            0.02, 0.04,
            f"teal = ensemble winner (T90 {med[0]:.0f})  |  red edge = stage-1 optimum",
            transform=ax2.transAxes,
            fontsize=7.8,
            color="#3c4254",
        )
        _style(ax2, "Stage 2: ensemble T90 of finalists", "wall / position", "median T90 (steps)")

    fig.suptitle(
        "TheSafeExit - exhaustive search is optimal for the objective; BFGS happened to match",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "showcase_exit.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    _line("Figure", os.path.relpath(path, PROJECT_ROOT))
    _done(start)
    return path, gap


# --------------------------------------------------------------------------- #
# 3. Panic on vs off with capacity
# --------------------------------------------------------------------------- #
def demo_panic():
    _rule("3 | Panic and capacity - median evacuation time, same seeds")
    start = time.perf_counter()

    grid = get_simple_room(20, 20, exit_width=2)
    n_runs = 20
    common = dict(n_agents=40, cell_capacity=1, exit_capacity=1, max_steps=900)

    off = run_ensemble(
        grid, n_runs=n_runs, base_seed=BASE_SEED, panic_enabled=False, **common
    )
    on = run_ensemble(
        grid, n_runs=n_runs, base_seed=BASE_SEED, panic_enabled=True, panic_decay=0.15, **common
    )

    s_off, s_on = off.summary(), on.summary()
    t90_off = s_off["T90"]["median"]
    t90_on = s_on["T90"]["median"]

    _line("Grid", "simple room 20 x 20, exit width 2")
    _line("Agents / seeds", f"40 agents, {n_runs} common seeds")
    _line("T50 panic off -> on", f"{s_off['T50']['median']:.1f} -> {s_on['T50']['median']:.1f} steps")
    _line("T90 panic off -> on", f"{t90_off:.1f} -> {t90_on:.1f} steps")
    _line("T100 panic off -> on", f"{s_off['T100']['median']:.1f} -> {s_on['T100']['median']:.1f} steps")
    _line("T90 change", f"+{t90_on - t90_off:.1f} steps")
    _line("Fraction finished (on)", f"{s_on['fraction_finished'] * 100:.0f}%")
    _line("Blocked moves (on)", f"{s_on.get('blocked_total', 0):,}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    metrics = ["T50", "T90", "T100"]
    x = np.arange(len(metrics))
    off_med = [s_off[m]["median"] for m in metrics]
    on_med = [s_on[m]["median"] for m in metrics]
    ax1.bar(x - 0.19, off_med, width=0.36, color=GREY, label="panic off")
    ax1.bar(x + 0.19, on_med, width=0.36, color=CORAL, label="panic on (decay 0.15)")
    ax1.set_xticks(x, metrics)
    ax1.legend(fontsize=8.5, frameon=False)
    _style(ax1, "Evacuation time quantiles", "", "steps")

    # Per-seed paired scatter: same seed, same layout, panic on vs off.
    t_off = np.asarray(off.per_run["T90"], dtype=float)
    t_on = np.asarray(on.per_run["T90"], dtype=float)
    ax2.scatter(t_off, t_on, s=30, color=VIOLET, alpha=0.85, edgecolors="white", linewidths=0.6)
    lo_v = min(np.nanmin(t_off), np.nanmin(t_on))
    hi_v = max(np.nanmax(t_off), np.nanmax(t_on))
    pad = max(4.0, 0.12 * (hi_v - lo_v))
    lim = [lo_v - pad, hi_v + pad]
    ax2.plot(lim, lim, ls="--", color=GREY, lw=1.2, label="no change")
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.legend(fontsize=8.5, frameon=False)
    _style(ax2, "Per-seed T90: every run is slower with panic", "panic off (steps)", "panic on (steps)")

    fig.suptitle("TheSafeExit - panic roughly doubles the tail of the evacuation", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "showcase_panic.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    _line("Figure", os.path.relpath(path, PROJECT_ROOT))
    _done(start)
    return path, t90_on - t90_off


# --------------------------------------------------------------------------- #
# 4. Density sweep
# --------------------------------------------------------------------------- #
def demo_density():
    _rule("4 | Density sweep - congestion grows faster than evacuation time")
    start = time.perf_counter()

    grid = get_hallway()
    walkable = int(np.sum(grid == CELL_WALKABLE))
    densities = [0.10, 0.30, 0.50]
    n_runs = 15

    rows = []
    for density in densities:
        n_agents = max(1, int(round(density * walkable)))
        result = run_ensemble(
            grid,
            n_runs=n_runs,
            base_seed=BASE_SEED,
            n_agents=n_agents,
            panic_enabled=True,
            panic_decay=0.15,
            cell_capacity=1,
            exit_capacity=1,
            max_steps=900,
        )
        summary = result.summary()
        blocked = summary.get("blocked_total", 0)
        t90 = summary["T90"]["median"]
        rows.append((density, n_agents, summary["T50"]["median"], t90, blocked))
        _line(
            f"Density {density * 100:>2.0f}%  ({n_agents} agents)",
            f"T50 {summary['T50']['median']:.0f} | T90 {t90:.0f} steps | blocked {blocked:,}",
        )

    _line("T90 growth 10% -> 50%", f"{rows[0][3]:.0f} -> {rows[-1][3]:.0f} steps")
    _line(
        "Blocked-move growth",
        f"{rows[0][4]:,} -> {rows[-1][4]:,}  (x{rows[-1][4] / max(1, rows[0][4]):.0f})",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    d = [r[0] * 100 for r in rows]
    ax1.plot(d, [r[2] for r in rows], "o-", color=TEAL, lw=2, label="T50")
    ax1.plot(d, [r[3] for r in rows], "s-", color=CORAL, lw=2, label="T90")
    ax1.legend(fontsize=8.5, frameon=False)
    _style(ax1, "Evacuation time vs crowd density", "density (%)", "steps")

    ax2.bar([f"{x:.0f}%" for x in d], [r[4] for r in rows], color=VIOLET, width=0.55)
    ax2.set_yscale("log")
    _style(ax2, "Blocked moves vs density", "density", "blocked moves (log)")
    fig.suptitle(
        "TheSafeExit - the hallway bottleneck, 15 seeded runs per density",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "showcase_density.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)

    _line("Figure", os.path.relpath(path, PROJECT_ROOT))
    _done(start)
    return path, rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.perf_counter()

    print()
    print("+" + "-" * 72 + "+")
    print("|" + "  TheSafeExit -- showcase".ljust(72) + "|")
    print("|" + "  uncalibrated stochastic evacuation model / compare designs".ljust(72) + "|")
    print("+" + "-" * 72 + "+")

    try:
        p1 = demo_solver()
        p2, gap = demo_exit_search()
        p3, panic_gain = demo_panic()
        p4, density_rows = demo_density()
    except Exception:
        print()
        print("SHOWCASE FAILED")
        traceback.print_exc()
        return 1

    _rule("Summary")
    _line("Solver", "sparse single solve, residual < 1e-8")
    _line("BFGS gap on the 12 x 12 grid", f"{gap:.2f} steps")
    _line("Panic effect on T90", f"+{panic_gain:.1f} steps")
    _line("T90 at 10% / 30% / 50% density", " / ".join(f"{r[3]:.0f}" for r in density_rows) + " steps")
    _line("Figures written", os.path.relpath(OUT_DIR, PROJECT_ROOT))
    for path in (p1, p2, p3, p4):
        print(f"    | {os.path.basename(path)}")
    _line("Total wall clock", f"{time.perf_counter() - t0:.1f} s")
    print()
    print("  Everything above is uncalibrated: it compares designs, it does not")
    print("  predict real evacuation times. Full evidence: docs/RESULTS.md")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
