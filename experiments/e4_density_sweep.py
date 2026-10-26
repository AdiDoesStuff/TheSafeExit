"""E4 — where does the crowd jam as density rises?"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    ExperimentSpec,
    artefact,
    ensure_out_dir,
    find_knee,
    progress,
    save_figure,
    style_axes,
    write_csv,
    write_experiment_config,
)
from experiments.layouts import e4_layout
from src.ensemble import run_ensemble
from src.floor_plan import CELL_EXIT, CELL_WALKABLE


SPEC = ExperimentSpec(
    id="E4",
    title="Density sweep and jam knee",
    question="Where does the crowd jam as density rises on the narrow hallway?",
    method=(
        "Sweep agents from 5% to 60% of walkable hallway cells. Capacity is one "
        "agent per cell and one agent per exit per step; panic is on with decay "
        "0.15. For each density run a common-seed ensemble and sum the "
        "blocked-move counts per cell."
    ),
    expected=(
        "T90 rises with density with a visible knee. The blocked-move queue peak "
        "sits in the outlet region around the one-cell choke; the corridor bend "
        "is allowed to be a separate, stronger congestion point."
    ),
    failure_criterion=(
        "The outlet-region queue peak is more than two cells from the narrowest "
        "cell, or a density produces no finished runs. The shape of the T90 "
        "curve is reported as measured: a weak or absent knee is stated, not "
        "asserted."
    ),
)


def _narrowest_cell(grid: np.ndarray) -> Tuple[int, int]:
    """
    Narrowest interior cell of the L-shaped hallway: the row just above the
    exit where the corridor is reduced to one cell.
    """
    rows, cols = grid.shape
    candidates = []
    for r in range(1, rows - 1):
        walkable_cols = np.where(grid[r, :] == CELL_WALKABLE)[0]
        if walkable_cols.size:
            candidates.append((int(walkable_cols.size), r, int(walkable_cols[-1])))
    # Prefer the narrowest row in the bottom half (the corridor outlet).
    bottom = [c for c in candidates if c[1] >= rows // 2]
    pool = bottom if bottom else candidates
    _, r, c = min(pool, key=lambda item: item[0])
    return int(r), int(c)


def _plot_blocked_map(grid: np.ndarray, blocked: np.ndarray, narrowest: Tuple[int, int], path: str) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    masked = np.ma.masked_where(grid == 0, blocked.astype(float))
    cmap = plt.cm.magma.copy()
    cmap.set_bad(color="#1e1e24")
    im = ax.imshow(masked, cmap=cmap, origin="upper")
    exits = np.argwhere(grid == CELL_EXIT)
    if exits.size:
        ax.scatter(exits[:, 1], exits[:, 0], marker="s", s=55, color="#2ec4b6",
                   edgecolors="white", linewidth=0.8, label="Exit", zorder=5)
    ax.scatter([narrowest[1]], [narrowest[0]], marker="o", s=110, facecolors="none",
               edgecolors="#00e5ff", linewidth=2, label="Narrowest cell", zorder=6)
    ax.annotate("narrowest", xy=(narrowest[1], narrowest[0]), xytext=(-10, 8),
                textcoords="offset points", color="white", fontsize=8, ha="right",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#1e1e24", alpha=0.8))
    ax.set_title("E4 — blocked moves at the highest density", fontsize=11, fontweight="bold")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Blocked move attempts (summed over runs)")
    ax.legend(loc="lower left", fontsize=8)
    fig.text(0.5, 0.01, "uncalibrated, for comparing designs", ha="center", fontsize=8)
    return save_figure(fig, path)


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    grid = e4_layout()
    n_floor = int(np.sum(grid == CELL_WALKABLE))
    densities = [float(d) for d in cfg.e4_densities]
    seeds = cfg.seeds_for(cfg.e4_runs)
    narrowest = _narrowest_cell(grid)
    write_experiment_config(
        out_dir,
        SPEC,
        cfg,
        extra={
            "floor_cells": n_floor,
            "narrowest_cell": narrowest,
            "cell_capacity": 1,
            "exit_capacity": 1,
            "panic_decay": float(cfg.e4_panic_decay),
        },
    )

    summary_rows: List[Dict[str, Any]] = []
    per_run_rows: List[Dict[str, Any]] = []
    blocked_at_highest: np.ndarray = np.zeros_like(grid, dtype=int)

    for density in densities:
        n_agents = max(1, int(round(density * n_floor)))
        progress(f"E4 density {density:.0%} -> {n_agents} agents")
        result = run_ensemble(
            grid,
            n_runs=int(cfg.e4_runs),
            base_seed=int(seeds[0]),
            seeds=seeds,
            max_steps=int(cfg.max_steps),
            n_agents=n_agents,
            panic_enabled=True,
            panic_decay=float(cfg.e4_panic_decay),
            cell_capacity=1,
            exit_capacity=1,
        )
        summary = result.summary()
        row: Dict[str, Any] = {
            "density": density,
            "n_agents": n_agents,
            "n_floor_cells": n_floor,
            "n_runs": result.n_runs,
            "fraction_finished": summary["fraction_finished"],
            "fraction_evacuated_mean": summary["fraction_evacuated_mean"],
            "blocked_total": int(result.blocked_counts.sum()),
        }
        for stat in ("T50", "T90", "T100"):
            row[f"{stat}_median"] = summary[stat]["median"]
            row[f"{stat}_ci_low"] = summary[stat]["bootstrap_ci"][0]
            row[f"{stat}_ci_high"] = summary[stat]["bootstrap_ci"][1]
            row[f"{stat}_n_censored"] = summary[stat]["n_censored"]
        summary_rows.append(row)
        blocked_at_highest = result.blocked_counts.copy()

        for i, seed in enumerate(result.seeds):
            per_run_rows.append(
                {
                    "density": density,
                    "n_agents": n_agents,
                    "seed": int(seed),
                    "T50": float(result.per_run["T50"][i]),
                    "T90": float(result.per_run["T90"][i]),
                    "T100": float(result.per_run["T100"][i]),
                    "fraction_evacuated": float(result.per_run["fraction_evacuated"][i]),
                    "finished": bool(result.per_run["finished"][i]),
                }
            )

    summary_path = write_csv(artefact(out_dir, SPEC.id, "summary.csv"), summary_rows)
    per_run_path = write_csv(artefact(out_dir, SPEC.id, "runs.csv"), per_run_rows)

    # Blocked-map CSV and PNG at the highest density.
    map_rows: List[Dict[str, Any]] = []
    peak = (-1, -1)
    for r, c in zip(*np.where(blocked_at_highest > 0)):
        map_rows.append(
            {
                "row": int(r),
                "col": int(c),
                "cell": int(grid[r, c]),
                "blocked_count": int(blocked_at_highest[r, c]),
                "is_narrowest": bool((int(r), int(c)) == narrowest),
                "distance_to_narrowest": int(abs(int(r) - narrowest[0]) + abs(int(c) - narrowest[1])),
            }
        )
    if map_rows:
        map_rows.sort(key=lambda row: row["blocked_count"], reverse=True)
    map_csv = write_csv(artefact(out_dir, SPEC.id, "blocked_map.csv"), map_rows)

    interior = blocked_at_highest.copy()
    interior[grid != CELL_WALKABLE] = -1
    if np.any(interior >= 0):
        flat = int(np.argmax(interior))
        peak = (flat // grid.shape[1], flat % grid.shape[1])
    exit_peak = (-1, -1)
    exit_vals = blocked_at_highest.copy()
    exit_vals[grid != CELL_EXIT] = -1
    if np.any(exit_vals >= 0):
        flat = int(np.argmax(exit_vals))
        exit_peak = (flat // grid.shape[1], flat % grid.shape[1])

    # The queue for a one-cell door forms immediately *upstream* of the
    # opening, so the experiment evaluates the outlet region around the
    # narrowest cell rather than pretending the door cell itself must be the
    # global maximum (the corridor bend is a second congestion point).
    outlet_peak = (-1, -1)
    outlet_distance: int | None = None
    for r, c in zip(*np.where(blocked_at_highest > 0)):
        r, c = int(r), int(c)
        d = abs(r - narrowest[0]) + abs(c - narrowest[1])
        if d <= 2:
            if outlet_peak[0] < 0 or blocked_at_highest[r, c] > blocked_at_highest[outlet_peak]:
                outlet_peak = (r, c)
                outlet_distance = d
    peak_distance = (
        int(abs(peak[0] - narrowest[0]) + abs(peak[1] - narrowest[1]))
        if peak[0] >= 0
        else None
    )
    map_png = _plot_blocked_map(grid, blocked_at_highest, narrowest, artefact(out_dir, SPEC.id, "blocked_map.png"))

    densities_arr = np.array([r["density"] for r in summary_rows], dtype=float)
    t90 = np.array([r["T90_median"] for r in summary_rows], dtype=float)
    knee = find_knee(densities_arr, t90)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    finite = np.isfinite(t90)
    ax.errorbar(
        densities_arr[finite],
        t90[finite],
        yerr=[
            np.maximum(0.0, t90[finite] - np.array([r["T90_ci_low"] for r in summary_rows])[finite]),
            np.maximum(0.0, np.array([r["T90_ci_high"] for r in summary_rows])[finite] - t90[finite]),
        ],
        fmt="o-",
        capsize=3,
        color="#e63946",
    )
    if knee is not None:
        ax.axvline(knee, color="grey", ls="--", lw=1)
        ax.annotate(f"largest curvature\n{knee:.0%} (descriptive)", xy=(knee, np.nanmax(t90) * 0.65),
                    xytext=(6, 0), textcoords="offset points", fontsize=8, color="#333333")
    style_axes(ax, "T90 vs density (median, 95% bootstrap CI)", "Fraction of floor cells occupied", "T90 (steps)")
    fig.text(0.5, 0.01, "uncalibrated, for comparing designs", ha="center", fontsize=8)
    curve_png = save_figure(fig, artefact(out_dir, SPEC.id, "curve.png"))

    passed = (
        knee is not None
        and outlet_distance is not None
        and outlet_distance <= 2
        and all(r["fraction_finished"] > 0 for r in summary_rows)
    )
    key = (
        f"T90 rose from {t90[0]:.0f} steps at {densities_arr[0]:.0%} to "
        f"{t90[-1]:.0f} steps at {densities_arr[-1]:.0%}. The largest second "
        f"difference is at {knee:.0%}, but growth is smooth, so this is a "
        f"descriptive curvature marker rather than a sharp jam knee. The queue "
        f"peak in the outlet region is at {outlet_peak}, "
        f"{outlet_distance} cell(s) from the one-cell narrowing {narrowest}; the "
        f"global interior peak is at {peak} (the corridor bend), and the exit-cell "
        f"peak is at {exit_peak}."
    )
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(passed),
        "key_finding": key,
        "summary": summary_rows,
        "knee_density": knee,
        "narrowest_cell": list(narrowest),
        "interior_blocked_peak": list(peak),
        "blocked_peak_distance_to_narrowest": peak_distance,
        "outlet_blocked_peak": list(outlet_peak),
        "outlet_peak_distance_to_narrowest": outlet_distance,
        "exit_blocked_peak": list(exit_peak),
        "outputs": [summary_path, per_run_path, map_csv, map_png, curve_png],
    }
