"""E1 — does the exact fixed-chain analysis match Monte Carlo?"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    ExperimentSpec,
    artefact,
    ensure_out_dir,
    progress,
    save_figure,
    style_axes,
    write_csv,
    write_experiment_config,
)
from experiments.layouts import e1_layouts
from src.markov import build_QR, build_state_map
from src.solvers import run_monte_carlo, solve_expected_time_sparse


SPEC = ExperimentSpec(
    id="E1",
    title="Exact fixed chain vs Monte Carlo",
    question="Does the exact fixed-chain analysis match Monte Carlo?",
    method=(
        "Three layouts, ten evenly spaced start cells, uniform four-neighbour "
        "chain. Compare (I-Q)^-1 1 with the Monte Carlo mean at M = 1e3, 1e4, "
        "1e5; 95% intervals use the sample standard error."
    ),
    expected=(
        "The exact value lies inside the 95% interval for at least 90% of cells "
        "and the relative error falls like 1/sqrt(M)."
    ),
    failure_criterion=(
        "Fewer than 90% of cells contain the exact value, or the fitted error "
        "slope is far from -0.5."
    ),
)


def _start_indices(n_transient: int, n_cells: int) -> List[int]:
    n = max(1, min(int(n_cells), n_transient))
    idx = np.unique(np.linspace(0, n_transient - 1, n).round().astype(int))
    return [int(i) for i in idx]


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    write_experiment_config(out_dir, SPEC, cfg)

    rows: List[Dict[str, Any]] = []
    layouts = e1_layouts()
    base_seed = int(cfg.seeds[0])

    for layout_index, (name, grid) in enumerate(layouts.items()):
        state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)
        Q, R = build_QR(grid, state_map, n_transient, n_absorbing)
        exact = solve_expected_time_sparse(Q).t
        starts = _start_indices(n_transient, cfg.e1_n_cells)
        progress(f"E1 {name}: {n_transient} transient states, {len(starts)} start cells")

        for start in starts:
            coord = index_to_state[start]
            for M in cfg.e1_m_values:
                rng = np.random.default_rng([base_seed, layout_index, start, int(M)])
                result = run_monte_carlo(
                    Q, R, start_state_idx=start, M=int(M), rng=rng
                )
                mean = float(result["mean"])
                se = float(result["std_error"])
                z = 1.96
                lo = mean - z * se
                hi = mean + z * se
                exact_value = float(exact[start])
                rows.append(
                    {
                        "layout": name,
                        "start_state": int(start),
                        "start_row": int(coord[0]),
                        "start_col": int(coord[1]),
                        "M": int(M),
                        "exact_steps": exact_value,
                        "mc_mean": mean,
                        "mc_std_error": se,
                        "ci95_low": lo,
                        "ci95_high": hi,
                        "inside_ci": bool(lo <= exact_value <= hi),
                        "relative_error": abs(mean - exact_value) / max(exact_value, 1e-12),
                    }
                )

    # Aggregate the pass criterion.
    m_values = sorted({int(r["M"]) for r in rows})
    summary_rows: List[Dict[str, Any]] = []
    all_pass = True
    for M in m_values:
        subset = [r for r in rows if r["M"] == M]
        frac = float(np.mean([r["inside_ci"] for r in subset]))
        mean_rel = float(np.mean([r["relative_error"] for r in subset]))
        passed = frac >= 0.90
        all_pass = all_pass and passed
        summary_rows.append(
            {
                "M": M,
                "n_cells": len(subset),
                "fraction_inside_ci": frac,
                "mean_relative_error": mean_rel,
                "pass_90pct": passed,
            }
        )

    # Error slope on the largest-M trend. With fewer than two M values (as in
    # the single-M smoke configuration) the slope is undefined, so report NaN
    # rather than letting polyfit warn about an ill-conditioned fit.
    means = np.array([r["mean_relative_error"] for r in summary_rows], dtype=float)
    ms = np.array([r["M"] for r in summary_rows], dtype=float)
    if len(m_values) >= 2 and np.all(means > 0):
        slope = float(np.polyfit(np.log(ms), np.log(means), 1)[0])
    else:
        slope = float("nan")

    rows_path = write_csv(artefact(out_dir, SPEC.id, "rows.csv"), rows)
    summary_path = write_csv(artefact(out_dir, SPEC.id, "summary.csv"), summary_rows)

    # Plot: exact-vs-MC at the largest M, and error vs M.
    largest_m = m_values[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    ax = axes[0]
    colours = plt.cm.tab10(np.linspace(0, 1, len(layouts)))
    for colour, (name, _grid) in zip(colours, layouts.items()):
        subset = [r for r in rows if r["layout"] == name and r["M"] == largest_m]
        x = [r["exact_steps"] for r in subset]
        y = [r["mc_mean"] for r in subset]
        yerr = 1.96 * np.array([r["mc_std_error"] for r in subset])
        ax.errorbar(x, y, yerr=yerr, fmt="o", ms=5, capsize=3, color=colour, label=name)
    all_vals = [r["exact_steps"] for r in rows] + [r["mc_mean"] for r in rows]
    lo_lim, hi_lim = min(all_vals), max(all_vals)
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], "k--", lw=1, label="y = x")
    style_axes(ax, f"Exact vs Monte Carlo (M = {largest_m:,})", "Exact expected steps", "MC mean ± 95% CI")
    ax.legend(fontsize=8)

    ax = axes[1]
    for colour, (name, _grid) in zip(colours, layouts.items()):
        xs, ys = [], []
        for M in m_values:
            vals = [
                r["relative_error"]
                for r in rows
                if r["layout"] == name and r["M"] == M
            ]
            xs.append(M)
            ys.append(float(np.mean(vals)))
        ax.loglog(xs, ys, "o-", color=colour, label=name)
    ref = np.array(m_values, dtype=float)
    ref_y = max(float(means[0]), 1e-12) * np.sqrt(m_values[0] / ref)
    ax.loglog(ref, ref_y, "k:", lw=1.2, label="1/sqrt(M) reference")
    style_axes(ax, "Monte Carlo error vs trials", "Trials M", "Mean relative error")
    ax.legend(fontsize=8)
    fig.suptitle("E1 — fixed-chain validation (uncalibrated, for comparing designs)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    fractions = ", ".join(
        "{} at M={:,}".format(
            "{:.0%}".format(row["fraction_inside_ci"]), row["M"]
        )
        for row in summary_rows
    )
    key = (
        f"Exact value inside the 95% interval for {fractions}; "
        + (
            f"fitted log-log slope {slope:.2f}."
            if np.isfinite(slope)
            else "slope not estimated (fewer than two M values)."
        )
    )
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(all_pass),
        "key_finding": key,
        "rows": rows,
        "summary": summary_rows,
        "error_slope": slope,
        "outputs": [rows_path, summary_path, png_path],
    }
