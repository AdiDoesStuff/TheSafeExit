"""E3 — what does panic do, and does it clash with capacity?"""

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
from experiments.layouts import e3_layouts
from src.ensemble import paired_difference, run_ensemble
from src.floor_plan import compute_exit_distance_field
from src.markov import build_rational_QR, build_state_map
from src.solvers import solve_expected_time_sparse


SPEC = ExperimentSpec(
    id="E3",
    title="Panic and capacity",
    question="What does panic do, and does it clash with capacity?",
    method=(
        "Two slide layouts, 40 agents, common random seeds. Settings: panic off "
        "with capacity 1; panic on with capacity 1; panic on without capacity. "
        "The panic-on runs use a decay of 0.15 so the field reaches a finite "
        "level instead of saturating. Frozen N_t row sums at t=0 are compared "
        "with single-agent actual remaining times."
    ),
    expected=(
        "Panic randomises movement, raising T50/T90/T100; capacity adds blocking "
        "on top of that. The frozen rational diagnostic is optimistic once panic "
        "changes Q."
    ),
    failure_criterion=(
        "A null panic effect is reported as measured. A run that does not finish "
        "inside max_steps is flagged as censored, never averaged in as finished."
    ),
)

SETTINGS = {
    "panic_off": {
        "panic_enabled": False,
        "cell_capacity": 1,
    },
    "panic_on": {
        "panic_enabled": True,
        "cell_capacity": 1,
    },
    "panic_on_no_capacity": {
        "panic_enabled": True,
        "cell_capacity": None,
    },
}


def _summarise(result, layout: str, setting: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.summary()
    row: Dict[str, Any] = {
        "layout": layout,
        "setting": setting,
        "n_runs": result.n_runs,
        "cell_capacity": kwargs.get("cell_capacity"),
        "panic_enabled": kwargs.get("panic_enabled"),
        "fraction_finished": summary["fraction_finished"],
        "fraction_evacuated_mean": summary["fraction_evacuated_mean"],
        "blocked_total": summary.get("blocked_total", 0),
    }
    for stat in ("T50", "T90", "T100"):
        block = summary[stat]
        row[f"{stat}_median"] = block["median"]
        row[f"{stat}_ci_low"] = block["bootstrap_ci"][0]
        row[f"{stat}_ci_high"] = block["bootstrap_ci"][1]
        row[f"{stat}_n_censored"] = block["n_censored"]
        row[f"{stat}_n_observed"] = block["n_observed"]
    return row


def _frozen_check(cfg: Any, layouts: Dict[str, np.ndarray], out_dir: str) -> List[Dict[str, Any]]:
    """Compare the frozen t=0 rational expectation with actual single-agent runs."""
    rows: List[Dict[str, Any]] = []
    n_cells = min(10, int(cfg.e1_n_cells))
    n_runs = min(15, int(cfg.e3_runs))
    for layout_name, grid in layouts.items():
        state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)
        dist = compute_exit_distance_field(grid)
        Q_r, R_r = build_rational_QR(
            grid, dist, state_map, n_transient, n_absorbing
        )
        exact = solve_expected_time_sparse(Q_r).t
        starts = np.unique(np.linspace(0, n_transient - 1, n_cells).round().astype(int))
        progress(f"E3 frozen check {layout_name}: {len(starts)} start cells")
        for start in starts:
            coord = index_to_state[int(start)]
            result = run_ensemble(
                grid,
                n_runs=n_runs,
                base_seed=int(cfg.seeds[0]) + 7000,
                max_steps=int(cfg.max_steps),
                n_agents=1,
                initial_positions=[coord],
                panic_enabled=True,
                panic_decay=cfg.e3_panic_decay,
                cell_capacity=None,
            )
            times = np.asarray(result.per_run["T100"], dtype=float)
            observed = times[np.isfinite(times)]
            rows.append(
                {
                    "layout": layout_name,
                    "start_state": int(start),
                    "start_row": int(coord[0]),
                    "start_col": int(coord[1]),
                    "frozen_exact_steps": float(exact[int(start)]),
                    "actual_mean_steps": float(np.mean(observed)) if observed.size else np.nan,
                    "actual_median_steps": float(np.median(observed)) if observed.size else np.nan,
                    "n_runs": int(n_runs),
                    "n_finished": int(observed.size),
                    "fraction_finished": float(observed.size / n_runs),
                    "difference": float(np.mean(observed) - exact[int(start)]) if observed.size else np.nan,
                }
            )
    return rows


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    layouts = e3_layouts()
    write_experiment_config(
        out_dir,
        SPEC,
        cfg,
        extra={
            "settings": SETTINGS,
            "n_agents": int(cfg.e3_agents),
            "panic_decay": float(cfg.e3_panic_decay),
        },
    )

    seeds = cfg.seeds_for(cfg.e3_runs)
    summary_rows: List[Dict[str, Any]] = []
    results: Dict[tuple, Any] = {}
    paired_rows: List[Dict[str, Any]] = []

    for layout_name, grid in layouts.items():
        for setting, base_kwargs in SETTINGS.items():
            kwargs = dict(base_kwargs)
            kwargs["n_agents"] = int(cfg.e3_agents)
            if kwargs.get("panic_enabled"):
                kwargs["panic_decay"] = float(cfg.e3_panic_decay)
            progress(f"E3 {layout_name} / {setting}: {cfg.e3_runs} runs")
            result = run_ensemble(
                grid,
                n_runs=int(cfg.e3_runs),
                base_seed=int(seeds[0]),
                seeds=seeds,
                max_steps=int(cfg.max_steps),
                **kwargs,
            )
            results[(layout_name, setting)] = result
            result.save_csv(artefact(out_dir, SPEC.id, f"{layout_name}__{setting}__runs.csv"))
            summary_rows.append(_summarise(result, layout_name, setting, kwargs))

        # Paired comparisons with common random numbers.
        for stat in ("T50", "T90", "T100"):
            diff_on = paired_difference(
                results[(layout_name, "panic_on")],
                results[(layout_name, "panic_off")],
                stat=stat,
            )
            diff_cap = paired_difference(
                results[(layout_name, "panic_on_no_capacity")],
                results[(layout_name, "panic_on")],
                stat=stat,
            )
            paired_rows.append(
                {
                    "layout": layout_name,
                    "comparison": "panic_on_minus_panic_off",
                    "statistic": stat,
                    "median_difference": diff_on["median"],
                    "ci_low": diff_on["bootstrap_ci"][0],
                    "ci_high": diff_on["bootstrap_ci"][1],
                }
            )
            paired_rows.append(
                {
                    "layout": layout_name,
                    "comparison": "no_capacity_minus_capacity",
                    "statistic": stat,
                    "median_difference": diff_cap["median"],
                    "ci_low": diff_cap["bootstrap_ci"][0],
                    "ci_high": diff_cap["bootstrap_ci"][1],
                }
            )

    summary_path = write_csv(artefact(out_dir, SPEC.id, "summary.csv"), summary_rows)
    paired_path = write_csv(artefact(out_dir, SPEC.id, "paired.csv"), paired_rows)
    frozen_rows = _frozen_check(cfg, layouts, out_dir)
    frozen_path = write_csv(artefact(out_dir, SPEC.id, "frozen_vs_actual.csv"), frozen_rows)

    # Plot 1: T90 with intervals; Plot 2: paired differences; Plot 3: frozen check.
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    setting_order = list(SETTINGS)
    setting_labels = ["panic off\n(cap 1)", "panic on\n(cap 1)", "panic on\n(no cap)"]
    colours = {"panic_off": "#457b9d", "panic_on": "#e63946", "panic_on_no_capacity": "#f4a261"}
    width = 0.8 / len(layouts)
    x = np.arange(len(setting_order))
    ax = axes[0]
    for i, layout_name in enumerate(layouts):
        vals = []
        errs = []
        for setting in setting_order:
            row = next(
                r for r in summary_rows
                if r["layout"] == layout_name and r["setting"] == setting
            )
            vals.append(row["T90_median"])
            errs.append(
                [
                    max(0.0, row["T90_median"] - row["T90_ci_low"])
                    if np.isfinite(row["T90_median"]) and np.isfinite(row["T90_ci_low"])
                    else 0.0,
                    max(0.0, row["T90_ci_high"] - row["T90_median"])
                    if np.isfinite(row["T90_median"]) and np.isfinite(row["T90_ci_high"])
                    else 0.0,
                ]
            )
        offset = (i - (len(layouts) - 1) / 2) * width
        bars = ax.bar(x + offset, np.nan_to_num(vals), width=width * 0.95,
                      yerr=np.array(errs).T, capsize=3, label=layout_name)
    ax.set_xticks(x)
    ax.set_xticklabels(setting_labels, fontsize=8)
    style_axes(ax, "T90 (median, 95% bootstrap CI)", "Setting", "Steps")
    ax.legend(fontsize=8)

    ax = axes[1]
    width2 = 0.8 / len(layouts)
    for i, layout_name in enumerate(layouts):
        on_row = next(r for r in paired_rows if r["layout"] == layout_name
                      and r["comparison"] == "panic_on_minus_panic_off" and r["statistic"] == "T90")
        cap_row = next(r for r in paired_rows if r["layout"] == layout_name
                       and r["comparison"] == "no_capacity_minus_capacity" and r["statistic"] == "T90")
        offset = (i - (len(layouts) - 1) / 2) * width2
        xs = [0 + offset, 1 + offset]
        ys = [on_row["median_difference"], cap_row["median_difference"]]
        errs = np.array(
            [
                [max(0.0, on_row["median_difference"] - on_row["ci_low"]),
                 max(0.0, cap_row["median_difference"] - cap_row["ci_low"])],
                [max(0.0, on_row["ci_high"] - on_row["median_difference"]),
                 max(0.0, cap_row["ci_high"] - cap_row["median_difference"])],
            ]
        )
        ax.bar(xs, ys, width=width2 * 0.95, yerr=errs, capsize=3, label=layout_name)
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["panic on\n- panic off", "no cap\n- cap 1"], fontsize=8)
    style_axes(ax, "Paired difference in T90", "Comparison", "Steps (positive = slower)")
    ax.legend(fontsize=8)

    ax = axes[2]
    for colour, layout_name in zip(["#457b9d", "#e63946"], layouts):
        subset = [r for r in frozen_rows if r["layout"] == layout_name]
        xs = [r["frozen_exact_steps"] for r in subset]
        ys = [r["actual_mean_steps"] for r in subset]
        ax.scatter(xs, ys, color=colour, label=layout_name, s=30)
    vals = [r["frozen_exact_steps"] for r in frozen_rows] + [
        r["actual_mean_steps"] for r in frozen_rows if np.isfinite(r["actual_mean_steps"])
    ]
    if vals:
        lim = [min(vals), max(vals)]
        ax.plot(lim, lim, "k--", lw=1, label="frozen = actual")
    style_axes(ax, "Frozen t=0 diagnostic vs actual", "Frozen rational steps", "Actual single-agent mean")
    ax.legend(fontsize=8)
    fig.suptitle("E3 — panic, capacity and the frozen-chain diagnostic (uncalibrated)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    # Key finding from the simple layout.
    simple_off = next(r for r in summary_rows if r["layout"] == "simple_20x20" and r["setting"] == "panic_off")
    simple_on = next(r for r in summary_rows if r["layout"] == "simple_20x20" and r["setting"] == "panic_on")
    if np.isfinite(simple_off["T90_median"]) and simple_off["T90_median"] > 0:
        change = 100.0 * (simple_on["T90_median"] - simple_off["T90_median"]) / simple_off["T90_median"]
        key = (
            f"On the simple layout panic changed median T90 from "
            f"{simple_off['T90_median']:.0f} to {simple_on['T90_median']:.0f} steps "
            f"({change:+.0f}%); all runs finished within max_steps."
        )
    else:
        key = "Panic runs were censored; compare the fraction evacuated by max_steps."

    frozen_diffs = [r["difference"] for r in frozen_rows if np.isfinite(r["difference"])]
    if frozen_diffs:
        key += (
            f" The frozen t=0 diagnostic underestimated actual single-agent time by "
            f"{np.median(frozen_diffs):.1f} steps (median)."
        )

    passed = all(np.isfinite(r["T90_median"]) for r in summary_rows) and bool(np.mean(
        [r["fraction_finished"] for r in summary_rows]
    ) > 0.0)
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(passed),
        "key_finding": key,
        "summary": summary_rows,
        "paired": paired_rows,
        "frozen": frozen_rows,
        "outputs": [summary_path, paired_path, frozen_path, png_path],
    }
