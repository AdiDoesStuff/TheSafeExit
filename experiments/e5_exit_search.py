"""E5 — is the exact exit search better than the BFGS heuristic?"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import (
    ExperimentSpec,
    artefact,
    ensure_out_dir,
    kendall_tau,
    progress,
    rank_values,
    save_figure,
    style_axes,
    write_csv,
    write_experiment_config,
)
from experiments.layouts import e5_cases
from src.ensemble import run_ensemble
from src.floor_plan import CELL_WALKABLE
from src.optimization import (
    exact_exit_search,
    optimize_exit_placement,
    rational_max_distance,
)


SPEC = ExperimentSpec(
    id="E5",
    title="Exact vs heuristic exit placement",
    question="Is the exact exit search better than BFGS, and do the rankings agree?",
    method=(
        "Five small grids. Enumerate every legal boundary position, score by the "
        "rational worst-case BFS distance, re-score the top five by ensemble T90 "
        "with common seeds, and compare with the BFGS heuristic result."
    ),
    expected=(
        "The exhaustive search returns the true minimum of the enumerated "
        "objective; BFGS may miss it. Max-distance and ensemble rankings need not "
        "agree, which is why both are reported."
    ),
    failure_criterion=(
        "A claimed optimum that is not the minimum of the enumerated scores, or a "
        "BFGS gap that is hidden by the summary."
    ),
)


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    seeds = cfg.seeds_for(cfg.e5_runs)
    write_experiment_config(
        out_dir,
        SPEC,
        cfg,
        extra={"n_runs": int(cfg.e5_runs), "top_k": int(cfg.e5_top_k)},
    )

    summary_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    first_case = None

    for case in e5_cases():
        name = case["name"]
        rows = int(case["rows"])
        cols = int(case["cols"])
        exit_width = int(case["exit_width"])
        obstacles = case["obstacles"]
        floor_cells = int(rows * cols)
        n_agents = min(40, floor_cells)
        progress(f"E5 {name}: exhaustive search")
        exact = exact_exit_search(
            rows=rows,
            cols=cols,
            exit_width=exit_width,
            obstacles=obstacles,
            ensemble_top_k=int(cfg.e5_top_k),
            n_runs=int(cfg.e5_runs),
            base_seed=int(seeds[0]),
            seeds=seeds,
            max_steps=int(cfg.max_steps),
            sim_kwargs={
                "n_agents": n_agents,
                "panic_enabled": False,
                "cell_capacity": 1,
            },
        )
        progress(f"E5 {name}: BFGS heuristic")
        bfgs = optimize_exit_placement(
            rows=rows,
            cols=cols,
            exit_width=exit_width,
            obstacles=obstacles,
        )
        bfgs_grid = bfgs["best_grid"]
        bfgs_score = rational_max_distance(bfgs_grid)

        bfgs_ensemble = run_ensemble(
            bfgs_grid,
            n_runs=int(cfg.e5_runs),
            base_seed=int(seeds[0]),
            seeds=seeds,
            max_steps=int(cfg.max_steps),
            n_agents=n_agents,
            panic_enabled=False,
            cell_capacity=1,
        )
        bfgs_summary = bfgs_ensemble.summary()

        exact_best = exact["best"]
        exact_best_ensemble = exact_best.get("ensemble")
        ensemble_best = exact["best_by_ensemble"]
        exact_best_t90 = (
            exact_best_ensemble["T90_median"]
            if exact_best_ensemble is not None
            else np.nan
        )
        ensemble_best_t90 = (
            ensemble_best["ensemble"]["T90_median"]
            if ensemble_best is not None
            else np.nan
        )
        bfgs_t90 = bfgs_summary["T90"]["median"]
        gap = float(bfgs_score - exact["best_score"])
        miss = gap > 1e-9
        enumerated_min = min(c["max_distance"] for c in exact["candidates"])
        ensemble_matches_exact = bool(
            ensemble_best is not None
            and ensemble_best["wall"] == exact_best["wall"]
            and ensemble_best["position"] == exact_best["position"]
        )
        summary_rows.append(
            {
                "case": name,
                "rows": rows,
                "cols": cols,
                "exit_width": exit_width,
                "n_candidates": exact["n_candidates"],
                "exact_wall": exact_best["wall"],
                "exact_position": exact_best["position"],
                "exact_score": exact["best_score"],
                "exact_score_is_enumerated_min": bool(
                    abs(exact["best_score"] - enumerated_min) < 1e-12
                ),
                "exact_ensemble_T90": exact_best_t90,
                "bfgs_wall": bfgs["best_wall"],
                "bfgs_position": bfgs["best_position"],
                "bfgs_score": bfgs_score,
                "bfgs_gap": gap,
                "bfgs_missed_optimum": miss,
                "bfgs_ensemble_T90": bfgs_t90,
                "bfgs_T90_minus_exact_T90": (
                    float(bfgs_t90 - exact_best_t90)
                    if np.isfinite(bfgs_t90) and np.isfinite(exact_best_t90)
                    else np.nan
                ),
                "ensemble_best_wall": ensemble_best["wall"]
                if ensemble_best is not None
                else None,
                "ensemble_best_position": ensemble_best["position"]
                if ensemble_best is not None
                else None,
                "ensemble_best_T90": ensemble_best_t90,
                "ensemble_best_matches_exact": ensemble_matches_exact,
                "exact_fraction_finished": exact_best_ensemble["fraction_finished"]
                if exact_best_ensemble is not None
                else np.nan,
                "bfgs_fraction_finished": bfgs_summary["fraction_finished"],
            }
        )

        # Rank agreement among the re-scored finalists.
        finalists = [c for c in exact["candidates"] if "ensemble" in c]
        distance_ranks = rank_values([c["max_distance"] for c in finalists])
        ensemble_ranks = rank_values(
            [c["ensemble"]["T90_median"] for c in finalists]
        )
        tau = (
            kendall_tau(
                [c["max_distance"] for c in finalists],
                [c["ensemble"]["T90_median"] for c in finalists],
            )
            if len(finalists) >= 2
            else float("nan")
        )
        for candidate, d_rank, e_rank in zip(finalists, distance_ranks, ensemble_ranks):
            top_rows.append(
                {
                    "case": name,
                    "wall": candidate["wall"],
                    "position": candidate["position"],
                    "max_distance": candidate["max_distance"],
                    "distance_rank": d_rank,
                    "ensemble_T90": candidate["ensemble"]["T90_median"],
                    "ensemble_rank": e_rank,
                    "fraction_finished": candidate["ensemble"]["fraction_finished"],
                    "kendall_tau_distance_vs_T90": tau,
                }
            )
        if first_case is None:
            first_case = name

    summary_path = write_csv(artefact(out_dir, SPEC.id, "summary.csv"), summary_rows)
    top_path = write_csv(artefact(out_dir, SPEC.id, "top_candidates.csv"), top_rows)

    # Plot: gap per case and the ranking scatter for one case.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = [r["case"] for r in summary_rows]
    gaps = [r["bfgs_gap"] for r in summary_rows]
    colours = ["#e63946" if g > 1e-9 else "#2a9d8f" for g in gaps]
    ax.bar(names, gaps, color=colours)
    ax.axhline(0.0, color="black", lw=1)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    style_axes(ax, "BFGS heuristic gap to exhaustive optimum", "Grid", "Score gap (steps)")
    for i, gap_value in enumerate(gaps):
        ax.annotate(f"{gap_value:.1f}", xy=(i, gap_value), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8)

    ax = axes[1]
    subset = [r for r in top_rows if r["case"] == first_case]
    ax.scatter(
        [r["max_distance"] for r in subset],
        [r["ensemble_T90"] for r in subset],
        s=60,
        color="#00798c",
    )
    for r in subset:
        ax.annotate(
            f"{r['wall'][0]}{r['position']}",
            xy=(r["max_distance"], r["ensemble_T90"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    style_axes(
        ax,
        f"Top finalists on {first_case}: distance vs ensemble",
        "Rational max BFS distance",
        "Ensemble T90 (median)",
    )
    fig.suptitle("E5 — exhaustive search vs the BFGS heuristic (uncalibrated)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    n_missed = sum(1 for r in summary_rows if r["bfgs_missed_optimum"])
    n_ensemble_diff = sum(
        1 for r in summary_rows if not r["ensemble_best_matches_exact"]
    )
    taus = [r["kendall_tau_distance_vs_T90"] for r in top_rows]
    finite_taus = [t for t in taus if np.isfinite(t)]
    key = (
        f"BFGS missed the enumerated optimum on {n_missed}/{len(summary_rows)} grids; "
        f"largest score gap {max(r['bfgs_gap'] for r in summary_rows):.1f} steps. "
        f"The ensemble re-score preferred a different position from the stage-1 "
        f"optimum on {n_ensemble_diff}/{len(summary_rows)} grids."
    )
    if finite_taus:
        key += (
            f" Median distance-vs-T90 Kendall tau among finalists "
            f"{np.median(finite_taus):.2f}."
        )
    passed = all(r["exact_score_is_enumerated_min"] for r in summary_rows)
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(passed),
        "key_finding": key,
        "summary": summary_rows,
        "top_candidates": top_rows,
        "n_bfgs_misses": int(n_missed),
        "n_ensemble_disagreements": int(n_ensemble_diff),
        "outputs": [summary_path, top_path, png_path],
    }
