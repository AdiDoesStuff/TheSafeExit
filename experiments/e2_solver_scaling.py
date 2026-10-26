"""E2 — how does the numerical engine scale?"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable, Dict, List, Tuple

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
from src.floor_plan import compute_exit_distance_field, get_simple_room
from src.markov import build_QR, build_rational_QR, build_state_map
from src.solvers import (
    expected_time_residual,
    solve_expected_time_iterative,
    solve_expected_time_sparse,
    solve_fundamental_lu,
)


SPEC = ExperimentSpec(
    id="E2",
    title="Solver scaling",
    question="How does the solver scale, and which methods make 100x100 feasible?",
    method=(
        "Grids 20..150 with exit width max(1, size/10). Time and peak Python "
        "allocation are recorded for full dense LU, one sparse spsolve, CG on "
        "the uniform chain, and BiCGSTAB on the rational chain. Dense LU is "
        "skipped when its estimated allocation exceeds the configured cap."
    ),
    expected=(
        "Dense LU grows fastest and eventually cannot fit; the sparse single "
        "solve and iterative methods reach 100x100 with relative residual below "
        "1e-8."
    ),
    failure_criterion=(
        "A method is claimed feasible only if it finished with residual below "
        "1e-8; any larger residual or a memory skip is reported as such."
    ),
)


def _dense_estimate_mb(n: int) -> float:
    # Two dense n x n arrays (identity / N) plus factorisation workspace.
    return float(3.0 * 8.0 * n * n / 1e6)


def _timed(fn: Callable[[], Tuple[np.ndarray, float, int]]) -> Dict[str, Any]:
    """Run fn, return time, peak traced memory and the function's payload."""
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        payload = fn()
        seconds = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return {
        "payload": payload,
        "seconds": float(seconds),
        "peak_mb": float(peak / 1e6),
    }


def _dense_lu_payload(Q):
    N = solve_fundamental_lu(Q)
    t = np.sum(N, axis=1)
    return t, expected_time_residual(Q, t), 1


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    write_experiment_config(out_dir, SPEC, cfg)

    rows: List[Dict[str, Any]] = []
    sizes = [int(s) for s in cfg.e2_sizes]
    for size in sizes:
        grid = get_simple_room(size, size, exit_width=max(1, size // 10))
        state_map, _, n_transient, n_absorbing = build_state_map(grid)
        Q_uniform, _R_uniform = build_QR(grid, state_map, n_transient, n_absorbing)
        dist = compute_exit_distance_field(grid)
        Q_rational, _R_rational = build_rational_QR(
            grid, dist, state_map, n_transient, n_absorbing
        )
        dense_mb = _dense_estimate_mb(n_transient)
        progress(f"E2 {size}x{size}: {n_transient} states")

        jobs: List[Tuple[str, str, float, Callable[[], Tuple[np.ndarray, float, int]]]] = []
        if dense_mb <= cfg.e2_dense_memory_cap_mb:
            jobs.append(("dense_lu_full_N", "uniform", dense_mb, lambda Q=Q_uniform: _dense_lu_payload(Q)))
        else:
            rows.append(
                {
                    "size": size,
                    "n_states": n_transient,
                    "n_absorbing": n_absorbing,
                    "method": "dense_lu_full_N",
                    "chain": "uniform",
                    "status": "skipped_memory",
                    "seconds": np.nan,
                    "residual": np.nan,
                    "iterations": 0,
                    "peak_tracemalloc_mb": np.nan,
                    "estimated_dense_mb": dense_mb,
                    "error": "estimated allocation exceeds cap",
                }
            )

        def sparse_payload(Q=Q_uniform):
            report = solve_expected_time_sparse(Q)
            return report.t, report.residual, report.iterations

        def cg_payload(Q=Q_uniform):
            report = solve_expected_time_iterative(
                Q, method="cg", rtol=cfg.e2_iter_rtol
            )
            return report.t, report.residual, report.iterations

        def bicgstab_payload(Q=Q_rational):
            report = solve_expected_time_iterative(
                Q, method="bicgstab", rtol=cfg.e2_iter_rtol
            )
            return report.t, report.residual, report.iterations

        jobs.extend(
            [
                ("sparse_spsolve", "uniform", np.nan, sparse_payload),
                ("cg_uniform", "uniform", np.nan, cg_payload),
                ("bicgstab_rational", "rational", np.nan, bicgstab_payload),
            ]
        )

        for method, chain, estimate, fn in jobs:
            try:
                measured = _timed(fn)
                t, residual, iterations = measured["payload"]
                status = (
                    "ok"
                    if np.isfinite(residual) and residual < 1e-8 and np.all(np.isfinite(t))
                    else "residual_too_large"
                )
                rows.append(
                    {
                        "size": size,
                        "n_states": n_transient,
                        "n_absorbing": n_absorbing,
                        "method": method,
                        "chain": chain,
                        "status": status,
                        "seconds": measured["seconds"],
                        "residual": float(residual),
                        "iterations": int(iterations),
                        "peak_tracemalloc_mb": measured["peak_mb"],
                        "estimated_dense_mb": estimate,
                        "error": "",
                    }
                )
            except Exception as exc:  # keep the experiment running and record it
                rows.append(
                    {
                        "size": size,
                        "n_states": n_transient,
                        "n_absorbing": n_absorbing,
                        "method": method,
                        "chain": chain,
                        "status": "failed",
                        "seconds": np.nan,
                        "residual": np.nan,
                        "iterations": 0,
                        "peak_tracemalloc_mb": np.nan,
                        "estimated_dense_mb": estimate,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    csv_path = write_csv(artefact(out_dir, SPEC.id, "rows.csv"), rows)

    # Plot time and residual against state count.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colours = {
        "dense_lu_full_N": "#d1495b",
        "sparse_spsolve": "#00798c",
        "cg_uniform": "#edae49",
        "bicgstab_rational": "#6a4c93",
    }
    markers = {
        "dense_lu_full_N": "s",
        "sparse_spsolve": "o",
        "cg_uniform": "^",
        "bicgstab_rational": "D",
    }
    for method in colours:
        subset = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        if not subset:
            continue
        xs = [r["n_states"] for r in subset]
        ys = [r["seconds"] for r in subset]
        axes[0].loglog(xs, ys, marker=markers[method], color=colours[method], label=method)
        axes[1].semilogy(
            xs,
            [max(r["residual"], 1e-18) for r in subset],
            marker=markers[method],
            color=colours[method],
            label=method,
        )
    # Mark skipped dense runs.
    skipped = [r for r in rows if r["status"] == "skipped_memory"]
    for r in skipped:
        axes[0].scatter(
            r["n_states"], cfg.e2_dense_memory_cap_mb / 1000.0,
            marker="x", color=colours["dense_lu_full_N"], s=55,
        )
    axes[1].axhline(1e-8, color="black", ls=":", lw=1.2, label="residual 1e-8")
    size_100 = [r for r in rows if r["size"] == 100]
    if size_100:
        axes[0].axvline(size_100[0]["n_states"], color="grey", ls="--", lw=1, alpha=0.7)
        axes[0].annotate("100x100", xy=(size_100[0]["n_states"], axes[0].get_ylim()[1] * 0.5),
                         xytext=(4, 0), textcoords="offset points", fontsize=8, color="grey")
    style_axes(axes[0], "Solve time", "Transient states", "Seconds (log scale)")
    style_axes(axes[1], "Relative residual", "Transient states", "||(I-Q)t - 1|| / ||1||")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("E2 — numerical engine scaling (uncalibrated, for comparing designs)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    target_size = 100 if 100 in sizes else max(sizes)
    feasible = sorted(
        {
            r["method"]
            for r in rows
            if r["size"] == target_size
            and r["status"] == "ok"
            and r["residual"] < 1e-8
        }
    )
    failed = sorted({r["method"] for r in rows if r["status"] == "failed"})
    passed = bool(feasible) and not failed
    key = (
        f"Methods that finished {target_size}x{target_size} with residual < 1e-8: "
        + (", ".join(feasible) if feasible else "none")
        + (
            f"; dense LU skipped at n >= {min([r['n_states'] for r in skipped], default=0)} states."
            if skipped
            else "; dense LU ran at every tested size."
        )
    )
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": passed,
        "key_finding": key,
        "rows": rows,
        "feasible_at_100x100": feasible,
        "outputs": [csv_path, png_path],
    }
