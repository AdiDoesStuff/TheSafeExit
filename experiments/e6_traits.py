"""E6 — do agent traits matter?"""

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


SPEC = ExperimentSpec(
    id="E6",
    title="Traits on/off and correlation",
    question="Do speed and reaction-time traits change evacuation time?",
    method=(
        "Open 20x20 room, 60 agents, capacity 1, panic off to isolate the trait "
        "effect. Traits off is compared with truncated bivariate-Gaussian "
        "speed/reaction-time traits at correlations -0.75, 0 and +0.75, using "
        "the same seeds throughout."
    ),
    expected=(
        "Traits spread start delays and move budgets; the effect may be small. "
        "T90 and T100 are reported with paired intervals, and a null result is "
        "kept."
    ),
    failure_criterion=(
        "A trait setting fails to reproduce when rerun with the same seed, or a "
        "null effect is described as a real improvement."
    ),
)


def _sigma_for_correlation(rho: float) -> np.ndarray:
    """Covariance with speed variance 0.04, reaction variance 0.01."""
    sd_speed = 0.20
    sd_react = 0.10
    cov = float(rho) * sd_speed * sd_react
    return np.array([[sd_speed**2, cov], [cov, sd_react**2]], dtype=float)


def _summary_row(result, setting: str, rho: float, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    summary = result.summary()
    row: Dict[str, Any] = {
        "setting": setting,
        "correlation": rho,
        "use_traits": kwargs.get("use_traits", False),
        "n_runs": result.n_runs,
        "fraction_finished": summary["fraction_finished"],
        "fraction_evacuated_mean": summary["fraction_evacuated_mean"],
        "blocked_total": summary.get("blocked_total", 0),
        "truncated_traits_total": summary.get("truncated_traits_total", 0),
    }
    for stat in ("T50", "T90", "T100"):
        block = summary[stat]
        row[f"{stat}_median"] = block["median"]
        row[f"{stat}_ci_low"] = block["bootstrap_ci"][0]
        row[f"{stat}_ci_high"] = block["bootstrap_ci"][1]
        row[f"{stat}_n_censored"] = block["n_censored"]
    return row


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    grid = e3_layouts()["simple_20x20"]
    seeds = cfg.seeds_for(cfg.e6_runs)
    write_experiment_config(
        out_dir,
        SPEC,
        cfg,
        extra={
            "n_agents": int(cfg.e6_agents),
            "cell_capacity": 1,
            "panic_enabled": False,
            "correlations": list(cfg.e6_correlations),
        },
    )

    results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    off_kwargs = {
        "n_agents": int(cfg.e6_agents),
        "panic_enabled": False,
        "cell_capacity": 1,
        "use_traits": False,
    }
    progress("E6 traits off")
    off_result = run_ensemble(
        grid, n_runs=int(cfg.e6_runs), base_seed=int(seeds[0]), seeds=seeds,
        max_steps=int(cfg.max_steps), **off_kwargs,
    )
    results["traits_off"] = off_result
    off_result.save_csv(artefact(out_dir, SPEC.id, "traits_off__runs.csv"))
    summary_rows.append(_summary_row(off_result, "traits_off", float("nan"), off_kwargs))

    for rho in cfg.e6_correlations:
        label = f"traits_rho_{rho:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        kwargs = {
            "n_agents": int(cfg.e6_agents),
            "panic_enabled": False,
            "cell_capacity": 1,
            "use_traits": True,
            "trait_sigma": _sigma_for_correlation(float(rho)),
        }
        progress(f"E6 traits on, correlation {rho:+.2f}")
        result = run_ensemble(
            grid, n_runs=int(cfg.e6_runs), base_seed=int(seeds[0]), seeds=seeds,
            max_steps=int(cfg.max_steps), **kwargs,
        )
        results[label] = result
        result.save_csv(artefact(out_dir, SPEC.id, f"{label}__runs.csv"))
        summary_rows.append(_summary_row(result, label, float(rho), kwargs))

    paired_rows: List[Dict[str, Any]] = []
    for rho in cfg.e6_correlations:
        label = f"traits_rho_{rho:+.2f}".replace("+", "p").replace("-", "m").replace(".", "_")
        for stat in ("T50", "T90", "T100"):
            diff = paired_difference(results[label], off_result, stat=stat)
            paired_rows.append(
                {
                    "correlation": float(rho),
                    "statistic": stat,
                    "median_difference": diff["median"],
                    "ci_low": diff["bootstrap_ci"][0],
                    "ci_high": diff["bootstrap_ci"][1],
                }
            )

    summary_path = write_csv(artefact(out_dir, SPEC.id, "summary.csv"), summary_rows)
    paired_path = write_csv(artefact(out_dir, SPEC.id, "paired.csv"), paired_rows)

    # Plot: T90/T100 by correlation and paired differences.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    on_rows = [r for r in summary_rows if r["use_traits"]]
    xs = [r["correlation"] for r in on_rows]
    for stat, colour in (("T90", "#e63946"), ("T100", "#6a4c93")):
        ys = [r[f"{stat}_median"] for r in on_rows]
        lo = [r[f"{stat}_ci_low"] for r in on_rows]
        hi = [r[f"{stat}_ci_high"] for r in on_rows]
        yerr = [
            [max(0.0, y - l) if np.isfinite(y) and np.isfinite(l) else 0.0 for y, l in zip(ys, lo)],
            [max(0.0, h - y) if np.isfinite(y) and np.isfinite(h) else 0.0 for y, h in zip(ys, hi)],
        ]
        ax.errorbar(xs, np.nan_to_num(ys, nan=0.0), yerr=np.array(yerr), fmt="o-",
                    capsize=3, color=colour, label=stat)
    off_row = next(r for r in summary_rows if r["setting"] == "traits_off")
    ax.axhline(off_row["T90_median"], color="#e63946", ls=":", lw=1, alpha=0.7)
    ax.axhline(off_row["T100_median"], color="#6a4c93", ls=":", lw=1, alpha=0.7)
    ax.annotate("traits off (T90 dotted)", xy=(xs[0] if xs else 0, off_row["T90_median"]),
                xytext=(4, 4), textcoords="offset points", fontsize=8)
    style_axes(ax, "Trait effect by speed-reaction correlation", "Correlation", "Steps (median)")
    ax.legend(fontsize=8)

    ax = axes[1]
    width = 0.8 / max(1, len(cfg.e6_correlations))
    x = np.arange(2)
    for i, rho in enumerate(cfg.e6_correlations):
        subset = [r for r in paired_rows if r["correlation"] == float(rho)]
        vals = [next(r for r in subset if r["statistic"] == stat)["median_difference"]
                for stat in ("T90", "T100")]
        errs = []
        for stat in ("T90", "T100"):
            row = next(r for r in subset if r["statistic"] == stat)
            errs.append([row["median_difference"] - row["ci_low"], row["ci_high"] - row["median_difference"]])
        offset = (i - (len(cfg.e6_correlations) - 1) / 2) * width
        ax.bar(x + offset, vals, width=width * 0.95, yerr=np.array(errs).T,
               capsize=3, label=f"rho {rho:+.2f}")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["T90", "T100"])
    style_axes(ax, "Paired difference: traits on - off", "Statistic", "Steps (positive = slower)")
    ax.legend(fontsize=8)
    fig.suptitle("E6 — trait effects with capacity (uncalibrated, for comparing designs)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    default = next(r for r in summary_rows if r["correlation"] == 0.75)
    off = next(r for r in summary_rows if r["setting"] == "traits_off")
    if np.isfinite(default["T90_median"]) and np.isfinite(off["T90_median"]):
        change = 100.0 * (default["T90_median"] - off["T90_median"]) / max(off["T90_median"], 1e-12)
        key = (
            f"At correlation +0.75, traits changed median T90 from "
            f"{off['T90_median']:.0f} to {default['T90_median']:.0f} steps "
            f"({change:+.1f}%); truncated draws per ensemble "
            f"{default['truncated_traits_total']}. A small or null change is reported as such."
        )
    else:
        key = "Trait runs were censored; compare the fraction evacuated by max_steps."

    passed = all(np.isfinite(r["fraction_finished"]) for r in summary_rows)
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(passed),
        "key_finding": key,
        "summary": summary_rows,
        "paired": paired_rows,
        "outputs": [summary_path, paired_path, png_path],
    }
