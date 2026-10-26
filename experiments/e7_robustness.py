"""E7 — are design rankings robust to the invented parameters?"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

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
from experiments.layouts import e7_configurations
from src.ensemble import run_ensemble


SPEC = ExperimentSpec(
    id="E7",
    title="Parameter robustness and Kendall tau",
    question="Are conclusions robust to the invented parameters?",
    method=(
        "One-at-a-time sweeps of D, k, gamma, beta, panic dt and capacity. Four "
        "candidate designs are ranked by median ensemble T90 for every value; "
        "Kendall tau compares each ranking with the default-parameter ranking."
    ),
    expected=(
        "Rankings stay stable across the swept range, or the unstable parameter "
        "is named explicitly in the report."
    ),
    failure_criterion=(
        "An unstable parameter is not named, or a ranking is computed from "
        "censored runs without reporting the censoring."
    ),
)

# Auto dt for D=0.2 on a unit grid is 0.9 * 0.5 / (0.2 * 2) = 1.125.
_AUTO_DT = 1.125

PARAM_SPECS: List[Dict[str, Any]] = [
    {
        "name": "D",
        "baseline": "0.20",
        "values": [("0.10", {"D": 0.1}), ("0.20", {"D": 0.2}), ("0.40", {"D": 0.4})],
    },
    {
        "name": "k",
        "baseline": "0.50",
        "values": [("0.25", {"k": 0.25}), ("0.50", {"k": 0.5}), ("1.00", {"k": 1.0})],
    },
    {
        "name": "gamma",
        "baseline": "0.15",
        "values": [
            ("0.05", {"panic_decay": 0.05}),
            ("0.15", {"panic_decay": 0.15}),
            ("0.40", {"panic_decay": 0.40}),
        ],
    },
    {
        "name": "beta",
        "baseline": "0.40",
        "values": [("0.20", {"beta": 0.2}), ("0.40", {"beta": 0.4}), ("0.80", {"beta": 0.8})],
    },
    {
        "name": "dt",
        "baseline": "auto",
        "values": [
            ("auto", {}),
            ("0.5x", {"panic_dt": 0.5 * _AUTO_DT}),
            ("0.25x", {"panic_dt": 0.25 * _AUTO_DT}),
        ],
    },
    {
        "name": "capacity",
        "baseline": "1",
        "values": [
            ("1", {"cell_capacity": 1, "exit_capacity": 1}),
            ("2", {"cell_capacity": 2, "exit_capacity": 2}),
            ("inf", {"cell_capacity": None, "exit_capacity": None}),
        ],
    },
]


def run(cfg: Any, out_dir: str) -> Dict[str, Any]:
    out_dir = ensure_out_dir(out_dir)
    layouts = e7_configurations()
    seeds = cfg.seeds_for(cfg.e7_runs)
    write_experiment_config(
        out_dir,
        SPEC,
        cfg,
        extra={
            "parameters": PARAM_SPECS,
            "configurations": list(layouts),
            "n_agents": int(cfg.e7_agents),
            "max_steps": int(cfg.max_steps),
        },
    )

    base_kwargs = {
        "n_agents": int(cfg.e7_agents),
        "panic_enabled": True,
        "panic_decay": 0.15,
        "cell_capacity": 1,
        "exit_capacity": 1,
    }

    # results[(parameter, value_label, config)] = summary blocks
    results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for spec in PARAM_SPECS:
        for value_label, update in spec["values"]:
            kwargs = dict(base_kwargs)
            kwargs.update(update)
            for config_name, grid in layouts.items():
                progress(
                    f"E7 {spec['name']}={value_label} / {config_name}: "
                    f"{cfg.e7_runs} runs"
                )
                result = run_ensemble(
                    grid,
                    n_runs=int(cfg.e7_runs),
                    base_seed=int(seeds[0]),
                    seeds=seeds,
                    max_steps=int(cfg.max_steps),
                    **kwargs,
                )
                summary = result.summary()
                results[(spec["name"], value_label, config_name)] = {
                    "T50": summary["T50"]["median"],
                    "T90": summary["T90"]["median"],
                    "T100": summary["T100"]["median"],
                    "fraction_finished": summary["fraction_finished"],
                    "blocked_total": summary.get("blocked_total", 0),
                    "censored_T90": summary["T90"]["n_censored"],
                }

    # Compute rankings and Kendall tau against the default value.
    rows: List[Dict[str, Any]] = []
    tau_matrix: List[List[float]] = []
    for spec in PARAM_SPECS:
        param = spec["name"]
        config_names = list(layouts)
        baseline_t90: List[float] = []
        baseline_ranks: List[int] = []
        for value_label, _update in spec["values"]:
            if value_label == spec["baseline"]:
                baseline_t90 = [
                    results[(param, value_label, c)]["T90"] for c in config_names
                ]
                baseline_ranks = rank_values(baseline_t90)
                break
        if not baseline_ranks:
            raise RuntimeError(f"baseline value missing for parameter {param}")

        param_taus: List[float] = []
        for value_label, _update in spec["values"]:
            t90_values = [results[(param, value_label, c)]["T90"] for c in config_names]
            ranks = rank_values(t90_values)
            # tau-b is computed on the raw T90 values so that ties are handled
            # correctly; rank_values is only for the human-readable rank column.
            tau = kendall_tau(t90_values, baseline_t90)
            param_taus.append(tau)
            for config_name, t90, rank_value in zip(config_names, t90_values, ranks):
                block = results[(param, value_label, config_name)]
                rows.append(
                    {
                        "parameter": param,
                        "value": value_label,
                        "config": config_name,
                        "T50": block["T50"],
                        "T90": t90,
                        "T100": block["T100"],
                        "fraction_finished": block["fraction_finished"],
                        "blocked_total": block["blocked_total"],
                        "censored_T90": block["censored_T90"],
                        "rank": rank_value,
                        "baseline_rank": baseline_ranks[config_names.index(config_name)],
                        "kendall_tau_vs_default": tau,
                    }
                )
        tau_matrix.append(param_taus)

    csv_path = write_csv(artefact(out_dir, SPEC.id, "rows.csv"), rows)

    # Plot: tau heatmap and one full sweep (beta) for context.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax = axes[0]
    matrix = np.array(tau_matrix, dtype=float)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(PARAM_SPECS)))
    ax.set_yticklabels([s["name"] for s in PARAM_SPECS])
    max_values = max(len(spec["values"]) for spec in PARAM_SPECS)
    ax.set_xticks(range(max_values))
    ax.set_xticklabels([f"value {i + 1}" for i in range(max_values)], fontsize=8)
    for i, spec in enumerate(PARAM_SPECS):
        for j, (value_label, _update) in enumerate(spec["values"]):
            tau_value = matrix[i, j]
            text = f"{value_label}\n{tau_value:.2f}" if np.isfinite(tau_value) else value_label
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5)
    ax.set_title("Kendall tau vs default ranking", fontsize=11, fontweight="bold")
    ax.set_xlabel("Swept value (actual value label shown in each cell)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    beta_spec = next(s for s in PARAM_SPECS if s["name"] == "beta")
    beta_labels = [v[0] for v in beta_spec["values"]]
    x = np.arange(len(beta_labels))
    for config_name in layouts:
        ys = [results[("beta", label, config_name)]["T90"] for label in beta_labels]
        ax.plot(x, ys, "o-", label=config_name)
    ax.set_xticks(x)
    ax.set_xticklabels(beta_labels)
    style_axes(ax, "T90 vs beta (one-at-a-time)", "beta", "T90 (median steps)")
    ax.legend(fontsize=8)

    fig.suptitle("E7 — parameter robustness of design rankings (uncalibrated)", fontsize=12)
    fig.tight_layout()
    png_path = save_figure(fig, artefact(out_dir, SPEC.id, "plot.png"))

    min_taus: Dict[str, float] = {}
    unstable: List[str] = []
    undetermined: List[str] = []
    for i, spec in enumerate(PARAM_SPECS):
        finite = np.asarray(tau_matrix[i], dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            min_taus[spec["name"]] = float("nan")
            undetermined.append(spec["name"])
            continue
        value = float(np.min(finite))
        min_taus[spec["name"]] = value
        if value < 1.0 - 1e-9:
            unstable.append(spec["name"])
    censored = sum(1 for row in rows if row["censored_T90"])
    if unstable:
        key = (
            f"Rankings changed for {', '.join(unstable)} (minimum tau "
            f"{min(min_taus[n] for n in unstable):.2f}); these parameters must be "
            f"named as limitations. Censored T90 values: {censored}."
        )
    else:
        key = (
            "Rankings were identical (tau = 1) for every one-at-a-time sweep; "
            f"censored T90 values: {censored}."
        )
    if undetermined:
        key += (
            f" Rankings could not be tested for {', '.join(undetermined)} because "
            "too few runs finished."
        )

    passed = len(rows) == len(PARAM_SPECS) * 3 * len(layouts) and all(
        np.isfinite(row["T90"]) for row in rows if row["censored_T90"] == 0
    )
    return {
        "id": SPEC.id,
        "spec": SPEC.to_dict(),
        "passed": bool(passed),
        "key_finding": key,
        "min_tau_by_parameter": min_taus,
        "unstable_parameters": unstable,
        "undetermined_parameters": undetermined,
        "n_censored_rows": int(censored),
        "rows": rows,
        "outputs": [csv_path, png_path],
    }
