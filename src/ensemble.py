import os
import sys
import csv
import math
import numpy as np
from typing import Any, Dict, List, Optional, Sequence

# ensure importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import EvacuationSimulation


class EnsembleResult:
    def __init__(self, seeds: Sequence[int], base_seed: int, per_run: Dict[str, np.ndarray], max_steps: int):
        self.seeds = np.asarray(seeds, dtype=int)
        self.base_seed = int(base_seed)
        self.max_steps = int(max_steps)
        self.per_run = per_run  # dict of arrays length n_runs

    @property
    def n_runs(self) -> int:
        return len(self.seeds)

    def _quantile_value(self, times: List[int], n: int, p: float) -> float:
        m = int(math.ceil(p * n))
        if len(times) < m:
            return float("nan")
        st = sorted(times)
        return float(st[m - 1])

    def summary(self) -> Dict[str, Any]:
        rng = np.random.default_rng([self.base_seed, 3])
        out: Dict[str, Any] = {}
        for name, p in (("T50", 0.5), ("T90", 0.9), ("T100", 1.0)):
            arr = np.asarray(self.per_run[name], dtype=float)
            cens_key = f"censored_{name}"
            cens = np.asarray(self.per_run.get(cens_key, np.isnan(arr)), dtype=bool)
            vals = arr[~np.isnan(arr)]
            if vals.size > 0:
                med = float(np.median(vals))
                # bootstrap
                boot = np.empty(2000, dtype=float)
                for b in range(2000):
                    sample = rng.choice(vals, size=vals.size, replace=True)
                    boot[b] = np.median(sample)
                lo = float(np.percentile(boot, 2.5))
                hi = float(np.percentile(boot, 97.5))
            else:
                med = float("nan")
                lo = float("nan")
                hi = float("nan")
            out[name] = {
                "median": med,
                "bootstrap_ci": (lo, hi),
                "n_censored": int(cens.sum()),
                "n_observed": int(vals.size),
            }
        out["fraction_finished"] = float(np.mean(self.per_run["finished"]))
        out["fraction_evacuated_mean"] = float(np.mean(self.per_run["fraction_evacuated"]))
        return out

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = list(self.per_run.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["seed"] + keys)
            for i, seed in enumerate(self.seeds):
                w.writerow([int(seed)] + [self.per_run[k][i] for k in keys])


def run_ensemble(grid, n_runs: int = 200, base_seed: int = 0, seeds=None, max_steps: int = 600, **sim_kwargs) -> EnsembleResult:
    if seeds is None:
        seeds = [base_seed + i for i in range(n_runs)]
    seeds = list(seeds)
    if len(seeds) != n_runs:
        raise ValueError("len(seeds) must equal n_runs")

    T50 = np.empty(n_runs, dtype=float)
    T90 = np.empty(n_runs, dtype=float)
    T100 = np.empty(n_runs, dtype=float)
    cens50 = np.zeros(n_runs, dtype=bool)
    cens90 = np.zeros(n_runs, dtype=bool)
    cens100 = np.zeros(n_runs, dtype=bool)
    frac_evac = np.empty(n_runs, dtype=float)
    finished = np.zeros(n_runs, dtype=bool)

    for i, seed in enumerate(seeds):
        sim = EvacuationSimulation(floor_plan=np.asarray(grid), seed=seed, **sim_kwargs)
        res = sim.run(max_steps=max_steps)
        times = list(sim.evacuation_times.values())
        n_agents = sim.n_agents
        t50 = _quantile_static(times, n_agents, 0.5)
        t90 = _quantile_static(times, n_agents, 0.9)
        t100 = _quantile_static(times, n_agents, 1.0)
        T50[i] = t50
        T90[i] = t90
        T100[i] = t100
        cens50[i] = math.isnan(t50)
        cens90[i] = math.isnan(t90)
        cens100[i] = math.isnan(t100)
        frac_evac[i] = len(times) / n_agents if n_agents > 0 else 1.0
        finished[i] = (len(times) == n_agents)

    per_run = {
        "T50": T50,
        "T90": T90,
        "T100": T100,
        "censored_T50": cens50,
        "censored_T90": cens90,
        "censored_T100": cens100,
        "fraction_evacuated": frac_evac,
        "finished": finished,
    }
    return EnsembleResult(seeds=seeds, base_seed=base_seed, per_run=per_run, max_steps=max_steps)


def _quantile_static(times, n, p):
    m = int(math.ceil(p * n))
    if len(times) < m:
        return float("nan")
    return float(sorted(times)[m - 1])


def paired_difference(a: EnsembleResult, b: EnsembleResult, stat: str = "T90") -> Dict[str, Any]:
    if not np.array_equal(a.seeds, b.seeds):
        raise ValueError("seeds differ")
    av = np.asarray(a.per_run[stat], dtype=float)
    bv = np.asarray(b.per_run[stat], dtype=float)
    diff = av - bv
    mask = ~np.isnan(diff)
    vals = diff[mask]
    if vals.size == 0:
        return {"median": float("nan"), "bootstrap_ci": (float("nan"), float("nan"))}
    rng = np.random.default_rng([a.base_seed, 3])
    med = float(np.median(vals))
    boot = np.empty(2000, dtype=float)
    for b_i in range(2000):
        sample = rng.choice(vals, size=vals.size, replace=True)
        boot[b_i] = np.median(sample)
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return {"median": med, "bootstrap_ci": (lo, hi)}