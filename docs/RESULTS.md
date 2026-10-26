# TheSafeExit — Results of the Full Experiment Run

This document records the numbers from the canonical full run of E1–E7. All
outputs are **uncalibrated and meant for comparing designs**, not for predicting
real evacuation times.

## Provenance

| Field | Value |
|---|---|
| Run date | 2026-10-26 |
| Command | `python experiments/run_all.py` |
| Configuration | full (not `--quick`) |
| Seeds | 1000–1199 (each experiment takes the first *n* it needs) |
| Git commit | `b3bc488` (tracked tree clean, `git_dirty: false`) |
| Python / NumPy / SciPy / Matplotlib | 3.14.3 / 2.5.2 / 1.18.0 / 3.11.1 |
| Platform | Windows 11, 16 logical CPUs |
| Wall-clock total | ≈ 15.6 min (E1 37 s, E2 6 s, E3 111 s, E4 268 s, E5 104 s, E6 43 s, E7 367 s) |
| Status | 7/7 experiments `ok`, all pre-registered criteria met |

The machine-readable record is `experiments/out/environment.json`; the
per-experiment tables and figures are `experiments/out/eN_*.csv` and
`experiments/out/eN_*.png`. The four untracked files listed in
`environment.json` are the working plan document and three pre-existing demo
figures; none is used by these experiments.

**Determinism check.** The run was repeated from a clean tree after the E4/E5
reporting fixes. Every seed-dependent number below (E1 coverage and errors, E3
and E4 medians, E5 choices, E6 medians, E7 taus) reproduced exactly; only
wall-clock times changed. The two runs were at different commits, so this is a
check on the random streams, not a formal reproducibility test of the analysis
code itself.

## Summary

| ID | Question | Result | Criterion |
|---|---|---|---|
| E1 | Does the exact fixed chain match Monte Carlo? | 90% / 97% / 97% interval coverage at M = 10³ / 10⁴ / 10⁵; slope −0.48 | Pass (≥ 90%, slope ≈ −1/2) |
| E2 | How does the solver scale? | Sparse, CG and BiCGSTAB solved 100×100 and 150×150 with residual < 1e-8; dense LU skipped from 80×80 | Pass |
| E3 | What do panic and capacity change? | Panic raises median T90 from 32 to 75 steps; capacity adds 7.5 steps under panic on the open room | Pass (no censoring) |
| E4 | Where does the crowd jam as density rises? | T90 69 → 386.5 steps (5% → 60%); outlet queue peak one cell from the choke | Pass (outlet peak ≤ 2 cells) |
| E5 | Is exhaustive exit search better than BFGS? | BFGS matched the enumerated optimum on 5/5 small grids; ensemble re-score changed position by at most 1 step | Pass (enumerated optimum verified) |
| E6 | Do traits matter? | T90 +1 to +2 steps (2–4.5%); truncation 1–2 draws in ~4,800 | Pass (reported as small) |
| E7 | Are rankings robust to the invented parameters? | All six parameters changed the design ranking at some value; minimum tau 0.55 (γ and capacity) | Pass as limitation (named) |

## E1 — Exact fixed chain vs Monte Carlo

Three layouts (simple 8×8, pillar 9×9, two-exit 9×9), ten start cells each
(30 cells), uniform four-neighbour chain, exact `t = (I − Q)⁻¹ 1` from one
sparse solve.

| M | Fraction inside the 95% interval | Mean relative error |
|---|---|---|
| 1,000 | 0.90 | 2.41 × 10⁻² |
| 10,000 | 0.967 | 8.42 × 10⁻³ |
| 100,000 | 0.967 | 2.60 × 10⁻³ |

Fitted log-log slope: **−0.48**, against the expected −1/2 for Monte Carlo
error. Coverage at M = 1,000 is exactly the pass threshold, as expected: with
30 cells and a 95% interval, a few misses are normal. This validates the
fixed-chain solve only; the coupled simulation is not validated against data.

## E2 — Solver scaling and SVD role

Exit width `max(1, size/10)`. Dense LU forms the full `N`; sparse is one
`spsolve`; CG runs on the symmetric uniform chain; BiCGSTAB runs on the
non-symmetric rational chain.

| Size | States | Dense LU | Sparse `spsolve` | CG (uniform) | BiCGSTAB (rational) |
|---|---|---|---|---|---|
| 20×20 | 324 | 0.009 s | 0.001 s | 0.005 s, 71 it | 0.001 s, 1 it |
| 60×60 | 3,364 | 0.672 s | 0.007 s | 0.020 s, 263 it | 0.005 s, 1 it |
| 80×80 | 6,084 | skipped (memory cap) | 0.013 s | 0.031 s, 355 it | 0.008 s, 1 it |
| 100×100 | 9,604 | skipped (est. 2,214 MB) | 0.021 s | 0.051 s, 448 it | 0.011 s, 1 it |
| 150×150 | 21,904 | skipped (est. 11,515 MB) | 0.056 s | 0.432 s, 909 it | 0.032 s, 2 it |

All completed solves have relative residual below `1e-8`; the largest is
1.56 × 10⁻¹⁰ (CG at 150×150). BiCGSTAB converges in one or two iterations on
the rational chain because rational moves strictly reduce the BFS distance, so
the system is effectively triangular and the ILU preconditioner is exact — this
is a property of that objective, not a general claim about BiCGSTAB. Peak traced
allocation stays near 5 MB at 150×150 for the sparse and iterative paths.

**SVD diagnostic (40×40, 1,444 states).** Condition number **178.2**. Rank
truncation is useless until nearly full rank: relative expected-time error is
≈ 1.0 at ranks 1–500 and falls to 3.1 × 10⁻¹⁵ only at full rank. The SVD is
therefore a diagnostic for small systems, not a compression method.

**Claim kept:** "100×100 is feasible" applies only to the sparse single solve
and the iterative solvers measured here, with the residuals above.

## E3 — Panic and capacity

40 agents, 100 common seeds per setting, cell capacity 1, exit capacity 1,
panic decay 0.15 when panic is on. No runs were censored.

| Layout | Setting | T50 | T90 | T100 |
|---|---|---|---|---|
| simple 20×20 | panic off | 18.0 | 32.0 | 38.0 |
| simple 20×20 | panic on | 40.0 | 75.0 | 88.5 |
| simple 20×20 | panic on, no capacity | 39.0 | 84.5 | 104.0 |
| pillar 20×20 | panic off | 19.0 | 32.0 | 37.0 |
| pillar 20×20 | panic on | 39.5 | 73.0 | 87.0 |
| pillar 20×20 | panic on, no capacity | 39.0 | 75.0 | 94.5 |

Paired differences (median [95% interval], steps):

| Comparison | Simple T90 | Pillar T90 |
|---|---|---|
| panic on − panic off | +43.0 [42.0, 44.5] | +41.0 [39.0, 43.0] |
| no capacity − capacity | +7.5 [4.0, 12.0] | +3.0 [0.0, 4.0] |

Panic more than doubles the median evacuation time here. Capacity adds a
modest delay on the open room under panic; on the pillar layout the interval
touches zero. The paired design (same seeds) is what makes these differences
interpretable.

The frozen t=0 diagnostic (row sums of `N` at the initial state) underestimated
the actual single-agent time by **6.3 steps** (median) across ten start cells.
Panic grows after t = 0, so the frozen chain is optimistic — consistent with
calling it a diagnostic, not a forecast.

## E4 — Density sweep on the narrow hallway

40 common seeds per density, cell capacity 1, exit capacity 1, panic on with
decay 0.15. All runs finished.

| Density | Agents | T50 | T90 | T100 | Blocked moves |
|---|---|---|---|---|---|
| 5% | 6 | 31.5 | 69.0 | 69.0 | 325 |
| 10% | 13 | 49.0 | 82.0 | 91.0 | 2,657 |
| 20% | 25 | 69.0 | 122.0 | 134.0 | 14,407 |
| 30% | 38 | 102.5 | 188.5 | 205.5 | 46,873 |
| 40% | 50 | 137.5 | 259.5 | 288.5 | 97,774 |
| 50% | 63 | 174.0 | 312.5 | 346.5 | 168,525 |
| 60% | 76 | 205.0 | 386.5 | 422.0 | 268,180 |

T90 rises by 5.6× while blocked-move attempts rise by ~825×, so most of the
early cost is queueing rather than distance. The blocked-move map peaks at
**(5, 23)**, the corridor bend, and second at the outlet. The outlet-region peak
is **(20, 23)**, one cell from the one-cell narrowing **(21, 23)**; the exit
cell itself is **(23, 23)**. The criterion only requires the outlet peak within
two cells of the choke, which holds.

The largest second difference is at **20% density**, but the curve is smooth
(T90 roughly doubles from 20% to 40% and again by 60%). We report this as a
descriptive curvature marker, not a sharp jam threshold.

## E5 — Exhaustive vs heuristic exit placement

Five small grids, every legal boundary position enumerated, stage-1 objective
= rational worst-case BFS distance, stage-2 = ensemble T90 (60 seeds, panic
off, capacity 1, 40 agents).

| Grid | Positions | Exact choice (score) | BFGS choice (score) | Gap | Ensemble winner | Stage-2 gain |
|---|---|---|---|---|---|---|
| simple 12×12 | 36 | top 4 (14) | top 4 (14) | 0 | bottom 4 | 0 |
| simple 16×16 | 52 | top 6 (20) | top 6 (20) | 0 | bottom 6 | −1 |
| rect 14×10 | 36 | left 5 (13) | left 5 (13) | 0 | right 5 | −1 |
| pillar 14×14 | 44 | top 5 (17) | top 5 (17) | 0 | right 5 | −1 |
| barrier 14×14 | 44 | bottom 5 (17) | bottom 5 (17) | 0 | bottom 5 | 0 |

BFGS matched the enumerated optimum on all five grids. That is an honest
negative result for this comparison: on these small, symmetric rooms the
heuristic is not beaten. Its gap is not guaranteed to be zero in general, and
the exhaustive search's guarantee is only over the enumerated positions for the
stated objective.

The stage-2 ensemble chose a different position on 4/5 grids, always a
mirror-image exit (top/bottom, left/right), with a T90 change of at most **1
step (4%)**. The median Kendall tau between the finalists' distance rank and
their T90 rank is **0.76**: the rational objective orders finalists sensibly but
not perfectly.

## E6 — Traits

60 agents, 80 common seeds, capacity 1, panic off to isolate the trait effect.
Truncated bivariate Gaussian speed/reaction time at three correlations.

| Setting | T50 | T90 | T100 | Truncated draws |
|---|---|---|---|---|
| traits off | 24.0 | 44.0 | 53.0 | 0 |
| ρ = −0.75 | 25.0 | 45.0 | 54.0 | 2 |
| ρ = 0.00 | 25.0 | 46.0 | 56.0 | 2 |
| ρ = +0.75 | 25.0 | 46.0 | 54.0 | 1 |

Paired T90 differences against traits off are +1.0 [0.5, 2.0], +2.0 [0.5, 2.0]
and +2.0 [1.0, 2.0] steps. The effect is small and consistently **slower** with
traits on at these settings: the reaction-time start delay costs more than the
speed budget recovers. Truncation is negligible (1–2 of ~4,800 draws). This is
a small effect, not evidence that traits improve evacuation.

## E7 — Parameter robustness

One-at-a-time sweeps, 30 common seeds per cell, four designs ranked by median
T90. Kendall tau is computed on raw T90 values (ties handled) against the
default ranking.

| Parameter | Values swept | Minimum tau | Verdict |
|---|---|---|---|
| D | 0.10 / 0.20 / 0.40 | 0.91 | ranking changed |
| k | 0.25 / 0.50 / 1.00 | 0.80 | ranking changed |
| γ (panic decay) | 0.05 / 0.15 / 0.40 | **0.55** | ranking changed |
| β | 0.20 / 0.40 / 0.80 | 0.80 | ranking changed |
| Δt | auto / 0.5× / 0.25× | 0.91 | ranking changed |
| capacity | 1 / 2 / ∞ | **0.55** | ranking changed |

No E7 run was censored. The practical consequence: the tool can rank designs
**only under a stated parameter set**. γ and capacity are the most consequential
invented parameters, followed by k and β. This is the experiment that most
directly limits the project's claims, and it is reported as a limitation rather
than tuned away.

## What this run supports

- The fixed-chain expected-time solve is correct, within Monte Carlo error (E1).
- Sparse and iterative solves reach 100×100 and 150×150 with residuals below
  1e-8 on this machine; dense `N` does not (E2).
- Panic roughly doubles median evacuation time under stated settings; capacity
  adds a smaller delay (E3).
- Congestion grows much faster than evacuation time as density rises, and the
  queue sits at the corridor bend and the outlet, not only at the door (E4).
- The exhaustive search is optimal over enumerated positions for the rational
  objective; BFGS happened to match on five small grids, and the ensemble
  re-score changed the choice by at most one step (E5).
- Trait effects are small and, here, slightly negative for speed (E6).
- Design rankings are not robust to the invented parameters (E7).

## What this run does not support

- Any real-world evacuation time, casualty estimate or "fatal bottleneck".
- Validation of the coupled simulation against drills or published data.
- A claim that BFGS is generally suboptimal, or that exhaustive search improves
  real designs.
- A claim that panic or traits "improve" outcomes.
- A claim that 100×100 works for dense `N`, or for methods not measured here.
- A claim that the quick configuration (`--quick`) proves any trend.

## Reproducing

```powershell
.\.venv\Scripts\python.exe -m pytest -q                 # 283 passed today; 269 at this run's commit
.\.venv\Scripts\python.exe experiments\run_all.py       # ~16 min, writes experiments\out\
.\.venv\Scripts\python.exe experiments\run_all.py --quick   # ~6 min smoke configuration
```

`experiments/out/` is gitignored; regenerate it from the tagged commit to get
the same seed-dependent numbers. `docs/EXPERIMENTS.md` explains each output
file and how to read it.
