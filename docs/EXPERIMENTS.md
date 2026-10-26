# Experiments E1–E7

This document describes what each experiment asks, how it is run, and how to read
its outputs. It is the companion to `experiments/out/results.md`, which records
the actual results of the last run together with `environment.json` (Python,
NumPy, SciPy, Matplotlib versions and git commit).

Every experiment states a question, a method, an expectation and a failure
criterion before it runs. Failure criteria are evaluated in code; a failed
criterion is reported as `failed_criterion`, not hidden.

## Running

```bash
python experiments/run_all.py                 # full suite
python experiments/run_all.py --quick         # small smoke configuration
python experiments/run_all.py --only E1,E3    # selected experiments
python experiments/run_all.py --dry-run       # environment, config and plan only
python experiments/run_all.py --list          # experiment ids and titles
```

Outputs are written to `experiments/out/`:

| File | Content |
|---|---|
| `environment.json` | library versions, platform, CPU count, git commit, dirty flag, UTC timestamp |
| `config.json` | the full configuration, including seeds and run counts |
| `results.csv` / `results.md` | status, time, key finding and artefact paths for E1–E7 |
| `eN_config.json` | the pre-registered question, method, expectation and failure criterion |
| `eN_*.csv` | tabular results |
| `eN_*.png` | figures |

The quick configuration exists only to prove the harness works. Its intervals
are wide and its run counts are small; do not treat it as evidence for a trend.

## Seeds and randomness

The seed list is explicit: `SEEDS = range(1000, 1200)` in `experiments/config.py`.
Each experiment takes the first `n` seeds it needs. All randomness goes through
`numpy.random.default_rng` with an explicit seed or a passed `Generator`;
experiments never read or write the global NumPy random state. A reproducibility
test in `tests/test_traits.py` and `tests/test_solvers_phase2.py` checks this.

Paired comparisons (E3, E6) use the same seeds for both arms, so the reported
difference is a paired difference with a bootstrap interval, not a difference of
two unpaired medians.

## E1 — Exact fixed chain vs Monte Carlo

**Question.** Does the exact fixed-chain analysis match Monte Carlo?

**Method.** Three small layouts (simple 8x8, pillar 9x9, two-exit 9x9), ten
evenly spaced start cells, the uniform four-neighbour chain. The exact value is
`t_i = ((I - Q)^(-1) 1)_i`, computed with one sparse solve. Monte Carlo uses
M = 1,000, 10,000 and 100,000 trials and a 95% interval from the sample standard
error.

**Pass rule.** The exact value lies inside the 95% interval for at least 90% of
cells at each M, and the fitted log-log error slope is close to the expected
-1/2.

**Read.** `e1_rows.csv` has one row per (layout, start cell, M). `e1_summary.csv`
aggregates the interval coverage and mean relative error per M. The left panel
of `e1_plot.png` is exact vs Monte Carlo at the largest M; the right panel is
mean relative error against M with a `1/sqrt(M)` reference line.

## E2 — Solver scaling

**Question.** How do solver time, memory and residual scale, and which methods
make 100x100 feasible?

**Method.** Grids 20, 40, 60, 80, 100 and 150 with exit width `max(1, size/10)`.
Four methods: full dense LU (forms `N`), one sparse `spsolve`, CG on the uniform
chain, and BiCGSTAB with ILU on the rational chain. Time and peak Python
allocation are measured with `time.perf_counter` and `tracemalloc`. Dense LU is
skipped when its estimated allocation exceeds 512 MB; the skip is a recorded
row, not a silent omission. A separate SVD diagnostic at the largest size up to
40x40 reports the singular-value spectrum, the condition number of `I - Q`, and
the error of rank-truncated expected times.

**Pass rule.** At least one method finishes 100x100 with relative residual below
`1e-8`, and no method fails. A large residual or a memory skip is reported as
such.

**Read.** `e2_rows.csv` records status (`ok`, `skipped_memory`, `failed`,
`residual_too_large`), seconds, residual, iterations and peak traced memory.
`e2_svd.csv` records the rank-truncation error, residual, smallest kept singular
value and condition number. `e2_plot.png` shows time, residual and the SVD
diagnostic. The honest conclusion is about measured sizes only: dense `N` does
not scale, while the sparse and iterative solves do; rank truncation is accurate
only near full rank, so the SVD is a diagnostic rather than the default solver.

## E3 — Panic and capacity

**Question.** What does panic change, and how does it interact with capacity?

**Method.** Two slide layouts (open 20x20 room, 20x20 room with a central
pillar), 40 agents, 100 common seeds. Three settings: panic off with capacity 1;
panic on with capacity 1; panic on without capacity. Panic-on runs use decay
0.15 so the field reaches a finite level instead of saturating (which would
censor every run). T50, T90 and T100 are reported with bootstrap intervals and
censoring counts.

A separate frozen-chain check compares the rational t=0 diagnostic with actual
single-agent runs: one agent, panic on, capacity off, ten start cells. This tests
the wording that `N_t` is a frozen-chain diagnostic, not a forecast.

**Pass rule.** Every setting finishes at least some runs within `max_steps`;
censored runs are counted separately and never averaged in as finished.

**Read.** `e3_summary.csv` is the main table. `e3_paired.csv` gives paired
differences (panic on minus off; no capacity minus capacity). 
`e3_<layout>__<setting>__runs.csv` holds per-run values. The third panel of
`e3_plot.png` compares frozen t=0 steps with actual single-agent time; points
above the diagonal mean the frozen diagnostic is optimistic.

## E4 — Density sweep and jam knee

**Question.** Where does the crowd jam as density rises on the narrow L-shaped
hallway?

**Method.** Occupancy from 5% to 60% of walkable cells, 40 common seeds per
density, cell capacity 1, exit capacity 1, panic on with decay 0.15. Blocked-move
counts are summed per cell over the ensemble. The one-cell choke is defined
geometrically as the narrowest walkable row in the lower half of the hallway.

**Pass rule.** T90 rises with density and a descriptive knee is found; the
blocked-move queue peak in the outlet region is within two cells of the choke.
The global interior peak may be a different congestion point (the corridor bend),
and that is reported rather than smoothed away.

**Read.** `e4_summary.csv` has T50/T90/T100, intervals, censoring and blocked
totals per density. `e4_runs.csv` has per-run values. `e4_blocked_map.csv` and
`e4_blocked_map.png` locate the queue; `e4_curve.png` shows T90 against density
with the knee marked. The knee is the largest second difference of the T90
curve: descriptive, not a formal changepoint estimate.

## E5 — Exhaustive vs heuristic exit placement

**Question.** Is the exhaustive exit search better than BFGS, and do the
rational-distance and ensemble rankings agree?

**Method.** Five small grids (12x12, 16x16, 14x10, and two 14x14 rooms with
obstacles). Every legal boundary position is enumerated and scored by
`rational_max_distance`, the worst-case BFS distance to the nearest exit. The
best five are re-scored by ensemble T90 with common seeds (panic off, capacity 1,
40 agents). The BFGS heuristic is run on the same grid and its score and
ensemble T90 are compared.

**Pass rule.** The reported exact score is the minimum of the enumerated scores
on every grid. A BFGS gap is reported as measured, including a zero gap.

**Read.** `e5_summary.csv` has the exact and BFGS choices, scores, gap, miss flag
and both ensemble T90 values. `e5_top_candidates.csv` ranks the finalists by
rational distance and by T90 and reports Kendall tau between the two. A zero
gap means BFGS matched the enumerated optimum on these grids; it does not make
BFGS exact in general, and the exhaustive search's guarantee is only over the
enumerated positions and the stated objective.

## E6 — Traits on/off and correlation

**Question.** Do speed and reaction-time traits change evacuation time?

**Method.** Open 20x20 room, 60 agents, capacity 1, panic off to isolate the
trait effect. Traits off is compared with truncated bivariate-Gaussian
speed/reaction-time traits at correlations -0.75, 0 and +0.75, using the same 80
seeds. Truncated draws are counted per ensemble.

**Pass rule.** Every setting runs reproducibly; a null or small effect is
reported as such, not described as an improvement.

**Read.** `e6_summary.csv` has medians, intervals, censoring and truncated-draw
counts. `e6_paired.csv` gives paired differences against traits off.
`e6_plot.png` shows the correlation sweep and the paired differences. The
decision to turn panic off is deliberate: with panic on, the trait effect is
confounded by the panic field and by total censoring at these settings.

## E7 — Parameter robustness

**Question.** Are design rankings robust to the invented parameters?

**Method.** One-at-a-time sweeps of `D`, `k`, `gamma` (panic decay), `beta`,
panic `dt` and capacity, at three values each. Four candidate designs (centred
exit, corner exit, central pillar, central barrier) are ranked by median ensemble
T90 for every value, with 30 common seeds. Kendall tau compares each ranking
with the default-parameter ranking.

**Pass rule.** The rankings are stable, or every unstable parameter is named as
a limitation. Censored T90 values are counted and reported.

**Read.** `e7_rows.csv` has every (parameter, value, design) combination with
T90, rank and tau versus the default. `e7_plot.png` shows the tau matrix and the
`beta` sweep. A tau below 1 means the design ranking changed. This is the
experiment that most directly limits the project's claims: if rankings move with
an invented parameter, the tool can rank designs only under that parameter set.

## Reporting rules

- Report intervals, not only point estimates.
- Report censoring; never average a censored run in as if it finished.
- Report the quick configuration only as a harness smoke test.
- Every figure is labelled "uncalibrated, for comparing designs".
- A null result (for example a small trait effect) is a result, not a failure to
  be tuned away.
- Do not describe the exhaustive search as globally optimal: it is optimal over
  the enumerated boundary positions for the rational BFS objective.
