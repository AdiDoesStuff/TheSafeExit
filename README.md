# TheSafeExit

TheSafeExit is a stochastic crowd-evacuation model for 2D floor plans. You draw a
building as a grid, and it compares candidate layouts: where the exits go, how
panic changes the tail of the evacuation, and where the crowd queues as density
rises. Agents move on an absorbing Markov chain; panic spreads as a wall-aware
diffusion field and mixes each movement row between rational and random.

**Every number is uncalibrated.** The tool compares designs under stated
assumptions. It does not predict real evacuation times.

## Quickstart

```bat
REM one click: starts the web editor and opens the browser
demo\run_demo.bat

REM or: print the headline numbers and save four plots (under a minute)
.venv\Scripts\python.exe demo\showcase.py
```

First time only:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-web.txt
```

`requirements.txt` is enough for the model, tests and experiments.
`requirements-web.txt` adds Flask and Flask-CORS for the editor; without it the
editor tests skip and everything else still runs.

## The web editor

`demo\run_demo.bat` serves <http://127.0.0.1:5050>. Draw walls, floor and exits
on the grid, then run any of five analyses:

| Analysis | What it reports |
|---|---|
| **Analyze** | Expected time to exit from every cell, from one sparse solve of `(I − Q)t = 1`; worst case, mean, solver method, relative residual, solve time and unknown count |
| **Monte Carlo** | Sampled absorption times from a chosen cell against the exact fixed-chain value, with standard error and the absolute difference over standard error |
| **Optimize** | Exhaustive search over every legal exit position (guaranteed optimal for the rational BFS objective) beside the BFGS heuristic, with the gap |
| **Simulate Panic** | Coupled density → panic → mixing → sampling loop, heatmap snapshots, T50 / T90 / T100, blocked moves and the largest queue cells |
| **Compare** | Same seeds, panic off vs on: T50 / T90 / T100 medians with 95% bootstrap intervals and the paired difference |

A **Scenario** panel controls panic decay, per-cell and per-exit capacity, and
speed/reaction-time traits. Each option has an off switch; the API defaults
(decay 0, no capacity, traits off) reproduce the original behaviour. Every
response carries the residual, so the numerical claim is visible rather than
implied.

The editor is detailed in [`demo/README.md`](demo/README.md): controls, endpoint
contracts, limits, safety notes and troubleshooting. For a click-by-click run
sheet, see [`demo/DEMO_CHECKLIST.md`](demo/DEMO_CHECKLIST.md).

## Experiments

`experiments/run_all.py` runs E1–E7 from one recorded environment and writes
CSV, PNG, configuration and environment files to `experiments/out/`.

```bash
.venv/Scripts/python.exe experiments/run_all.py              # full suite, about 16 min
.venv/Scripts/python.exe experiments/run_all.py --quick      # smoke run, about 6 min
.venv/Scripts/python.exe experiments/run_all.py --only E1,E3 # subset
.venv/Scripts/python.exe experiments/run_all.py --dry-run    # environment and plan only
```

| ID | Question | Headline result (full run) |
|---|---|---|
| E1 | Does the exact fixed chain match Monte Carlo? | 90% / 97% / 97% interval coverage at M = 10³ / 10⁴ / 10⁵; error slope −0.48 |
| E2 | How do solver time, memory and residual scale? | Sparse, CG and BiCGSTAB solve 100×100 and 150×150 with residual < 1e-8; dense `N` skipped from 80×80 |
| E3 | What do panic and capacity change? | Simple room T90 32 → 75 steps with panic (+43 [42, 44.5]); the frozen t = 0 diagnostic is optimistic by 6.3 steps |
| E4 | Where does the crowd jam as density rises? | T90 69 → 386.5 steps (5% → 60%); the outlet queue peaks one cell from the choke; smooth growth, no sharp knee |
| E5 | Is exhaustive exit search better than BFGS? | BFGS matched the enumerated optimum on 5/5 small grids; ensemble re-score moves the exit by at most one step; τ = 0.76 |
| E6 | Do speed and reaction-time traits matter? | T90 +1 to +2 steps (2–4.5%); 1–2 draws truncated of ~4,800 — small, and reported as small |
| E7 | Are design rankings robust to the invented parameters? | All six parameters change the ranking at some value; minimum τ 0.55 (γ and capacity) |

The quick configuration is a smoke test: its intervals are wide and its run
counts small, so it is not evidence for a trend.

### What each file is

| Path | Contents |
|---|---|
| `experiments/out/results.csv`, `results.md` | Status, time, key finding and artefact paths for E1–E7 |
| `experiments/out/eN_*.csv` | Tabular results, one row per configuration |
| `experiments/out/eN_*.png` | Figures |
| `experiments/out/eN_config.json` | The pre-registered question, method, expectation and failure criterion |
| `experiments/out/environment.json` | Library versions, platform, CPU count, git commit and dirty flag |
| `experiments/out/config.json` | The full configuration, including seeds |
| `experiments/out/showcase/` | The four `demo/showcase.py` plots |
| `docs/RESULTS.md` | Canonical numbers of the last full run, with provenance and caveats |
| `docs/EXPERIMENTS.md` | What each experiment asks and how to read its outputs |

`experiments/out/` is gitignored; regenerate it rather than committing it.

## How it works

```text
agent positions -> crowd density -> panic field -> movement probabilities -> new positions
```

The floor plan becomes an absorbing Markov chain:

```text
P = [ Q  R ]
    [ 0  I ]
```

`Q` holds floor-to-floor transitions and `R` holds floor-to-exit transitions;
exits absorb. For a fixed chain, the row sums of

```text
N = (I − Q)^(-1)
```

are the expected number of steps to leave from each cell. Those values map back
onto the grid as the "time to exit" heatmap.

During a coupled run the matrix changes every step:

```text
Q_t = (1 − λ_t) Q_rational + λ_t Q_uniform
λ_t = clip(β · panic_t, 0, 1)
panic_{t+1} = panic_t + Δt (D ∇² panic_t + k ρ_t − γ panic_t)
```

so `N_t = (I − Q_t)^(-1)` is a **frozen-chain diagnostic**, not a forecast: it
assumes `Q_t` never changes again. Forecasts — T50, T90, T100, blocked moves —
come from seeded ensembles of the actual simulation.

Panic decay `γ` (default 0), per-cell and per-exit capacity (default off) and
per-agent speed/reaction-time traits (default off) are layered on top. With
every option off, seeded histories are identical to the tagged baseline;
`tests/test_baseline_regression.py` and the off-switch tests enforce it.

## Numerical engine

For a fixed chain only

```text
t = (I − Q)^(-1) 1
```

is needed, so the default path never forms the dense fundamental matrix.

| Function | Method | Typical use |
|---|---|---|
| `solve_expected_time_sparse` | one sparse direct solve (`spsolve`) | default fixed-chain analysis |
| `solve_expected_time_lu` | sparse LU factorisation, reused for one right-hand side | repeated solves of the same chain |
| `solve_expected_time_dense` | dense direct solve | small comparison grids |
| `solve_expected_time_iterative` | CG, BiCGSTAB or GMRES, optional ILU | large chains; CG for the symmetric uniform walk, BiCGSTAB/GMRES for rational or mixed chains |

Each returns a `SolveReport` with the solution vector, the relative residual
`‖(I − Q)t − 1‖ / ‖1‖`, wall-clock seconds, iterations and the system size. A
solve is treated as usable only below a `1e-8` residual. The dense `N` remains
available through `solve_fundamental_lu` for small-grid maps and SVD comparison.

`run_monte_carlo` samples the same fixed chain with an explicit
`numpy.random.Generator`. A reachability check rejects chains that cannot reach
an exit, and `max_steps_per_trial` raises `NonAbsorbingChainError` instead of
hanging. No function touches the global NumPy random state.

`matrix_condition_number` reports the 2-norm condition number of `I − Q` (dense
SVD below 2,500 states, `svds` above), and `svd_rank_truncation_errors` reports
the expected-time error of rank-truncated reconstructions. On these chains the
effective rank is close to the number of states, so the SVD is a diagnostic, not
a compression method or the default solver.

### Exit placement

`rational_max_distance` is the maximum BFS distance from a floor cell to the
nearest exit. For the rational steepest-descent chain the absorption time is
deterministic and equals that distance, which makes it an exact and cheap
objective. `exact_exit_search` enumerates every legal boundary position, returns
the true minimum on that set, and optionally re-scores the best five by ensemble
T90 with common seeds.

`optimize_exit_placement` (exported also as `heuristic_exit_search`) is the older
BFGS search over a rounded interpolation of the same objective. It is kept as a
heuristic baseline: it has no optimality guarantee, and E5 reports any gap rather
than hiding it.

## Implemented

- Grid floor plans with walls, floor and exits; five named layouts; BFS
  reachability validation and BFS exit-distance fields.
- Uniform, rational and panic-mixed `Q`/`R`; row-stochastic checks.
- Wall-aware panic diffusion with CFL guard and optional decay.
- Coupled simulation with panic decay, per-cell capacity, per-exit capacity and
  bivariate Gaussian speed/reaction-time traits, each behind an off switch.
- Sparse, LU, dense and Krylov expected-time solves with residual reporting;
  Monte Carlo guard; condition number and SVD diagnostics.
- Exhaustive exit search with ensemble re-scoring, and the BFGS heuristic
  baseline.
- Ensemble runner: T50/T90/T100, bootstrap intervals, censoring flags, paired
  differences, blocked-move counts and truncated-trait counts.
- Time-to-exit, expected-visits and blocked-move maps; heatmap sequences, GIFs
  and bottleneck-migration tracking.
- E1–E7 harness with quick/full configurations and an environment record.
- Flask web editor exposing all of the above, with JSON errors and a
  residual-bearing analysis response.

## Repository structure

```text
src/
  floor_plan.py         Floor-plan generation, validation, plotting, distance fields
  markov.py             State maps and Markov transition matrices
  solvers.py            Sparse/dense/iterative expected-time solves, Monte Carlo, SVD
  optimization.py       Exhaustive and BFGS exit-placement search
  ensemble.py           Common-seed ensemble runner and paired differences
  heatmap.py            Static time-to-exit, expected-visit and blocked-move maps
  heatmap_sequence.py   Time-evolving heatmap grids, GIFs, migration tracking
  gaussian_sampler.py   Bivariate Gaussian behaviour sampling (with truncation)
  panic.py              Panic diffusion PDE
  simulation.py         Coupled evacuation simulation

experiments/
  config.py             Seeds and full/quick configurations
  common.py             Output helpers, statistics, plotting
  layouts.py            Named layouts used by the experiments
  e1_exact_vs_mc.py     Fixed-chain validation against Monte Carlo
  e2_solver_scaling.py  Solver time, memory and residual scaling
  e3_panic_capacity.py  Panic and capacity effects
  e4_density_sweep.py   Density sweep, jam knee, blocked-move map
  e5_exit_search.py     Exhaustive vs BFGS exit placement
  e6_traits.py          Traits on/off and correlation
  e7_robustness.py      Parameter sweeps and Kendall tau
  run_all.py            Harness: environment record, experiments, results table

demo/
  run_demo.bat          One-click launcher: editor + browser
  showcase.py           Headline numbers and four plots, under a minute
  README.md             Editor guide, endpoint contracts, troubleshooting
  DEMO_CHECKLIST.md     Click-by-click run sheet for the demo
  web_editor/           Flask backend, editor page, launcher
  figures/              Earlier phase figures (committed artefacts)
  demo_phase*.py        Earlier phase scripts

docs/
  RESULTS.md            Canonical numbers of the last full run
  EXPERIMENTS.md        What each experiment asks and how to read it
  SCOPE_AND_CLAIMS.md   Claims ledger

tests/                  Floor plans, Markov, panic, solvers, simulation, ensemble,
                        maps, optimisation, experiments smoke, web editor endpoints
```

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

Last full run: **283 passed**, one pre-existing Matplotlib
`PendingDeprecationWarning` from `cmap.set_bad` in `src/heatmap.py`. The suite
runs headless through `conftest.py` (Agg backend), so no display is needed.
Fourteen of the tests exercise the web-editor endpoints and are skipped only when
Flask is not installed.

The native Tk Matplotlib backend is a known failure on this host; Agg is the
supported backend and the one the suite uses.

## What we claim

- Compares candidate layouts, exit positions, panic on/off and density under
  stated assumptions.
- The fixed chain is exact and is validated against Monte Carlo (E1).
- The coupled model is a seeded stochastic simulation with ensemble statistics
  and intervals; rankings are checked for robustness (E7), not assumed.
- The expected-time solve is sparse, residual-checked and benchmarked at stated
  sizes (E2).
- Exhaustive exit search is optimal over enumerated boundary positions for the
  rational BFS objective; BFGS is reported as a heuristic baseline.

## What we do not claim

- Real evacuation times, or accuracy against drill data — nothing is calibrated.
- Any casualty or injury model. Blocked moves mean an agent was refused entry to
  a full cell.
- Continuous crowd dynamics. It is a four-neighbour discrete grid.
- That panic, capacities or traits are physical quantities; panic is a
  dimensionless index and time is in steps.
- That the frozen-chain `N_t` forecasts the run — it is a per-step diagnostic.
- Optimality beyond the enumerated set and the stated objective.
- That every design ranking is robust: E7 finds parameters that change it, and
  names them.

The full ledger, with the wording to use for each claim, is
[`docs/SCOPE_AND_CLAIMS.md`](docs/SCOPE_AND_CLAIMS.md).

## Optional packaging

What remains is presentation, not modelling: slides built from the E1–E7 outputs,
the written report, and a final freeze (tag, fresh clone, install, rerun tests
and experiments). The project also keeps `TheSafeExit Implementation Plan to
Deliver the Promised Project (1).md` for the phase history and the viva questions.
