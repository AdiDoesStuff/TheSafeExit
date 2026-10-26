# TheSafeExit

TheSafeExit is a stochastic crowd evacuation simulation project. It models a building as a grid, moves agents using absorbing Markov chains, spreads panic with a wall-aware diffusion model, and visualizes evacuation bottlenecks over time. Uncalibrated, for comparing designs.

The project is designed around the mathematics of evacuation rather than only animation. It combines graph search, probability, Markov chains, sparse linear algebra, numerical PDEs, stochastic simulation, and visualization.

## Project Idea

The main feedback loop is:

```text
agent positions -> crowd density -> panic field -> movement probabilities -> new agent positions
```

Agents normally move according to rational shortest-path probabilities. As panic increases, their movement becomes more random. Exits are absorbing states, so once an agent reaches an exit, it is removed from the active simulation.

The project produces:

- evacuation statistics,
- static bottleneck heatmaps,
- time-evolving bottleneck heatmap grids,
- GIF animations of bottleneck evolution,
- bottleneck migration plots,
- geometry comparison demos.

## Current Features

- Grid-based floor plans with walls, walkable cells, and exits.
- Simple room, two-exit room, pillar room, and L-shaped hallway layouts.
- BFS reachability validation.
- BFS exit-distance fields for rational movement.
- Absorbing Markov chain construction using `Q` and `R`.
- Uniform random-walk movement model.
- Rational shortest-path movement model.
- Panic-based convex mixing between rational and random movement.
- Wall-aware panic diffusion PDE with CFL stability checking.
- Sparse single-solve for expected times, `t = (I - Q)^(-1) 1`, with a residual check.
- Iterative Krylov solvers: CG for the symmetric uniform chain, BiCGSTAB/GMRES with ILU for rational and panic-mixed chains.
- Dense LU/SVD kept for small grids and for SVD condition-number diagnostics.
- Monte Carlo guard for non-absorbing chains and a hard per-trial step cap.
- Exhaustive exit-placement search scored by rational worst-case BFS distance, with an ensemble T90 re-score of the finalists.
- Panic decay, per-cell and per-exit capacity, and per-agent speed/reaction-time traits, each behind an off switch.
- Ensemble runner with T50/T90/T100, bootstrap intervals, censoring flags, blocked-move maps and truncated-trait counts.
- E1–E7 experiment harness writing CSV, PNG, configuration and environment records.
- Static fundamental matrix heatmaps.
- Reduced time-evolving heatmap snapshots.
- Multi-panel heatmap sequence plots.
- Synchronized heatmap GIF animation.
- Primary bottleneck migration tracking.
- Per-agent position history for analysis.
- Phase 3b geometry generalization study.
- Standalone multivariate Gaussian agent trait sampler.
- Pytest coverage for floor plans, Markov matrices, panic, solvers, simulation, and heatmap sequences.

## Repository Structure

```text
src/
  floor_plan.py         Floor-plan generation, validation, plotting, distance fields
  markov.py             State maps and Markov transition matrices
  solvers.py            Sparse/dense/iterative expected-time solves, Monte Carlo, SVD diagnostics
  optimization.py       Heuristic (BFGS) and exhaustive exit-placement search
  ensemble.py           Common-seed ensemble runner and paired differences
  heatmap.py            Static bottleneck heatmaps
  heatmap_sequence.py   Time-evolving heatmap grids, GIFs, migration tracking
  gaussian_sampler.py   Multivariate Gaussian behavior sampling (with truncation)
  panic.py              Panic diffusion PDE
  simulation.py         Coupled evacuation simulation

experiments/
  config.py             Seeds, full and quick configurations
  common.py             Output helpers, statistics, plotting
  layouts.py            Named layouts used by the experiments
  e1_exact_vs_mc.py     E1 fixed-chain validation against Monte Carlo
  e2_solver_scaling.py  E2 solver time/memory/residual scaling
  e3_panic_capacity.py  E3 panic and capacity effects
  e4_density_sweep.py   E4 density sweep, jam knee, blocked-move map
  e5_exit_search.py     E5 exhaustive vs BFGS exit placement
  e6_traits.py          E6 traits on/off and correlation
  e7_robustness.py      E7 parameter sweeps and Kendall tau
  run_all.py            Harness: environment record, all experiments, results table

docs/
  SCOPE_AND_CLAIMS.md   What may and may not be claimed
  EXPERIMENTS.md        What E1-E7 ask, how to run them, how to read the outputs

tests/
  test_floor_plan.py
  test_markov.py
  test_solvers.py
  test_solvers_phase2.py
  test_exact_exit_search.py
  test_panic.py
  test_simulation.py
  test_heatmap_sequence.py
  test_experiments_smoke.py

demo/
  mid_sem_demo.py
  demo_phase2.py
  demo_phase3.py
  demo_phase3b.py
  figures/
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

- NumPy
- SciPy
- Matplotlib
- Pytest

## Running Tests

```bash
python -m pytest -q
```

The suite is configured (via `conftest.py`) to use the headless `Agg` backend
automatically, so no environment variables are needed. On Windows using the
included virtual environment:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

The native Tk Matplotlib backend is a known failure on this host; the `Agg`
backend is the supported way to run the suite.

## Running Demos

```bash
python demo/mid_sem_demo.py
python demo/demo_phase2.py
python demo/demo_phase3.py
python demo/demo_phase3b.py
```

Generated figures are saved in:

```text
demo/figures/
```

## Core Model

The floor plan is converted into an absorbing Markov chain:

```text
P = [ Q  R ]
    [ 0  I ]
```

Where:

- `Q` stores transitions between walkable cells.
- `R` stores transitions from walkable cells to exits.
- exits are absorbing states.

For a fixed transition matrix:

```text
N = (I - Q)^(-1)
```

The row sums of `N` give expected evacuation steps from each starting cell. These values are mapped back onto the floor plan to form bottleneck heatmaps.

In the panic-coupled simulation, panic changes the movement matrix over time:

```text
Q_t = (1 - lambda_t) Q_rational + lambda_t Q_uniform
```

where:

```text
lambda_t = clip(beta * panic, 0, 1)
```

So `N_t = (I - Q_t)^(-1)` is a frozen-chain diagnostic: it assumes `Q_t` never changes again. Results are uncalibrated, for comparing designs.

Panic decay (`panic_decay`, default 0) subtracts a fraction of the field each step;
per-cell and per-exit capacity (default `None`) block moves into full cells;
speed/reaction-time traits (default off) change movement budget and start delay.
Each is an off-by-default option, so the original seeded baseline is unchanged
when they are not requested. `tests/test_baseline_regression.py` and the
off-switch tests enforce this.

## Numerical Engine

For a fixed chain the analysis only needs the expected time to absorption,

```text
t = (I - Q)^(-1) 1
```

so the engine never has to form the dense fundamental matrix. `src/solvers.py` provides:

| Function | Method | Typical use |
|---|---|---|
| `solve_expected_time_sparse` | one sparse direct solve (`spsolve`) | default fixed-chain analysis |
| `solve_expected_time_lu` | sparse LU factorisation, reused for one right-hand side | repeated solves of the same chain |
| `solve_expected_time_dense` | dense direct solve | small comparison grids |
| `solve_expected_time_iterative` | CG, BiCGSTAB or GMRES, optional ILU | large chains; CG for the symmetric uniform walk, BiCGSTAB/GMRES for rational or mixed chains |

Each returns a `SolveReport` with the solution vector, the relative residual
`||(I - Q)t - 1|| / ||1||`, wall-clock seconds, iterations and the system size.
A solve is only treated as usable when the residual is below `1e-8`. The full
`N` is still available through `solve_fundamental_lu` for the small-grid maps
and for SVD comparison.

`matrix_condition_number` reports the 2-norm condition number of `I - Q`
(dense SVD below 2500 states, `svds` above), and
`svd_rank_truncation_errors` reports the expected-time error of rank-truncated
reconstructions. On the evacuation chains used here the effective rank is close
to the number of states: truncation is accurate only near full rank. The SVD is
therefore a diagnostic, not a compression method or the default solver; its cost
is cubic in the number of states. E2 plots the spectrum and the truncation
error.

`run_monte_carlo` samples the same fixed chain with an explicit
`numpy.random.Generator`. A graph reachability check rejects chains that cannot
reach an exit, and `max_steps_per_trial` turns an effectively non-absorbing
chain into a `NonAbsorbingChainError` instead of a hang.

### Exit placement

`rational_max_distance` is the maximum BFS distance from a walkable cell to the
nearest exit. For the rational steepest-descent chain the absorption time is
deterministic and equals that distance, so it is an exact and cheap objective.
`exact_exit_search` enumerates every legal boundary position, returns the true
minimum on the enumerated set, and optionally re-scores the best five by
ensemble T90 with common random seeds. `optimize_exit_placement` (also exported
as `heuristic_exit_search`) is the older BFGS search over a rounded
interpolation of that objective. It is kept as a heuristic baseline: it has no
optimality guarantee, and E5 reports any gap rather than hiding it.

## Experiments

`experiments/run_all.py` runs E1–E7 with recorded seeds, writes CSV and PNG
artefacts plus `environment.json`, `config.json`, `results.csv` and
`results.md` under `experiments/out/`.

```bash
python experiments/run_all.py              # full suite
python experiments/run_all.py --quick      # small smoke configuration
python experiments/run_all.py --only E1,E3 # selected experiments
python experiments/run_all.py --dry-run    # environment and plan only
```

| ID | Question | Main outputs |
|---|---|---|
| E1 | Does the exact fixed chain match Monte Carlo? | exact vs MC intervals, error slope, `e1_plot.png` |
| E2 | How do solver time, memory and residual scale? | method comparison, 100x100 feasibility, `e2_plot.png` |
| E3 | What do panic and capacity change? | T50/T90/T100 by setting, paired differences, frozen-chain check |
| E4 | Where does the crowd jam as density rises? | density sweep, descriptive knee, blocked-move map |
| E5 | Is exhaustive exit search better than BFGS? | gaps, ensemble re-score, distance-vs-T90 rank correlation |
| E6 | Do speed and reaction-time traits matter? | traits on/off and correlation sweep, paired differences |
| E7 | Are design rankings robust to the invented parameters? | one-at-a-time sweeps, Kendall tau table |

The quick configuration is for smoke tests only; its small run counts give wide
intervals. `docs/EXPERIMENTS.md` describes each experiment and how to read its
outputs. Every result is uncalibrated and is meant for comparing designs.

## Heatmap Snapshots

The simulation stores reduced heatmap snapshots instead of raw `N_t` matrices. Each snapshot stores:

- timestep,
- 2D heatmap,
- top three bottleneck cells,
- mean panic,
- active-agent count.

`heatmap_sequence.py` can then:

- apply a global color scale across all frames,
- plot a static sequence grid,
- generate a GIF animation,
- track primary bottleneck migration over time.
