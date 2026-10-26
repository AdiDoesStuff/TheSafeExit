# Current Status Report

Last audited: 2026-10-27, on branch `main`, after Phases 0–3 and the demo packaging.

This document records what is implemented, what the test suite covers, and what
the project does **not** claim. It replaces the earlier audit that predated the
panic-decay, capacity, trait, numerical-engine and experiment work.

## Test suite

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

- **Result:** 283 passed, 0 failed.
- **Warning:** one pre-existing Matplotlib `PendingDeprecationWarning` from
  `cmap.set_bad` in `src/heatmap.py`; it does not affect results.
- The suite runs headless through `conftest.py` (Agg backend), so no display is
  needed.
- Fourteen tests cover the web-editor endpoints. They `importorskip` Flask, so
  they are skipped only when `requirements-web.txt` is not installed.

New coverage added in Phases 0–3:

| Test file | What it checks |
|---|---|
| `tests/test_baseline_regression.py` | seeded baseline histories and evacuation times are unchanged |
| `tests/test_invariants.py` | chain rows, capacity limits, seed reproducibility, invariant checker |
| `tests/test_panic_decay.py` | decay reduces the field and remains stable |
| `tests/test_capacity.py` | per-cell and per-exit capacity, off switch preserves baseline |
| `tests/test_traits.py` | traits are reproducible, truncation works, off switch preserves baseline, no global RNG use |
| `tests/test_ensemble.py` | ensemble statistics, bootstrap intervals, censoring, paired differences |
| `tests/test_maps.py` | time-to-exit, expected-visits and blocked-moves maps |
| `tests/test_solvers_phase2.py` | sparse/LU/dense/iterative agreement, residuals below 1e-8, MC guards, SVD diagnostics |
| `tests/test_exact_exit_search.py` | hand-checked enumeration optimum, ensemble re-score, heuristic alias |
| `tests/test_experiments_smoke.py` | E1/E2 run end to end, harness dry run, hallway choke cell |
| `tests/test_web_editor.py` | editor page and health, analyze residual, Monte Carlo vs exact, optimizer exact-vs-BFGS, simulate scenario pass-through, ensemble quantiles, localhost-only guard, JSON errors |

## Implemented components

| Layer | Item | Status | File |
|---|---|---|---|
| Engine A | Absorbing chain construction: uniform, rational, panic-mixed `Q`/`R` | Implemented | `src/markov.py` |
| Engine B | Bivariate Gaussian speed/reaction traits, truncated sampling | Implemented | `src/gaussian_sampler.py` |
| Engine C | Wall-aware panic diffusion, CFL guard, decay, explicit `dt` | Implemented | `src/panic.py` |
| Solver layer | One sparse expected-time solve, sparse LU, dense solve | Implemented | `src/solvers.py` |
| Solver layer | CG for the uniform chain; BiCGSTAB/GMRES with ILU for mixed chains | Implemented | `src/solvers.py` |
| Solver layer | Relative residual, condition number, rank-truncated SVD errors | Implemented | `src/solvers.py` |
| Monte Carlo | Vectorised fixed-chain trials, non-absorbing detection, step guard | Implemented | `src/solvers.py` |
| Optimiser | Exhaustive boundary enumeration scored by rational max BFS distance | Implemented | `src/optimization.py` |
| Optimiser | Ensemble T90 re-score of the best five positions, common seeds | Implemented | `src/optimization.py` |
| Optimiser | BFGS search, explicitly labelled a heuristic baseline | Implemented | `src/optimization.py` |
| Simulation | Coupled density → panic → mixing → sampling loop | Implemented | `src/simulation.py` |
| Simulation | Panic decay, per-cell capacity, per-exit capacity, traits, `panic_dt` | Implemented | `src/simulation.py` |
| Simulation | Optional per-step invariant checks | Implemented | `src/simulation.py` |
| Ensemble | T50/T90/T100, bootstrap intervals, censoring flag, paired differences | Implemented | `src/ensemble.py` |
| Ensemble | Summed blocked-move counts and truncated-trait counts | Implemented | `src/ensemble.py` |
| Maps | Time-to-exit, expected-visits and blocked-moves maps | Implemented | `src/heatmap.py` |
| Experiments | E1–E7 harness, quick/full configs, environment record | Implemented | `experiments/` |
| Web editor | Optional Flask editor: analyze, Monte Carlo, optimizer, simulate, ensemble; localhost-only heavy endpoints | Implemented and covered | `demo/web_editor/`, `tests/test_web_editor.py` |
| Demo | One-click launcher, showcase script, editor walkthrough | Implemented | `demo/run_demo.bat`, `demo/showcase.py`, `demo/README.md` |

## Numerical engine

The default fixed-chain analysis solves

```text
(I - Q) t = 1
```

with one sparse direct solve, so no dense `n x n` matrix is formed. Dense
`solve_expected_time_dense` and the full `solve_fundamental_lu` remain for small
grids and for comparison. Iterative solves return their iteration count and
residual; a solve is called usable only below a `1e-8` relative residual.

The Monte Carlo path rejects chains that cannot reach an exit and raises
`NonAbsorbingChainError` when `max_steps_per_trial` is exceeded. Randomness is
always an explicit `numpy.random.Generator`; no function touches the global
NumPy state.

`rational_max_distance` is exact for the rational steepest-descent chain because
every step reduces the BFS distance by one. `exact_exit_search` is therefore
optimal over the enumerated boundary positions for that objective. The BFGS
search has no such guarantee and is reported as a heuristic baseline.

## Experiments

The harness writes `environment.json`, `config.json`, `results.csv`,
`results.md`, per-experiment CSV/PNG artefacts and a pre-registered question,
method, expectation and failure criterion for each experiment.

| ID | Scope | Headline result (full run, `docs/RESULTS.md`) |
|---|---|---|
| E1 | exact fixed chain vs Monte Carlo | coverage 90% / 97% / 97% at M = 10³ / 10⁴ / 10⁵; slope −0.48 |
| E2 | solver scaling | 100×100 and 150×150 solved by sparse/CG/BiCGSTAB, residual < 1e-8; dense `N` skipped from 80×80 |
| E3 | panic and capacity | simple layout T90 32 → 75 with panic (+43 steps); capacity adds 7.5 steps; frozen t=0 diagnostic optimistic by 6.3 steps |
| E4 | density sweep | T90 69 → 386.5 steps (5% → 60%); outlet queue peak one cell from the choke; smooth curve, no sharp knee |
| E5 | exhaustive vs BFGS exits | enumerated optimum verified on all grids; BFGS gap 0 on 5 small grids; stage-2 changes ≤ 1 step; tau 0.76 |
| E6 | traits on/off and correlation | T90 +1 to +2 steps (2–4.5%); truncation 1–2 of ~4,800 draws; small effect, reported as such |
| E7 | parameter robustness | all six parameters change the ranking at some value; minimum tau 0.55 (γ and capacity) |

The canonical numbers are in `docs/RESULTS.md`, with the raw tables in
`experiments/out/results.md` and its CSVs, written beside the git commit and
library versions. The quick configuration is a smoke test only and is not
evidence for a trend. See `docs/EXPERIMENTS.md` for how to read each output.

## What the project does not claim

- It does not predict real evacuation times and is not calibrated against drill
  data.
- It has no casualty model.
- It is a 2D discrete-grid model, not a continuous crowd-dynamics model.
- Panic is a dimensionless qualitative index, not a physical quantity.
- Design rankings can change when the invented parameters change; E7 measures
  this rather than assuming robustness.
- The exhaustive exit search is optimal only over enumerated boundary positions
  for the rational BFS objective, not for every possible building design.
- `N_t` during a panic-coupled run is a frozen-chain diagnostic, not a forecast.
- The web editor is a demonstration surface: its endpoints are tested, but that
  tests the plumbing, not the model's real-world accuracy.

## Demo

| Command | Result |
|---|---|
| `demo\run_demo.bat` | Starts the editor on `127.0.0.1:5050` and opens the browser |
| `.venv\Scripts\python.exe demo\showcase.py` | Four headline plots plus printed numbers in under a minute, under `experiments/out/showcase/` |
| `.venv\Scripts\python.exe demo\web_editor\app.py` | Manual editor start |

The editor exposes analyze (with residual and solve time), Monte Carlo, exact
vs BFGS exit placement, the coupled panic simulation with T50/T90/T100 and
blocked moves, and a common-seed panic on/off ensemble comparison. Scenario
controls cover panic decay, capacity and traits, each with an off switch.
`demo/README.md` is the walkthrough.

## Remaining work (Phase 4)

- Replace stock slide figures with E1–E7 outputs and add the limitations slide.
- Write the report sections: method, validation, results, limitations,
  reproducibility appendix.
- Prepare viva questions and a mock viva.
- Final freeze: tag `final`, fresh clone, install, run the tests and the full
  experiment suite, and confirm the report numbers match the outputs.

Phases 0–3 and the demo packaging are complete; only presentation and the final
freeze remain.
