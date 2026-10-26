# Scope and Claims Ledger

## What the tool is

A qualitative, comparative evacuation tool for 2D grids. It compares layouts,
exit positions, panic on or off, and density using ensemble statistics. Fixed-chain
analysis is exact and checked by Monte Carlo. Every output is labelled
"uncalibrated, for comparing designs".

## What it is not

It does not predict real evacuation times. It is not calibrated or validated
against drill data. It has no casualty model. It does not replace field-validated
tools.

## Claims table

| Do not say | Say instead |
|---|---|
| "Predicts performance before a building is built" | "Compares candidate layouts under stated, uncalibrated assumptions" |
| "Heatmap of fatal bottlenecks" | "Map of where agents are blocked and queue" |
| "Row sums of N show bottlenecks" | "Row sums of N give expected time to exit from each cell" |
| "Q_t forecast each step" | "Frozen-chain diagnostic at each step; forecasts come from ensemble runs" |
| "Validated via Monte Carlo" | "Fixed chain validated against Monte Carlo" |
| "Panic makes agents collide" | "Panic randomises movement; with capacity this increases blocking" |
| "Sigma links panic, speed and pathfinding" | "Sigma links walking speed and reaction time" |
| "LU / SVD / CG are interchangeable" | "Sparse solve by default; CG for the uniform walk, BiCGSTAB/GMRES for mixed chains; dense SVD as a diagnostic" |
| "100x100 is feasible" | "Sparse and iterative solves reached 100x100 with residual below 1e-8 in E2; dense N did not, and only measured sizes count" |
| "BFGS finds the optimal exit" | "Exhaustive search returns the minimum over enumerated positions; BFGS is a heuristic baseline and E5 reports its gap" |
| "Exact search predicts real evacuations" | "Exact search minimises the stated BFS objective; the ensemble re-score is still uncalibrated" |
| "Agent traits improve evacuation" | "Trait effects are measured under stated settings; they may be small or null" |
| "The parameters are calibrated" | "D, k, gamma, beta, dt and capacity are invented; E7 reports how rankings move when they change" |
| "The quick experiment run proves a trend" | "The full run is the evidence; the quick run is a smoke test with wide intervals" |

## Evidence map

| Statement | Evidence |
|---|---|
| The fixed-chain expected-time solve is correct | `tests/test_solvers_phase2.py` residual and dense-row-sum checks; E1 compares against Monte Carlo |
| Iterative solvers agree with the direct solve | `tests/test_solvers_phase2.py` (CG and BiCGSTAB on 50x50); E2 residuals |
| The exhaustive exit search is optimal on its objective | `tests/test_exact_exit_search.py` on hand-checked 7x7 grids; E5 recomputes the enumerated minimum |
| The BFGS search is heuristic | `optimize_exit_placement` docstring; E5 reports the measured gap, including a zero gap |
| Panic, capacity and density change evacuation time | E3 and E4, with intervals and censoring flags |
| Trait effects are modest or null | E6, with paired intervals |
| Parameter choices can change design rankings | E7 Kendall tau versus the default ranking |

The canonical evidence is the full run recorded in `experiments/out/results.md` and
its CSVs/PNGs, beside `environment.json` with the git commit and library versions.

## Units

Panic is a dimensionless qualitative index. Time is in steps unless cell size and
step length are stated.
