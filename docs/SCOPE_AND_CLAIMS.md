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
| "LU / SVD / CG" | "Solvers that exist: sparse LU and dense SVD" |
| "100x100 is feasible" | "Only what was measured" |

## Units

Panic is a dimensionless qualitative index. Time is in steps unless cell size and
step length are stated.
