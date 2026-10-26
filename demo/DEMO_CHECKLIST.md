# Demo checklist

What to open, click and say. This is a separate file on purpose: the README
describes the project, this describes the demo.

## Start

- [ ] Double-click `demo\run_demo.bat`. The "regenerate evidence?" prompt
      defaults to **N**, so the editor opens at once. Press `Y` only if there is
      time (`run_all.py --quick` takes about 6 minutes).
- [ ] The browser should open <http://127.0.0.1:5050>. If not, open it manually.
- [ ] Optional second surface: `.venv\Scripts\python.exe demo\showcase.py` prints
      the headline numbers and saves four plots in under a minute
      (`experiments/out/showcase/`).

## Web editor: the five analyses

The simple 20×20 room is loaded by default.

1. **Analyze**
   - Click Analyze. The expected-time heatmap appears with the **Numerical
     engine** card.
   - Expected: 324 transient states, `J* = 27.00`, mean 13.6, method `spsolve`,
     residual `0` (passes `< 1e-8`), under 1 ms.
   - Say: one sparse solve of `(I − Q)t = 1`; the dense fundamental matrix is
     never formed.
2. **Monte Carlo**
   - Shift-click a floor cell, e.g. `(10, 2)`.
   - Expected: mean 17, exact 17, difference 0, `|difference| / std_error = 0`.
   - Say: the rational chain is deterministic, so sampling must reproduce the
     exact value; the uniform chain agrees within a couple of standard errors.
3. **Optimize**
   - Click Optimize on the 20×20 grid.
   - Expected: exhaustive = BFGS = top wall, position 8, `J = 26.0`,
     gap `0.0 steps`, 68 candidates in about 250 ms.
   - Say: exhaustive search is guaranteed optimal on the enumerated set; BFGS
     is a heuristic baseline and happened to match here.
4. **Simulate Panic**
   - Set **Panic decay = 0.15**, keep **capacity** checked, 30 agents.
   - Expected: T50 / T90 / T100 = **23 / 50 / 58**, 30/30 evacuated, blocked
     moves **143**.
   - Scrub the snapshots and point at the queue near the exit.
5. **Compare panic on / off**
   - With decay 0.15 and capacity on, click Compare.
   - Expected: T90 **27 → 58 steps (Δ +31, +115%)**; blocked moves
     **2,144 → 3,922**; 20 common seeds with 95% bootstrap intervals.
   - Say: same seeds on both arms, so the difference is not noise.

Optional: switch the chain selector to **uniform** and re-run Analyze to
contrast random-walk times with rational times.

## Showcase plots

| File | One-line explanation |
|---|---|
| `experiments/out/showcase/showcase_solver.png` | One sparse solve; CG iterations grow with the symmetric uniform chain |
| `experiments/out/showcase/showcase_exit.png` | Every legal exit position, then finalist T90; BFGS gap 0 here |
| `experiments/out/showcase/showcase_panic.png` | Every seeded run is slower with panic; the tail roughly doubles |
| `experiments/out/showcase/showcase_density.png` | T90 82 → 330 steps; blocked moves explode faster than time |

## Full-run evidence (`experiments/out/`)

| File | One-line explanation |
|---|---|
| `e1_plot.png` | Exact vs Monte Carlo; error falls like `1/sqrt(M)` |
| `e2_plot.png` | Sparse/CG/BiCGSTAB reach 100×100; dense `N` does not; SVD is diagnostic only |
| `e3_plot.png` | Panic doubles T90; the frozen t = 0 diagnostic is optimistic |
| `e4_curve.png` | T90 69 → 386 steps; smooth growth, no sharp knee |
| `e4_blocked_map.png` | The queue sits at the corridor bend and the outlet |
| `e5_plot.png` | BFGS gap 0 on these grids; distance and T90 rank correlate at τ = 0.76 |
| `e6_plot.png` | Traits change T90 by +1 to +2 steps — small, and reported as small |
| `e7_plot.png` | All six parameters can change the ranking; γ and capacity worst (τ = 0.55) |

## Numbers to quote

- Tests: **283 passed**, headless, no flags (14 are editor endpoint tests).
- Seeds: 1000–1199, explicit `Generator`, no global RNG.
- Showcase: BFGS gap 0.0; panic +44 T90 steps; density T90 82 / 186 / 330.
- Full run: panic T90 32 → 75 (+43, 95% CI [42, 44.5]); frozen diagnostic
  optimistic by 6.3 steps.
- Density: T90 69 → 386.5; blocked moves 325 → 268,180.
- Traits: T90 44 → 46 at ρ = +0.75 (+4.5%).
- Solvers: 100×100 in 0.011–0.051 s, residual < 1e-8; dense `N` skipped from
  6,084 states.
- E7: all six parameters change the ranking; minimum τ 0.55 (γ and capacity).

## Safe claims

- "Compares candidate layouts under stated, uncalibrated assumptions."
- "Fixed chain validated against Monte Carlo (E1)."
- "Frozen-chain diagnostic at each step; forecasts come from ensembles."
- "Panic randomises movement; with capacity this increases blocking (E3)."
- "Exhaustive search over legal exit positions, re-scored by ensemble time."
- "Results are comparisons, not predictions."

## Never say

- "Predicts evacuation performance", "fatal bottlenecks", "validated"
  (unqualified), "CG" as if it were the only solver, or "100×100 feasible"
  without naming the method.
- "Blocked moves are injuries." They are refused moves into a full cell.

## If something breaks

- Editor will not start: run `demo\web_editor\launch_editor.bat`; check `.venv`.
- No plots: `.venv\Scripts\python.exe demo\showcase.py`, or
  `experiments\run_all.py --quick` (about 6 min).
- Port 5050 busy: close the old "TheSafeExit Web Editor" window, then restart.
- Analyze reports residual above `1e-8`: that is a real failure — stop and
  report it rather than showing the result.
- Compare takes a few seconds: it runs 20 seeded simulations twice.

## Deliberate deviations from the original demo spec

- The launcher's evidence prompt defaults to **N**, not Y, so the demo opens
  instantly.
- The heavy endpoints use a **localhost-only check** plus origin-restricted CORS
  instead of a `DEMO_TOKEN` environment variable, so no setup is required.
