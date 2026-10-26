# TheSafeExit: Implementation Plan to Deliver the Promised Project

Oct 26, 2026 · Aditya Vineeth, Krishiv S Iyer, Sibi Chakravarthi, Siddharth Maddali

## Tonight's plan: deadline 12 a.m. (read this first)

We have one evening, so we do the 17.5-day plan in its smallest honest form: reword every claim first, then build only what finishes. Everything after this section is the full plan, kept for the report and for later.

**Assumptions (correct me if wrong).** We start about 7 hours before midnight, all four of us are working, and the deliverable is slides plus a viva or short report. Times below are hours from our start (H0). If we have fewer than 5 hours, do Tier A only.

### What we will and will not deliver tonight

| Tier | What | Why this tier | Rule |
| --- | --- | --- | --- |
| A: must | Reworded slides plus a limitations slide; tagged baseline with 36 tests passing; fixed-chain Monte Carlo check (E1); ensemble runner (T50, T90, T100, censoring flag) | Mostly wording and short scripts on code that already works; removes every claim we cannot defend | Done before any Tier B work starts |
| B: should | Panic decay term; traits wired into the simulation; experiment E3 (panic on vs off, ensemble); sparse expected-time solve with a 100x100 timing | Makes two promised engines real and gives us real numbers for the slides | Each has an off switch that reproduces the baseline |
| C: stretch | Capacity (one agent per cell) with a blocked-move map | Needed for "panic causes clustering" and a true bottleneck map, but it touches the core loop | Hard gate at H4: if it does not pass its tests, drop it and use the fallback wording |
| Cut tonight | CG and BiCGSTAB, exact exit search, E2, E4 to E7, web editor update | Not enough time to build and test | Slides say "LU and SVD" and call exit placement a heuristic |

### Who does what, in order

| Window | Person | Task | Done when |
| --- | --- | --- | --- |
| H0 to H0.5 | Aditya | Commit the current changes on a branch, tag it `baseline-2026-10-26`, set `MPLBACKEND=Agg`, run `pytest -q` | 36 passed; tag exists; everyone pulls it |
| H0 to H1.5 | Krishiv | Rewrite the slides with the Section 8 table: no "predicts", no "fatal", fixed-chain validation, frozen-chain diagnostic, Σ links speed and reaction time, "LU / SVD" only | Every changed slide matches the ledger |
| H0 to H1 | Siddharth | E1: compare exact row sums with Monte Carlo (M = 1e3, 1e4, 1e5) on both slide layouts for about 10 start cells | One CSV and one plot; the exact value is inside the 95% interval for most cells |
| H0.5 to H2 | Aditya | `ensemble.py`: 100 fixed seeds, T50, T90, T100 per run, medians with bootstrap interval, fraction finished by `max_steps` | Same seeds give identical output; a cut-off run is flagged, never averaged in as finished |
| H1.5 to H2.5 | Krishiv | Panic decay: add −γu to the update, γ = 0 by default | With γ = 0 the old tests still pass; with no agents panic falls toward zero |
| H1 to H3 | Sibi | Traits: reaction time becomes a start delay, speed becomes a move budget (cell 0.5 m, step 0.4 s, speed truncated to \[0.3, 2.0\] m/s); off by default | Traits off reproduces the baseline; no agent moves before its delay |
| H1 to H2 | Siddharth | Sparse expected-time solve `spsolve(I−Q, 1)` and a timing at 100x100 | Residual below 1e-8; the measured time goes on the slide, and only that feasibility claim is kept |
| H2 to H4 | Aditya | Capacity: cell holds at most one agent, random order each step, blocked agents stay, blocked-move count per cell | Gate at H4 (below) |
| H2.5 to H4 | Krishiv | Merge decay, then traits (in that order) into `main`; fix conflicts; run the suite | Suite green on `main` |
| H3 to H4 | Sibi | Draft the report skeleton: method, validation, results placeholders, limitations | Headings and limitations text written |
| H4 | All | **Gate:** capacity either passes (no cell over capacity, agent count conserved, infinite capacity reproduces baseline) or is dropped | Decision made out loud, once |
| H4 to H5.5 | Siddharth, Krishiv | E3: both layouts, panic on and off (and capacity on if it passed), 100 seeds each, same seeds across settings | CSV and a figure with T50, T90, T100 and intervals |
| H4 to H5.5 | Sibi | If capacity passed: blocked-move map plus a one-line check that a narrowed corridor shows the highest count at the narrowing. If not: rename the row-sum map "time to exit" everywhere | Map or rename committed |
| H5.5 to H6.5 | Krishiv, Sibi | Paste results into slides and report; write each result as measured, including null results | Every number on a slide comes from a CSV |
| H6 to H7 | Siddharth, Aditya | Fresh clone, install, `pytest -q`, rerun E1 and E3 scripts; write the environment (Python, NumPy, SciPy, commit) beside the outputs | Fresh clone reproduces the files; tag `final` |
| Last 30 min | All | Rehearse the viva answers for what we actually built (below) | Each person can explain one other person's part |

Merge order is fixed to avoid conflicts in `simulation.py`: panic decay first, traits second, capacity last, each rebased on `main`.

### Fallback wording if capacity is dropped

- Slide 7 becomes: "Panic randomises movement; this model has no collisions or capacity, so it cannot show jams."
- The bottleneck output is called "time to exit" and "expected visits", never "fatal bottlenecks".
- The limitations slide adds: "No capacity model; blocking is future work."

### Viva answers we must be able to give tonight

- Exits are absorbing because once an agent leaves it never re-enters; the absorbing block is the identity.
- A row of N sums to expected steps to exit, because N counts expected visits to each state before absorption.
- N at a given step is a frozen-chain diagnostic, not a forecast, because the real Q changes at every step.
- Traits now change who starts late and who moves faster (if Sibi's task passed); otherwise we say plainly they are a sampler only.
- We have not calibrated or validated against real evacuations, so we claim comparisons, not predictions.

### Submit-ready checklist

- [ ] `pytest -q` passes on a fresh clone with no flags.
- [ ] E1 and E3 outputs regenerate from one script with the environment file beside them.
- [ ] No slide says "predicts", "fatal", "validated" (unqualified), "CG", or "100x100 feasible" without a measurement behind it.
- [ ] Limitations slide present.
- [ ] Tag `final` pushed; everyone has the same copy.

## 1. Bottom line

Delivering what the slides promise takes about 17.5 person-days in five phases. Nearly all of it is integration, evidence and honest wording, not new mathematics. The core pipeline already works, but two of the four promised engines (Gaussian traits and collision-style panic coupling) do not yet affect the simulation.

**What we can honestly deliver after this plan.** A qualitative, comparative evacuation tool for 2D grids. It compares layouts, exit positions, panic on or off, and crowd density using ensemble statistics. Its fixed-chain analysis is exact and checked by Monte Carlo, and its numerical engine is benchmarked at stated grid sizes.

**What we will not claim.** That it predicts real evacuation times, that it is calibrated or validated against drill data, that it shows where people would die, or that it replaces field-validated tools. Every output is labelled "uncalibrated, for comparing designs".

**Basis.** This plan rests on the audit dated 2026-10-26 (36 tests pass under the headless Agg backend) and the project slides. Line-level claims are the audit's; the code has not been re-run for this plan. Effort figures are estimates, plus or minus 30%, and assume the team can already read the code.

| Phase | Person-days | What it produces |
| --- | --- | --- |
| 0. Stabilise the baseline | 1 | Tagged baseline, reproducible tests, agreed scope |
| 1. Make the engines real | 5 | Traits, capacity, panic decay, ensemble runner, bottleneck maps |
| 2. Numerical engine and optimiser | 3.5 | Sparse and iterative solvers, scaling benchmark inputs, exact exit search |
| 3. Experiments and evidence | 3 | Experiments E1 to E7 with figures and CSVs from one recorded environment |
| 4. Package and defend | 5 | Slides, report, web editor update, viva preparation, final freeze |
| **Total** | **17.5** |  |

Phases 0 to 2 (9.5 person-days) are the engineering core. With four people in parallel the critical path is about 8 working days, because capacity must land before experiments, and the write-up cannot be finished before the experiments.

**Open question.** The deadline and the form of the deliverable (viva, report, slides, live demo) are not known. Section 9 gives cut orders for 5, 8 and 10 or more working days.

## 2. Gap analysis: promised vs built

Of 13 promised items, 2 hold as written, 3 need rewording only, and 8 need new work. The biggest gaps are traits, collisions and the bottleneck map.

| # | Promise on the slides | Reality (from the audit) | Status | Action | Phase |
| --- | --- | --- | --- | --- | --- |
| 1 | Grid in, expected time and heatmap out | Works end to end for the fixed chain | Holds | Keep | n/a |
| 2 | Absorbing chain with N = (I-Q)^-1; row sums are expected time | Implemented and tested | Holds | Keep | n/a |
| 3 | Monte Carlo validation | Exists, but only for a fixed chain, not the coupled simulation | Reword | Say "fixed-chain validation"; run E1 | 3 |
| 4 | Panic as a diffusion PDE with walls | Works; no decay, so panic only grows while agents are present | Build | Add decay term; label units qualitative | 1 |
| 5 | Row sums of N show bottlenecks | Row sum is expected time from a start cell, so it mostly marks cells far from exits; no capacity, so doors cannot jam | Build | Add capacity; blocked-move map; visit map s0^T N; rename row-sum map "time to exit" | 1 |
| 6 | Gaussian traits (speed, reaction time) drive behaviour | Standalone sampler; the simulation never calls it | Build | Integrate as move budget and start delay | 1 |
| 7 | Panic makes agents collide or cluster | Panic only blends each row toward a random walk; no collisions or capacity | Build | Capacity rule; reword slide | 1 |
| 8 | Time-varying Q\_t with (I-Q\_t)N\_t = I each step | Implemented, but N\_t assumes Q\_t never changes again, so it is a diagnostic, not a forecast | Reword | Call it "frozen-chain diagnostic"; use ensembles for forecasts | 1, 4 |
| 9 | Expected evacuation time from the coupled model | A single stochastic run, no ensemble, no censoring-aware statistic | Build | Ensemble runner with T50, T90, T100 and censoring flag | 1 |
| 10 | LU, SVD and CG | LU and SVD exist; CG does not; the LU path still returns a dense N | Build | Sparse single-solve, CG for the uniform chain, BiCGSTAB or GMRES for the others | 2 |
| 11 | 100x100 grids are computationally feasible | Unproven; a dense N at 100x100 is about 10^8 doubles (roughly 0.8 GB) | Build | Scaling benchmark E2 with a recorded environment | 2 |
| 12 | Exit-placement optimisation | Heuristic BFGS over a rounded interpolation; objective is effectively the longest shortest path | Build | Enumerate all legal positions; use ensemble time as objective | 2 |
| 13 | Predicts performance before a building exists; shows "fatal" bottlenecks | No calibration, no validation against data, no casualty model | Reword | Replace with "compares designs under stated assumptions" | 4 |

The web editor exists but was not promised on the slides, so it is optional work in Phase 4.

## 3. Ground rules

These apply to every task below, so they are stated once.

1. **Every new behaviour has an off switch that reproduces today's results.** Capacity infinite, traits off and panic decay zero must give the same output as the tagged baseline for the same seed. This protects the 36 passing tests and gives a clean ablation for every experiment.
2. **Seeds are explicit.** Every random draw comes from one `numpy.random.Generator` passed in or created from a recorded seed. Comparisons between two configurations use the same seeds (common random numbers), so differences are not noise.
3. **Tests before claims.** A feature counts as done only when a test pins its invariant (listed in each task) and the headless suite is green.
4. **One environment for evidence.** All figures and CSVs come from one script run, with Python, NumPy, SciPy, CPU, RAM and git commit written beside them.
5. **Wording follows evidence.** If a claim has no experiment or test behind it, it is reworded or removed (Section 8).
6. **Qualitative units.** Panic is a dimensionless index; time is in steps unless a cell size and step length are stated. No output is presented as a real-world time.

## 4. Phase plan

### Phase 0: Stabilise the baseline (1 person-day)

The repository is dirty (modified tracked files and untracked files), and the default test command fails on this host. Fix both before building on it.

| ID | Task | Detail | Done when | Days |
| --- | --- | --- | --- | --- |
| 0.1 | Freeze baseline | Review the diff, commit it on a branch, tag `baseline-audit-2026-10-26` | Clean `git status`; tag exists | 0.25 |
| 0.2 | Reproducible tests | `conftest.py` sets the Agg backend and a writable temp directory; add a `requirements-web.txt` with Flask and Flask-CORS; note the Tk failure as a known host issue | `pytest -q` gives 36 passed with no flags on a fresh clone | 0.25 |
| 0.3 | Scope and claims ledger | One page: what we claim, what we do not (Section 8), agreed by all four members | Signed off in the team chat | 0.25 |
| 0.4 | Experiment scaffolding | `experiments/` folder, `run_all.py`, `env_info.py` writing the environment record, fixed seed list | `run_all.py --dry-run` writes the environment file | 0.25 |

### Phase 1: Make the engines real (5 person-days)

This is where the two unreal engines become real. Design details are in Section 5.

| ID | Task | Files | Test that pins it | Days |
| --- | --- | --- | --- | --- |
| 1.1 | Panic decay and input validation. Add the sink term and validate `dx`, `dy`, `k`, `safety_factor` and `n_agents` as positive or non-negative | `panic.py`, `simulation.py` | With `gamma = 0` output matches baseline; with no agents panic decays monotonically to zero; CFL guard still passes | 0.5 |
| 1.2 | Capacity rule. Per-cell capacity and per-exit throughput per step, random arbitration, blocked agents stay put | `simulation.py` | No cell ever exceeds capacity; agent count is conserved; capacity set to infinity reproduces baseline | 1.25 |
| 1.3 | Trait integration. Speed becomes a move budget and reaction time becomes a start delay; the sampler takes a seed | `gaussian_sampler.py`, `simulation.py` | Traits off reproduces baseline; a faster agent finishes no later than a slower one on the same path in a deterministic rational run; no agent moves before its delay | 1.0 |
| 1.4 | Ensemble runner. Run K seeds, return T50, T90, T100 per run, medians and bootstrap CIs across runs, fraction evacuated by `max_steps`, and a censoring flag | new `ensemble.py` | Fixed seeds give identical output; a run cut off by `max_steps` is flagged and never reported as a finished time | 1.0 |
| 1.5 | Bottleneck maps. Count blocked-move events per cell; add the expected-visit map s0^T N for the fixed chain; rename the row-sum map "time to exit" in code, plots and API | `heatmap.py`, `heatmap_sequence.py`, `app.py` | A corridor narrowed to one cell shows its highest blocked-move count at the narrowing; s0^T N sums to the mean expected time over the start distribution | 0.75 |
| 1.6 | Frozen-chain wording. Rename `N_t` outputs and docstrings to "frozen-chain diagnostic" | `simulation.py`, `app.py`, README | Grep finds no remaining claim that `N_t` forecasts the run | 0.25 |
| 1.7 | Invariant test pack. Row-stochastic `Q_t,R_t` assertion in tests, capacity conservation, seed reproducibility | `tests/` | New tests green in the headless suite | 0.25 |

### Phase 2: Numerical engine and optimiser (3.5 person-days)

Phase 2 does not depend on Phase 1 and can start on day 2 in parallel.

| ID | Task | Detail | Done when | Days |
| --- | --- | --- | --- | --- |
| 2.1 | Sparse single-solve | Expected time from every cell is t = (I-Q)^-1 1, one sparse solve, no dense N. Keep the dense N only for small grids and for the maps in 1.5. Return the relative residual ‖(I-Q)t − 1‖/‖1‖ | Residual below 1e-8 on all test layouts; agrees with the dense row sums | 0.75 |
| 2.2 | Iterative solvers | CG for the uniform chain, whose Q is symmetric, so I-Q is symmetric positive definite. BiCGSTAB or GMRES, with an incomplete-LU preconditioner, for the rational and mixed chains, whose Q is not symmetric. Report iterations and residual | All solvers agree with LU to 1e-6 on a 50x50 grid | 1.0 |
| 2.3 | Monte Carlo guard | Add a maximum step count per trial and a clear error for non-absorbing chains | A deliberately broken chain raises instead of hanging | 0.25 |
| 2.4 | Exact exit search | The decision is one discrete position along a boundary, so enumerate every legal position instead of BFGS. Score each by the rational max BFS distance (cheap), then re-score the best 5 by ensemble T90 from 1.4 | Search returns the true minimum on three small grids checked by hand | 1.0 |
| 2.5 | Relabel the old optimiser | Keep BFGS only as a comparison baseline named "heuristic search"; E5 reports where it misses the true optimum | Docstring and README updated | 0.25 |
| 2.6 | SVD role | Keep SVD for small or ill-conditioned cases; report the condition number (exact on small grids, estimated via `svds` on moderate ones) and show a rank-truncated approximation error on one case | One figure and one sentence on when SVD helps | 0.25 |

### Phase 3: Experiments and evidence (3 person-days)

| ID | Task | Days |
| --- | --- | --- |
| 3.1 | Harness: one config file, fixed seed list, CSV plus PNG output, environment record, `run_all.py` | 0.5 |
| 3.2 | E1 and E2 (can run before capacity lands) | 0.5 |
| 3.3 | E3 and E4 | 0.75 |
| 3.4 | E5 and E6 | 0.75 |
| 3.5 | E7, a one-page results table, figure clean-up | 0.5 |

The experiment definitions are in Section 6.

### Phase 4: Package and defend (5 person-days)

| ID | Task | Days |
| --- | --- | --- |
| 4.1 | Slides: rewrite using Section 8 wording, replace stock figures with E1 to E7 output, add a limitations slide | 1.0 |
| 4.2 | Report: method, validation, results, limitations, reproducibility appendix | 2.0 |
| 4.3 | Web editor update (optional): expose capacity, traits, decay and ensemble statistics; return safe errors outside development; restrict CORS origins; declare Flask extras | 1.0 |
| 4.4 | Viva preparation: extend the audit's Q&A (Section 11), then one mock viva where each member answers about another's part | 0.75 |
| 4.5 | Final freeze: tag the release, clone fresh, install, run the tests and `run_all.py`, confirm outputs match the report | 0.25 |

## 5. Design decisions for the new engines

### Panic with decay

The governing equation becomes

∂u/∂t = D ∇²u + k ρ − γ u

with the same wall-aware conductances. In the explicit scheme this adds one term, u ← u + Δt(D∇²u + kρ − γu), and the existing CFL guard is unchanged because the decay term only adds a stable damping factor when γΔt < 1 (check this in the validator). With a constant density ρ the field settles at a steady level of about kρ/γ, which turns the "panic saturates" risk into a stated, testable property. Choose γ so panic fades over tens of steps after the crowd leaves, and sweep it in E7.

### Capacity and panic-driven clustering

- Each cell holds at most `c` agents (default 1, a parameter). Each exit passes at most `c_exit` agents per step, which is what lets a door jam.
- Every step, shuffle the active agents, let each draw its intended move from its row of Q\_t as today, and accept the move only if the target cell has room. A refused move leaves the agent where it is and adds one to that cell's blocked-move count.
- Panic coupling stays as built (rows blend toward a random walk) but now has a physical consequence: more random movement means more attempted moves into crowded cells, more refusals, and more clustering. The slide claim becomes "panic randomises movement, which under capacity produces more blocking", and E3 tests it directly.
- Arbitration order is randomised each step so no agent is favoured by its index.

### Traits

- Sample (speed, reaction time) per agent from the bivariate Gaussian with the slide's mean (1.34 m/s, 0.50 s); keep Σ as the existing covariance.
- Truncate to physical ranges (for example speed in \[0.3, 2.0\] m/s, reaction time at least 0), since a Gaussian can go negative, and count how many draws were truncated.
- State two constants: cell size (0.5 m) and step length (0.4 s), so the budget per step is v·Δt/cell. Each agent keeps a fractional budget and makes floor(budget) moves per step, capped at 2. Reaction time becomes a start delay of ceil(τ/Δt) steps.
- **Slide correction.** The slide describes Σ as linking panic, speed and pathfinding, which is three traits. The sampler has two. Reword the slide to "correlated speed and reaction time"; do not claim a pathfinding trait.

### Ensemble statistics

One run gives T50, T90 and T100 (the times by which 50%, 90% and all agents have left). Across K runs (default K = 200, seeds fixed) report the median of each with a 95% bootstrap interval. If a run reaches `max_steps` with agents left, T100 is censored: report the fraction of runs that finished and compare configurations on "fraction evacuated by step T" rather than a mean that hides the failures. Compare two configurations with paired seeds and report the difference with its interval.

### Bottleneck definition

Three maps replace the single row-sum map, each with a plain meaning:

1. **Time to exit** (row sum of N): expected steps from each cell; high means far from an exit.
2. **Expected visits** (s0^T N, where s0 is the starting distribution of agents): where the crowd is expected to spend time in the fixed chain.
3. **Blocked moves** (from ensemble runs with capacity): where agents actually queue. This is the map that is allowed to be called a bottleneck map.

## 6. Experiments

Every experiment writes a CSV, a PNG and its configuration to `experiments/out/`, and states its question, method, expected result and failure criterion before it runs.

| ID | Question | Method | Pass criterion |
| --- | --- | --- | --- |
| E1 | Does the exact fixed-chain analysis match Monte Carlo? | Three layouts; for 10 start cells compare N·1 with the Monte Carlo mean at M = 1e3, 1e4, 1e5 | Exact value inside the Monte Carlo 95% interval for at least 90% of cells; standard error falls like 1/√M |
| E2 | How does the solver scale? | Grids 20, 40, 60, 80, 100, 150; time, peak memory and residual for dense LU, sparse single-solve, CG (uniform chain) and BiCGSTAB (rational chain); mark runs that do not fit in memory | A table of measured times; the claim "100x100 is feasible" is kept only for the methods that finish with residual below 1e-8 |
| E3 | What does panic do, and does it clash with capacity? | Both slide layouts, panic on and off, capacity on, K = 200; also compare the frozen N\_t row sums at t = 0 with the actual mean remaining time per start cell | Report T50, T90, T100 differences with intervals; if panic shows no effect, say so and keep the result |
| E4 | Where does the crowd jam as density rises? | Sweep agents from 5% to 60% of floor cells on the narrow hallway | T90 against density shows a visible knee; blocked-move map peaks at the narrowest point |
| E5 | Is the exact exit search better than BFGS? | Enumerate all boundary positions on 5 grids; compare with BFGS result; re-score the top 5 by ensemble T90 | Report how often BFGS misses the true optimum and whether the max-distance and ensemble rankings agree |
| E6 | Do traits matter? | Traits off against on, same seeds, with capacity; also vary the speed-reaction correlation | Report the change in T90 and in the tail (T100); a null result is acceptable and reported |
| E7 | Are conclusions robust to the invented parameters? | One-at-a-time sweeps of D, k, γ, β, Δt and capacity; record whether the ranking of layouts and exit positions stays the same (Kendall tau) | Rankings stable across the swept range, or the unstable parameter is named in the report |

The slide claim "validated" is limited to what E1 shows. E2 decides what the slides may say about 100x100. E3 to E7 support only comparative statements.

## 7. Schedule and team allocation

The split below is a suggestion; adjust to each member's strengths. Day numbers assume four people working in parallel.

| Day | Aditya | Krishiv | Sibi | Siddharth |
| --- | --- | --- | --- | --- |
| 1 | 0.1, 0.2 | 0.3 | 0.4 | 0.3 (review) |
| 2 | 1.2 capacity | 1.1 panic decay | 1.3 traits | 2.1 sparse solve |
| 3 | 1.2 capacity | 1.5 maps | 1.3 traits | 2.2 CG and BiCGSTAB |
| 4 | 1.4 ensemble | 1.6, 1.7 | 2.4 exact search | 2.2, 2.3 |
| 5 | 3.1 harness | 3.2 E1 and E2 | 2.5, 2.6 | 4.3 web editor (optional) |
| 6 | 3.3 E3 and E4 | 3.4 E5 | 3.4 E6 | 4.1 slides (draft) |
| 7 | 3.5 E7 | 4.2 report | 4.2 report | 4.1 slides |
| 8 | 4.4 viva prep | 4.2 report | 4.4 viva prep | 4.5 freeze |

Critical path: 1.2 capacity (days 2 to 3) → 1.4 ensemble (day 4) → E3 to E7 (days 6 to 7) → report (days 7 to 8). Anything on that path slipping moves the deadline; Phase 2 tasks have slack until day 5.

Integration points: merge to `main` at the end of days 3, 5 and 7, each with the headless suite green, so that experiments always run on a tagged commit.

## 8. Claims ledger and slide rewording

| Slide wording now | Replace with |
| --- | --- |
| "Predicts evacuation performance before a building is built" | "Compares candidate layouts under stated, uncalibrated assumptions" |
| "Heatmap of where fatal bottlenecks form" | "Heatmap of where agents are blocked and queue (blocked-move map)" |
| "Row sums of N show bottlenecks" | "Row sums of N give expected time to exit from each cell" |
| "Time-varying Q\_t solved each step as a forecast" | "Frozen-chain diagnostic at each step; forecasts come from ensemble runs" |
| "Validated analytically via Monte Carlo" | "Fixed chain validated against Monte Carlo (E1)" |
| "Panic makes agents collide or cluster" | "Panic randomises movement; with capacity this increases blocking (E3)" |
| "Σ links panic, speed and pathfinding" | "Σ links walking speed and reaction time" |
| "LU / SVD / CG" | State exactly which are implemented, and which solver is used for which chain (Section 4, Phase 2) |
| "100x100 is computationally feasible" | Only the methods and times measured in E2 |
| "Exit-placement optimisation" | "Exhaustive search over legal exit positions, re-scored by ensemble time" |
| "Beats Helbing, Zheng and others" (the "What We Do Better" column) | "Complementary approach": stochastic and interpretable, but uncalibrated and with no social-force realism |

Also add one limitations slide: no calibration or drill validation, no casualty model, four-neighbour grid with no continuous movement, panic units qualitative, results are comparisons and not forecasts.

## 9. Cut orders

If time is short, cut from the bottom of each list upward; never cut the rewording.

**5 working days (minimum honest project).** Do: Phase 0, 1.1, 1.2, 1.3, 1.4, 1.5, E1, E3, E6, slides and a short report. Drop: CG and iterative solvers (the slides then say "LU and SVD"), the exact exit search (say "exit placement is a heuristic"), E2 to E5 and E7, the web editor. Keep the dense solve claim limited to small grids and delete the 100x100 claim.

**8 working days (planned project).** Everything in Phases 0 to 3 and 4.1, 4.2, 4.4, 4.5; the web editor stays at its audited state with the safe-error fix only if time allows. E7 may be reduced to two parameters.

**10 or more working days.** The full plan including the web editor update, plus two extras if useful: a second capacity regime (door throughput sweep) and a comparison of the frozen N\_t forecast against actual remaining time across the whole run, not only at t = 0.

## 10. Risks

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Capacity changes baseline behaviour and breaks tests | Medium | High | Infinite capacity must reproduce baseline (Ground rule 1); keep the old path behind the flag |
| Experiments show panic has little effect | Medium | Medium | Report it honestly; check the coupling strength β in E7; a null result is still evidence |
| Ensemble runs are slow (Python agent loop) | Medium | Medium | Measure one run first; reduce K or grid size; vectorise agent sampling per row type before cutting experiments |
| 100x100 does not fit with dense N | High | Low | Already expected; the sparse single-solve and CG answer it |
| Dynamic snapshots misread as forecasts | Medium | High | Rename everywhere (1.6); label every plot |
| Gaussian truncation distorts the sampled distribution | Low | Low | Report the truncated fraction; it is tiny at the stated mean and spread |
| Merge conflicts across four people editing `simulation.py` | High | Medium | Split changes into small commits; Aditya owns `simulation.py`, others submit changes via branches; merge points on days 3, 5, 7 |
| Host Tk failure hides real plot bugs | Low | Low | Agg in CI; one manual plot check on a machine with a working Tk |
| Deadline earlier than 8 days | Unknown | High | Use the 5-day cut order immediately; confirm the deadline on day 1 |

## 11. Definition of done and viva preparation

### Final checklist

- [ ] `pytest -q` passes on a fresh clone with no flags, including the new invariant tests.
- [ ] `run_all.py` regenerates every figure and CSV, with the environment file beside them.
- [ ] Every claim on the slides appears in the Section 8 ledger and has a test or experiment behind it.
- [ ] README, `currentstatus.md` and the slides agree; the "all phases 100% complete" line is gone.
- [ ] Frozen-chain wording is consistent in code, API responses, plots and slides.
- [ ] The limitations slide and report section are present.
- [ ] A tag marks the delivered commit.

### Extra viva questions to rehearse (beyond the audit's list)

- **Why is capacity needed for a bottleneck?** Without a limit on how many agents may share a cell or pass a door, nothing queues, so the only slowness left is distance.
- **Why is CG valid only for the uniform chain?** CG needs a symmetric positive definite matrix. The uniform Q is symmetric because every neighbour pair has equal mass in both directions; the rational and mixed Q are not.
- **Why does a single solve beat forming N?** Expected time needs only (I-Q)t = 1, which is one sparse solve with O(n) unknowns instead of an n×n dense inverse.
- **Why enumerate exit positions instead of BFGS?** The decision is a small discrete set, so checking all of it is both cheaper to reason about and guaranteed optimal for the chosen objective.
- **What does the panic decay term change?** It gives panic a steady level, about k·ρ/γ, instead of growing without bound, so results no longer depend on how long the simulation runs.
- **What would you need to claim real-world accuracy?** Calibration of speeds, reaction times and panic parameters against measured drill data, and validation on layouts not used for calibration.
- **Is a null result a failure?** No. If panic or traits change little under capacity, that is a finding about the model, reported with its interval.

## 12. Open questions for the team

1. What is the deadline, and is the deliverable a viva, a report, slides, a live demo, or a mix? This sets which cut order applies.
2. Does the course expect the web editor, or is the command-line pipeline enough?
3. Do we have drill or published evacuation data to compare against even loosely (for example the Nature 2000 door-flow observations)? If yes, add a small qualitative comparison to E4; if not, the "not calibrated" line stands.
4. Who owns `simulation.py`? One owner avoids most merge pain.
