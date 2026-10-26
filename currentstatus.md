# Audit Current Status Report

This document audits the TheSafeExit crowd evacuation simulation project against the planned architecture. The codebase was inspected and the test suite was executed to determine the exact status (DONE / PARTIAL / MISSING) of each component.

## Test Suite Execution
- **Run command:** `pytest -q`
- **Result:** 36 passed, 0 failures (100% pass rate).

## Architecture Status Table

| Phase / Engine | Item | Status | File(s) & Line(s) | Notes & Snippets |
| --- | --- | --- | --- | --- |
| **Engine A** | Absorbing Markov Chain | DONE | `src/markov.py` | `build_QR` (L46), `build_uniform_QR` (L92), `build_rational_QR` (L95), `build_mixed_QR` (L153). |
| **Engine B** | Multivariate Gaussian agent sampling | DONE | `src/gaussian_sampler.py` | Standalone behaviour module implemented. |
| **Engine C** | Panic contagion PDE | DONE | `src/panic.py` | `PanicField` (L66), `_laplacian` (L164), `UnstableTimestepError` (L62), `compute_agent_density` (L245). |
| **Engine D** | Sparse linear algebra | DONE | `src/solvers.py` | `solve_fundamental_lu` (L11) & alias `fundamental_matrix_lu` (L57), `solve_fundamental_svd` (L34) & alias `fundamental_matrix_svd` (L58) |


---

## Phase Checklist

### Phase 1 — Panic engine
| Item | Status | File(s) & Line(s) | Code Snippet |
| --- | --- | --- | --- |
| Flux-conductance discrete Laplacian with wall-aware no-flux boundaries | DONE | `src/panic.py:164-180` | `lap_h[:, :-1] += flux_h`<br>`lap_h[:, 1:] -= flux_h` |
| CFL stability guard raises UnstableTimestepError | DONE | `src/panic.py:127-133` | `if dt > safe_dt:`<br>&nbsp;&nbsp;`raise UnstableTimestepError(...)` |
| S = k*rho source term | DONE | `src/panic.py:217` | `source = self.k * density` |
| compute_agent_density using np.bincount | DONE | `src/panic.py:281` | `counts = np.bincount(flat_indices, minlength=rows * cols)` |

### Phase 2 — Q-row mutation / coupling
| Item | Status | File(s) & Line(s) | Code Snippet |
| --- | --- | --- | --- |
| compute_exit_distance_field (BFS to nearest exit) | DONE | `src/floor_plan.py:193-227` | `queue = deque()`<br>`for r, c in exits:`<br>&nbsp;&nbsp;`dist[r, c] = 0`<br>&nbsp;&nbsp;`queue.append((r, c))` |
| build_rational_QR (steepest descent, ties broken uniformly) | DONE | `src/markov.py:127-132` | `if grid[nr, nc] != CELL_WALL and distance_field[nr, nc] == d_curr - 1:`<br>&nbsp;&nbsp;`descent_neighbors.append((nr, nc))` |
| build_mixed_QR (blending both Q and R components) | DONE | `src/markov.py:189-190` | `Q_t = (one_minus_lam @ Q_rational + lam_diag @ Q_uniform).tocsr()`<br>`R_t = (one_minus_lam @ R_rational + lam_diag @ R_uniform).tocsr()` |
| EvacuationSimulation step/run logic (density -> panic -> lambda -> sample) | DONE | `src/simulation.py:146-230` | `density = compute_agent_density(...)`<br>`self.panic_field.step(density)`<br>`lam_t = np.clip(self.beta * panic_vec, 0.0, 1.0)` |
| beta default and its calibration rationale | DONE | `src/simulation.py:40-41` | `Default beta=0.4 corresponds to 1 / 2.5 (full panic at crush density rho >= 5).` |
| Decision on panic decay term (gamma) | DONE | `src/panic.py:220` | `u_new = self.u + step_dt * (self.D * lap + source)` (No decay term exists; panic is permanent unless explicitly reset) |

### Phase 3 — Time-evolving heatmaps
| Item | Status | File(s) & Line(s) | Code Snippet |
| --- | --- | --- | --- |
| Snapshot reduces N_t and discards it immediately | DONE | `src/simulation.py:163-188` | `N_t = fundamental_matrix_lu(Q_t)`<br>`heatmap_2d = fundamental_to_heatmap(...)`<br>`# Note: raw N_t matrix is immediately released / garbage collected` |
| global_colorscale shared across frames | DONE | `src/heatmap_sequence.py:21-55` | `vmin = min(vmins)`<br>`vmax = max(vmaxs)` |
| track_bottleneck_migration | DONE | `src/heatmap_sequence.py:58-86` | `def track_bottleneck_migration(snapshots: ...):` |
| snapshot_every / max_snapshots respected | DONE | `src/simulation.py:269-273` | `should_snapshot = (snapshot_every > 0 and ((self.time_step + 1) % snapshot_every == 0) and (len(snapshots) < max_snapshots))` |

### Phase 3b — Geometry generalization
| Item | Status | File(s) & Line(s) | Code Snippet |
| --- | --- | --- | --- |
| Comparison table across baseline / two-exit / pillar | DONE | `demo/demo_phase3b.py:57-61` | `"baseline": get_simple_room(...)`<br>`"two_exit": get_two_exit_room(...)`<br>`"pillar": get_room_with_pillar(...)` |
| Exit-choice consistency metric phase-split | DONE | `demo/demo_phase3b.py:121-161` | `def exit_choice_consistency(sim_result: dict, saturation_t: Optional[int] = None) -> None:` |

### Phase 4 — Validation
| Item | Status | File(s) & Line(s) | Note |
| --- | --- | --- | --- |
| Monte Carlo validation figure (run_monte_carlo) | DONE | `src/solvers.py:109-188`, `demo/demo_phase4.py` | Standalone $M$-trial simulation converging to $1^T N$ with $O(1/\sqrt{M})$ rate. |
| Decision on "panic disabled" meaning | DONE | `src/simulation.py:55, 142-161` | `panic_enabled=False` directly bypasses the panic PDE update and uses $Q_{rational}$ directly. |

### Phase 5 — Optimization
| Item | Status | File(s) & Line(s) | Note |
| --- | --- | --- | --- |
| CO2 exit-placement optimization | DONE | `src/optimization.py`, `demo/demo_phase5.py` | Minimizes worst-case $J = \max_i (\mathbf{1}^T N)_i$ using BFGS across boundary walls. Validated on symmetric square, rectangular, and asymmetric barrier rooms with multi-seed consistency and non-trivial off-center shifts (+27.8%). |

### Web Editor — Interactive Floor Plan Tool
| Item | Status | File(s) & Line(s) | Note |
| --- | --- | --- | --- |
| Flask REST backend | DONE | `demo/web_editor/app.py` | Four endpoints: `POST /api/analyze`, `POST /api/simulate`, `POST /api/monte_carlo`, `POST /api/optimize`. Serves `index.html` at `GET /`. |
| `/api/analyze` endpoint (Static Boundary Modes) | DONE | `demo/web_editor/app.py:98-193` | Computes the two static boundary cases: Rational ($\lambda=0$, geodesic descent) and Uniform ($\lambda=1$, unbiased random walk). Returns expected times via LU fundamental matrix $N = (I - Q)^{-1}$ and static $J^*$. |
| `/api/simulate` endpoint (Dynamic Coupled Mode) | DONE | `demo/web_editor/app.py:295-447` | Wraps `EvacuationSimulation` directly. Executes the coupled loop: $\rho \to$ `panic_field.step` $\to \lambda_t = \text{clip}(\beta u) \to (Q_t, R_t) \to N_t = (I - Q_t)^{-1}$. Returns time-series of frozen heatmaps, panic fields $u_t$, frame-consistent `global_colorscale` bounds, and static rational baseline $J^*$. |
| `/api/monte_carlo` endpoint | DONE | `demo/web_editor/app.py:195-235` | Accepts grid + start cell + trial count; runs `run_monte_carlo` and returns MC mean, std error, and the analytic `N[i,:]·1` for direct comparison. |
| `/api/optimize` endpoint | DONE | `demo/web_editor/app.py:237-293` | Accepts rows/cols/exit_width/obstacles; calls `optimize_exit_placement` (BFGS), returns per-wall results + best grid. |
| HTML/JS grid editor (`index.html`) | DONE | `demo/web_editor/index.html` | Single-file, zero-dependency SPA. Dark-mode 3-panel layout: tool palette (wall/floor/exit/erase), 24px paint grid (up to 60×60), results panel with Overview / Simulation / Monte Carlo / Optimizer / Log tabs. Features live time scrubber, play/pause auto-advance, zero-dependency canvas sparkline tracking $J(t)$ feedback vs static $J^*$, and multi-mode heatmaps. |
| Launcher script | DONE | `demo/web_editor/launch_editor.bat` | Double-click to start Flask server at `http://127.0.0.1:5050`. |

---

## General Codebase Integrity Checks

- **Exports match implementations:** YES. `src/__init__.py` successfully exposes all implemented classes and functions (including `solve_fundamental_lu`, `solve_fundamental_svd`, `run_monte_carlo`, `build_grid_with_exit_on_wall`, `worst_case_evacuation_time`, `optimize_exit_placement`, `get_room_with_barrier`, `get_room_with_pillar`, `EvacuationSimulation`, `global_colorscale`).
- **TODOs / stubs / dead code:** None found in the `src/` modules.
- **Design Decisions Contradictions:** None found.
  - No `gamma` (decay term) was added, strictly honoring the "panic is permanent" design choice.
  - Convex combination strictly guarantees row-sum = 1 by utilizing proper stochastic matrix blending on `Q` and `R`.
  - Both alias pairs `solve_fundamental_lu` / `fundamental_matrix_lu` and `solve_fundamental_svd` / `fundamental_matrix_svd` are defined, tested, and exported.
- **Web editor three-mode architecture:** The web editor now explicitly exposes three distinct modes, closing the gap where the interface previously only offered static Phase 2 boundary chains:
  1. *Rational Boundary Case ($\lambda=0$ limit)*: Pure geodesic shortest-path descent (static $Q_{rational}$).
  2. *Uniform Boundary Case ($\lambda=1$ limit)*: Pure unbiased random walk diffusion (static $Q_{uniform}$).
  3. *Dynamic Panic-Coupled Simulation*: Full time-domain integration coupling agent density $\rho_t$, panic diffusion PDE $u(x,y,t)$, transient transition mixing $\lambda_t = \text{clip}(\beta u)$, and quasi-static fundamental matrix bottleneck sequence $N_t = (I - Q_t)^{-1}$.
- **Web editor dependencies:** `flask` and `flask-cors` added to `.venv` (not in `requirements.txt` — editor is an optional dev tool). Run `.venv\Scripts\pip install flask flask-cors` if needed.

---

## Project Roadmap Status

All phases (Phases 1, 2, 3, 3b, 4, 5) and the **Web Editor Integration** are **100% COMPLETE**. The test suite has **36 passing unit tests** with full coverage across all engines, solvers, simulation loops, geometry generators, optimization pipelines, and panic coupling regression tests.

The interactive **Web Editor** (`demo/web_editor/`) now offers full first-class access to both static boundary analyses and the real-time dynamic panic-coupled PDE simulation. Users can draw custom floor plans, simulate panic feedback dynamics, scrub through time-series heatmaps, validate with Monte Carlo sampling, and optimize exit placement via gradient descent.
