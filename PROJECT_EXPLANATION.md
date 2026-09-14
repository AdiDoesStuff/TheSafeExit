# TheSafeExit: Current Project Explanation and Viva Preparation Notes

## 1. What This Project Does

TheSafeExit is a stochastic crowd evacuation simulation project. It represents a building as a 2D grid, treats walkable cells as Markov-chain states, treats exits as absorbing states, models panic as a diffusion field, and produces evacuation statistics plus bottleneck heatmap visualizations.

The current project is not just a static shortest-path model. Its main idea is a feedback loop:

```text
agent positions -> crowd density -> panic field -> movement probabilities -> new agent positions
```

In plain English:

1. Agents occupy walkable grid cells.
2. Crowded cells create panic.
3. Panic spreads through passable space but not through walls.
4. High panic makes movement less rational.
5. Agents move probabilistically until they reach exits.
6. The project records evacuation metrics and evolving bottleneck heatmaps.

The project currently includes:

- Phase 1 / mid-sem: floor plans, Markov matrices, solver benchmarks, static heatmaps, Gaussian sampling.
- Phase 2: coupled panic-driven evacuation simulation.
- Phase 3: time-evolving bottleneck heatmap sequences, animation, and bottleneck migration tracking.
- Phase 3b: geometry generalization study across baseline, two-exit, and pillar-room floor plans.

## 2. Project Structure

```text
TheSafeExit/
  src/
    __init__.py
    floor_plan.py
    markov.py
    solvers.py
    heatmap.py
    heatmap_sequence.py
    gaussian_sampler.py
    panic.py
    simulation.py

  tests/
    test_floor_plan.py
    test_markov.py
    test_solvers.py
    test_panic.py
    test_simulation.py
    test_heatmap_sequence.py

  demo/
    mid_sem_demo.py
    demo_phase2.py
    demo_phase3.py
    demo_phase3b.py
    figures/

  requirements.txt
  pytest.ini
  README.md
  PROJECT_EXPLANATION.md
```

Each source file has a clear responsibility:

- `floor_plan.py`: creates and validates grid layouts.
- `markov.py`: converts grids into absorbing Markov chain matrices.
- `solvers.py`: computes the fundamental matrix using LU or SVD.
- `heatmap.py`: turns a fundamental matrix into a 2D bottleneck heatmap.
- `heatmap_sequence.py`: visualizes heatmap snapshots across time.
- `gaussian_sampler.py`: samples behavioral traits from a multivariate Gaussian.
- `panic.py`: implements panic diffusion as a wall-aware numerical PDE.
- `simulation.py`: connects everything into the evacuation loop.

## 3. Floor Plan Model

File: `src/floor_plan.py`

The building is represented as a 2D NumPy array. Every cell has one integer code:

```python
CELL_WALL = 0
CELL_WALKABLE = 1
CELL_EXIT = 2
```

Interpretation:

- `0`: wall or obstacle.
- `1`: normal walkable space.
- `2`: exit.

This grid representation turns geometry into a graph-like problem. Rows and columns become coordinates, and valid motion is allowed only through four-neighbor adjacency: up, down, left, and right.

### 3.1 `get_simple_room`

Creates a rectangular room with outer walls and one exit on the right boundary.

The process is:

1. Fill the full grid with walkable cells.
2. Set the outer boundary to walls.
3. Replace a section of the right wall with exit cells.

This is the baseline geometry used in many tests and demos.

### 3.2 `get_two_exit_room`

Creates a rectangular room with two exits:

- one on the left wall,
- one on the right wall.

This was added for Phase 3b. It allows the project to study whether agents and bottlenecks behave differently when the building offers multiple exit choices.

Strong viva point:

The two-exit room lets you discuss symmetry, exit choice, and whether the panic/random-walk regime causes agents to cross between left and right exit regions.

### 3.3 `get_room_with_pillar`

Creates a rectangular room with one right-side exit and a central square pillar.

The pillar is simply a block of wall cells placed in the center. It changes the shortest paths and creates obstacle-induced bottlenecks.

This was also added for Phase 3b to test whether conclusions from the simple room still hold when geometry is more complex.

### 3.4 `get_hallway`

Creates an L-shaped corridor with a constriction near the end.

This is useful because corridors and chokepoints naturally create bottlenecks. It is used in the mid-sem demo and floor-plan tests.

### 3.5 `validate_reachability`

Checks that every walkable cell can reach at least one exit.

It uses multi-source breadth-first search, starting from all exit cells. It expands through walkable cells and ensures no walkable region is isolated.

Why this matters:

- If a walkable cell cannot reach an exit, agents placed there can never evacuate.
- Absorbing Markov chain analysis assumes eventual absorption is possible.
- The rational movement matrix depends on a valid distance gradient to an exit.

If unreachable cells exist, the function raises `ValueError`.

### 3.6 `compute_exit_distance_field`

Computes the shortest grid distance from every passable cell to the nearest exit.

Output convention:

- exits have distance `0`,
- walkable cells have positive integer distance,
- walls have distance `-1`.

This distance field is used to build rational movement. A rational agent tries to move from distance `d` to distance `d - 1`.

## 4. Markov Chain Construction

File: `src/markov.py`

The Markov model treats every walkable cell as a transient state and every exit as an absorbing state.

The absorbing Markov chain block structure is:

```text
P = [ Q  R ]
    [ 0  I ]
```

Where:

- `Q`: transient-to-transient probabilities.
- `R`: transient-to-exit probabilities.
- `I`: exits are absorbing.
- `0`: exited agents do not return to the building.

The code explicitly builds `Q` and `R`, because active agents only need transient rows.

### 4.1 `build_state_map`

This converts 2D coordinates into 1D state indices.

It returns:

```python
state_map
index_to_state
n_transient
n_absorbing
```

Index convention:

- walkable cells get indices `0` to `n_transient - 1`,
- exit cells get indices `n_transient` onward.

The function calls `validate_reachability` first, so invalid floor plans fail early.

### 4.2 `build_QR` / `build_uniform_QR`

Builds an unbiased random-walk Markov chain.

From each walkable cell, there are four possible directions. Each has probability `0.25`.

For each direction:

- walkable neighbor: probability goes into `Q`,
- exit neighbor: probability goes into `R`,
- wall or boundary: probability becomes a self-loop.

The self-loop means that if an agent tries to walk into a wall, it stays in the same cell for that part of the probability distribution.

Critical invariant:

```text
sum(Q row) + sum(R row) = 1
```

Every transient row must be row-stochastic.

### 4.3 `build_rational_QR`

Builds a shortest-path or steepest-descent transition model.

For a cell at distance `d` from the nearest exit, rational movement only allows moves to passable neighbors with distance `d - 1`.

If several neighbors reduce the distance equally, probability is split equally.

This represents calm, goal-directed evacuation behavior.

### 4.4 `build_mixed_QR`

Builds the panic-mixed transition matrices:

```text
Q_t[i, :] = (1 - lambda[i]) Q_rational[i, :] + lambda[i] Q_uniform[i, :]
R_t[i, :] = (1 - lambda[i]) R_rational[i, :] + lambda[i] R_uniform[i, :]
```

Interpretation:

- `lambda = 0`: fully rational.
- `lambda = 1`: fully random.
- values between 0 and 1: partial panic.

The current simulation computes:

```python
lambda_t = clip(beta * panic, 0, 1)
```

So high panic pushes the movement model toward random walking.

This is the core coupling between the panic PDE and the Markov chain.

## 5. Panic Diffusion

File: `src/panic.py`

Panic is stored as a 2D array `u` with the same shape as the floor plan.

The model is:

```text
du/dt = D * Laplacian(u) + S
```

In the code:

```text
S = k * rho
```

Where:

- `rho` is crowd density, computed from active agent coordinates.
- `k` controls how strongly crowding generates panic.
- `D` controls how quickly panic diffuses.

### 5.1 Wall-Aware Diffusion

Panic cannot diffuse through walls.

The implementation uses conductance arrays:

- `h_cond`: horizontal conductance between neighboring cells,
- `v_cond`: vertical conductance between neighboring cells.

Conductance is `1` only when both cells are passable. If either cell is a wall, conductance is `0`.

This creates no-flux boundaries at walls. In viva terms, this is a Neumann-style wall boundary condition.

### 5.2 Numerical Method

The method is explicit in time and grid-based in space. The code describes it as FTCS-like, implemented with finite-volume-style fluxes.

The update idea is:

```text
new panic = old panic + dt * (diffusion contribution + source contribution)
```

Then:

- negative panic is clipped to zero,
- wall cells are forced to zero.

### 5.3 Stability

Explicit diffusion schemes need a stable time step.

The stability condition checked by the code is:

```text
D * dt * (1/dx^2 + 1/dy^2) <= 0.5
```

`PanicField.max_stable_dt` computes a safe value using a safety factor. If an unstable `dt` is requested, `UnstableTimestepError` is raised.

This is a strong viva point because it shows numerical-method awareness.

### 5.4 `compute_agent_density`

This converts active agent coordinates into a density grid.

Multiple agents in the same cell are counted correctly. The implementation uses flattened indices and `np.bincount`, which is vectorized and efficient.

## 6. Fundamental Matrix and Solvers

File: `src/solvers.py`

For a fixed absorbing Markov chain:

```text
N = (I - Q)^(-1)
```

The fundamental matrix has this interpretation:

```text
N[i, j] = expected number of visits to transient state j
          before absorption, starting from transient state i
```

The row sum:

```text
sum_j N[i, j]
```

is the expected time to evacuation from state `i`, assuming `Q` remains fixed.

### 6.1 Why Solve a Linear System?

The project does not directly invert the matrix. It solves:

```text
(I - Q)N = I
```

This is better because direct matrix inversion is expensive and less numerically desirable.

### 6.2 Sparse LU

`solve_fundamental_lu` uses:

```python
scipy.sparse.linalg.splu
```

It converts `I - Q` to CSC sparse format, factorizes it, and solves for `N`.

Sparse LU is the practical solver because grid transition matrices are sparse.

### 6.3 Dense SVD

`solve_fundamental_svd` converts the matrix to dense form and uses singular value decomposition.

It is slower, but useful as a validation solver for smaller grids.

### 6.4 Solver Benchmarking

`benchmark_solvers` compares LU and SVD across grid sizes.

The demo caps SVD at smaller sizes because dense SVD becomes expensive quickly.

## 7. Static Heatmaps

File: `src/heatmap.py`

`fundamental_to_heatmap` maps fundamental matrix row sums back onto the 2D floor plan.

Output:

- walkable cells contain expected evacuation steps,
- exits contain `0.0`,
- walls contain `NaN`.

`plot_heatmap` visualizes this heatmap, marks exits, and highlights the top three bottleneck cells.

This is useful for static bottleneck analysis when `Q` is fixed.

## 8. Time-Evolving Heatmap Sequences

File: `src/heatmap_sequence.py`

This is the major Phase 3 addition.

Earlier versions could store raw `N_t` matrices. The current version is more memory-conscious: when a snapshot is requested, `simulation.py` computes `N_t`, immediately converts it into a 2D heatmap, extracts the top bottleneck cells, stores the reduced snapshot, and lets the raw dense matrix be released.

Each snapshot dictionary has this structure:

```python
{
    "t": timestep,
    "heatmap_2d": heatmap_2d,
    "top3_cells": [(row, col, value), ...],
    "mean_panic": mean_panic,
    "active_agents": active_agents,
}
```

This is important because full `N_t` matrices can be large. A reduced 2D heatmap is much easier to store, plot, and animate.

### 8.1 `global_colorscale`

Computes a single `(vmin, vmax)` across all heatmap snapshots.

Why it matters:

If every frame uses its own color scale, the animation can visually exaggerate or hide changes. A global color scale makes different time steps comparable.

### 8.2 `track_bottleneck_migration`

Tracks the primary bottleneck over time.

For each snapshot, it chooses the first entry in `top3_cells`, which represents the highest-value bottleneck cell. It returns records like:

```python
{"t": 20, "cell": (row, col), "value": expected_steps}
```

This allows the project to discuss not only where bottlenecks exist, but how the worst bottleneck migrates as panic evolves.

### 8.3 `plot_heatmap_grid`

Creates a static multi-panel grid of heatmaps across time.

It:

- uses a shared global color scale,
- marks exits,
- marks top bottleneck cells,
- displays timestep, mean panic, and active-agent count in each panel.

This produces `phase3_01_heatmap_grid.png`.

### 8.4 `animate_heatmap_sequence`

Creates a two-panel GIF animation:

- left panel: evolving bottleneck heatmap,
- right panel: mean panic and active-agent curves over time.

A cursor line shows which timestep the current heatmap frame corresponds to.

This produces `phase3_02_heatmap_animation.gif`.

## 9. Main Simulation Engine

File: `src/simulation.py`

Class: `EvacuationSimulation`

This is the orchestration layer. It connects:

- floor plan geometry,
- state mapping,
- uniform and rational Markov matrices,
- panic field,
- dynamic transition mixing,
- stochastic agent movement,
- reduced bottleneck snapshots,
- per-agent position history.

### 9.1 Constructor

Inputs:

```python
floor_plan
n_agents
D
k
beta
initial_positions
seed
```

Initialization steps:

1. Store the floor plan.
2. Compute the exit distance field.
3. Build the state map.
4. Build uniform `Q, R`.
5. Build rational `Q, R`.
6. Create a `PanicField`.
7. Initialize agents.
8. Precompute transient row/column arrays for fast panic extraction.

Agent state convention:

- active agent: transient state index,
- evacuated agent: `-1`.

### 9.2 `step(record_snapshot=False)`

One simulation step:

1. Find active agents.
2. Convert active state indices to grid coordinates.
3. Compute crowd density.
4. Advance panic diffusion.
5. Read panic values at all transient cells.
6. Compute `lambda_t = clip(beta * panic, 0, 1)`.
7. Build mixed `Q_t, R_t`.
8. Optionally compute and reduce a bottleneck snapshot.
9. Sample the next state for each active agent.
10. Mark agents that enter exits as evacuated.
11. Return metrics.

Returned fields:

```python
{
    "step": ...,
    "active_agents": ...,
    "evacuated_agents": ...,
    "mean_panic": ...,
    "max_panic": ...,
    "mean_lambda": ...,
    "snapshot": ...
}
```

### 9.3 Movement Sampling

For each agent, the simulation extracts one row from `Q_t` and one row from `R_t`.

It builds a list of possible destination states and their probabilities, normalizes them to avoid floating-point drift, and samples using:

```python
self.rng.choice(cand_indices, p=cand_probs)
```

If the chosen destination is transient, the agent moves there.

If the chosen destination is an exit, the agent is marked evacuated and its evacuation time is recorded.

### 9.4 `run(max_steps=600, snapshot_every=20, max_snapshots=30)`

The current `run` API records reduced snapshots by cadence.

Parameters:

- `max_steps`: maximum simulation length.
- `snapshot_every`: take a heatmap snapshot every N steps.
- `max_snapshots`: maximum number of snapshots to store.

Returned result:

```python
{
    "total_steps": ...,
    "n_agents": ...,
    "evacuated_agents": ...,
    "evacuation_rate": ...,
    "mean_evacuation_time": ...,
    "history": ...,
    "snapshots": ...,
    "agent_position_history": ...
}
```

Important current change:

`snapshots` is now a list of reduced snapshot dictionaries, not a dictionary of raw `N_t` matrices.

Another current change:

`agent_position_history` records each agent's coordinate after each step. Evacuated agents are stored as `[-1, -1]`. Phase 3b uses this to analyze exit-choice switching.

## 10. Gaussian Agent Sampler

File: `src/gaussian_sampler.py`

This module samples behavioral traits from a multivariate normal distribution:

```text
x ~ N(mu, Sigma)
```

Default mean:

```python
MU_DEFAULT = [1.34, 0.50]
```

Meaning:

- walking speed around `1.34 m/s`,
- reaction time around `0.50 s`.

Default covariance:

```python
SIGMA_DEFAULT = [
    [0.04, 0.015],
    [0.015, 0.010]
]
```

The off-diagonal covariance means the traits are correlated.

Functions:

- `sample_agents`: generates sampled trait vectors.
- `validate_sample_statistics`: checks sample mean/covariance against target values.
- `plot_agent_distribution`: plots samples with 1-sigma and 2-sigma ellipses.

Current status:

The Gaussian sampler is implemented and demonstrated in `mid_sem_demo.py`, but sampled speed and reaction time are not yet used by `EvacuationSimulation` to alter movement probabilities or timing. So Engine B exists as a standalone implemented module, not as a fully coupled part of the current main simulation loop.

## 11. Demo Scripts

### 11.1 `demo/mid_sem_demo.py`

Demonstrates:

- simple room and hallway floor plans,
- reachability checks,
- large sparse Markov matrix construction,
- LU vs SVD benchmark,
- static heatmaps,
- Gaussian agent sampling.

Generated figures:

- `demo_01_floor_plans.png`
- `demo_03_solver_benchmark.png`
- `demo_04_simple_room_heatmap.png`
- `demo_04_hallway_heatmap.png`
- `demo_05_gaussian_sampling.png`

### 11.2 `demo/demo_phase2.py`

Demonstrates:

- floor plan and exit distance field,
- rational vs uniform transition row comparison,
- coupled panic-driven simulation trajectory,
- panic field snapshots,
- one frozen-time fundamental matrix heatmap snapshot.

Generated figures:

- `phase2_01_floor_plan_and_distance.png`
- `phase2_02_rational_vs_uniform.png`
- `phase2_03_simulation_trajectory.png`
- `phase2_04_panic_snapshots.png`
- `phase2_05_heatmap_snapshot.png`

Important note:

This demo appears to use an older snapshot style in places. The current `EvacuationSimulation.run` returns snapshot lists through `snapshot_every` and `max_snapshots`, while older code used `record_heatmap` and `heatmap_interval`. If running old demos fails, this API mismatch is the reason.

### 11.3 `demo/demo_phase3.py`

Demonstrates the current Phase 3 heatmap-sequence system.

It:

1. Runs a 20x20 simple-room simulation with 80 agents.
2. Captures reduced `N_t` heatmap snapshots every 20 steps.
3. Produces a static multi-panel heatmap grid.
4. Produces a synchronized GIF animation.
5. Tracks and plots primary bottleneck migration.

Generated figures:

- `phase3_01_heatmap_grid.png`
- `phase3_02_heatmap_animation.gif`
- `phase3_03_bottleneck_migration.png`

### 11.4 `demo/demo_phase3b.py`

Runs a geometry generalization study.

It compares three floor plans:

- `baseline`: simple room,
- `two_exit`: two exits on opposite walls,
- `pillar`: simple room with central pillar.

The script holds simulation parameters fixed:

- rows and columns,
- number of agents,
- random seed,
- `D`,
- `k`,
- `beta`,
- max steps,
- snapshot cadence.

Only geometry changes.

This lets the project ask:

> Are the observed bottleneck patterns properties of the model, or artifacts of one room shape?

It also includes `exit_choice_consistency` for the two-exit case. That function uses `agent_position_history`, filters out evacuated agents, and counts whether active agents switch sides of the room before or after panic saturation.

Generated figures:

- `phase3b_01_geometries.png`
- `phase3b_02_migration_comparison.png`

## 12. Test Suite

The project uses `pytest`.

### 12.1 `test_floor_plan.py`

Checks:

- simple room creation,
- hallway creation,
- reachability validation,
- error on unreachable walkable cells.

### 12.2 `test_markov.py`

Checks:

- correct transient and absorbing state counts,
- matrix shapes,
- row-stochasticity,
- distance-field values,
- rational movement goes only to lower-distance cells,
- mixed matrices match rational at `lambda=0`,
- mixed matrices match uniform at `lambda=1`,
- mixed matrices remain row-stochastic for random lambda.

### 12.3 `test_panic.py`

Checks:

- automatic CFL-safe time step,
- unstable time step raises error,
- panic does not enter walls,
- panic does not leak across disconnected rooms,
- source term creates panic,
- reset clears panic,
- density counts multiple agents in one cell correctly.

### 12.4 `test_solvers.py`

Checks:

- LU fundamental matrix shape,
- residual of `(I - Q)N = I`,
- nonnegative fundamental matrix entries,
- LU and SVD agreement.

### 12.5 `test_simulation.py`

Checks:

- simulation initialization,
- one-step behavior,
- complete evacuation in a small room,
- custom initial positions.

### 12.6 `test_heatmap_sequence.py`

Checks Phase 3 functionality:

- global color scale calculation,
- empty color-scale fallback,
- snapshot cadence and max snapshot cap,
- snapshot structure,
- GIF animation creation,
- bottleneck migration output shape and validity,
- static heatmap grid output creation.

## 13. Current Implementation Status

Implemented:

- grid floor-plan representation,
- simple room, two-exit room, pillar room, and hallway generation,
- reachability validation,
- exit distance field computation,
- absorbing Markov chain construction,
- uniform random movement,
- rational shortest-path movement,
- panic-based convex transition mixing,
- wall-aware panic diffusion PDE,
- CFL stability guard,
- stochastic simulation loop,
- evacuation-time tracking,
- reduced time-evolving bottleneck snapshots,
- per-agent position history,
- static heatmap plotting,
- heatmap sequence grid plotting,
- animated heatmap sequence GIF generation,
- bottleneck migration tracking,
- Phase 3b geometry comparison demos,
- standalone multivariate Gaussian sampler.

Partially implemented / not fully coupled:

- Gaussian walking speed and reaction time are sampled and visualized, but not used in the main movement loop.
- The simulation is stochastic, but there is no dedicated Monte Carlo estimator module that runs many independent simulations and reports confidence intervals.
- Exit placement optimization is not implemented.
- Older Phase 2 demo/test calls may need API updates because snapshot handling changed.

## 14. Key Invariants to Defend in Viva

1. Every walkable cell should be reachable from an exit.
2. Every row of `[Q | R]` should sum to 1.
3. Exit states are absorbing.
4. Wall cells cannot be entered by movement.
5. Panic cannot diffuse through walls.
6. The explicit panic PDE must satisfy the CFL condition.
7. For fixed `Q`, the fundamental matrix satisfies `(I - Q)N = I`.
8. LU and SVD should agree on small validation cases.
9. `lambda` must stay in `[0, 1]`.
10. Snapshot cadence and cap prevent unlimited storage of large heatmap data.

## 15. Important Viva Questions

### What is the main novelty?

The strongest idea is coupling a continuous panic diffusion field with a discrete absorbing Markov chain. Panic changes the transition probabilities over time.

### Why use Markov chains?

Because evacuation movement can be represented as probabilistic transitions between cells. It gives a clean way to model uncertainty, absorption at exits, and expected evacuation behavior.

### Why are exits absorbing?

Once an agent reaches an exit, it has left the building. It should not return to normal walkable states.

### What does `Q` represent?

Transitions from walkable cells to other walkable cells.

### What does `R` represent?

Transitions from walkable cells to exit cells.

### What does the fundamental matrix mean?

For fixed `Q`, `N = (I - Q)^(-1)` gives expected visits to transient states before absorption.

### Is `N_t` exact when panic changes over time?

No. Since `Q_t` changes with panic, `N_t` is a frozen-time approximation. It answers: if the current movement probabilities stayed fixed, where would bottlenecks be?

### Why store reduced heatmaps instead of raw `N_t`?

Raw `N_t` can be large because it is `n_transient x n_transient`. For visualization, we only need row sums mapped back to the grid and top bottleneck cells. Storing reduced snapshots is more memory-efficient.

### Why use BFS?

BFS is used because all grid moves have equal cost. It finds shortest path distance in number of grid steps.

### Why use sparse matrices?

Each cell connects to only a few neighbors. Sparse matrices avoid storing thousands or millions of zeros.

### Why does panic use a PDE?

Panic is modeled as something that spreads locally over space, similar to heat diffusion. A diffusion PDE captures local spreading over time.

### Why is there a stability condition?

The panic update is explicit. If `dt` is too large relative to `D`, `dx`, and `dy`, the numerical solution can blow up. The CFL condition prevents that.

### Why compare multiple geometries in Phase 3b?

To check whether observed behavior is caused by the model generally or only by one specific floor plan.

### What is bottleneck migration?

It is the movement of the highest-risk or highest expected-time cell across time as panic and agent distribution evolve.

### What is still missing?

Gaussian traits are not yet coupled into movement, there is no full repeated-run Monte Carlo estimator with confidence intervals, and there is no exit-placement optimizer.

## 16. How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run tests:

```bash
pytest
```

or, using the local virtual environment on Windows:

```bash
.\.venv\Scripts\python.exe -m pytest
```

If Matplotlib tries to use a GUI backend and fails because Tk/Tcl is not available, run tests with the non-GUI backend:

```bash
$env:MPLBACKEND='Agg'; .\.venv\Scripts\python.exe -m pytest
```

Run demos:

```bash
python demo/mid_sem_demo.py
python demo/demo_phase2.py
python demo/demo_phase3.py
python demo/demo_phase3b.py
```

Generated visual artifacts are saved in:

```text
demo/figures/
```

## 17. Math Concepts Used

### Grid Graphs

The floor plan is a grid graph where passable cells are nodes and four-neighbor adjacency defines edges.

### Breadth-First Search

Used for reachability validation and shortest exit-distance computation.

### Probability Distributions

Each Markov row is a categorical probability distribution over possible next states.

### Markov Chains

Agent movement has the Markov property: the next state depends on the current state.

### Absorbing Markov Chains

Exits are absorbing states, allowing evacuation to be modeled as absorption.

### Row-Stochastic Matrices

Each row of `[Q | R]` must sum to 1.

### Fundamental Matrix

`N = (I - Q)^(-1)` gives expected visits before absorption for fixed `Q`.

### Linear Systems

The project solves `(I - Q)N = I` instead of directly inverting a matrix.

### Sparse Linear Algebra

Sparse matrices and sparse LU are used because grid transition matrices contain mostly zeros.

### LU Decomposition

Used to solve the fundamental matrix system efficiently.

### Singular Value Decomposition

Used as a dense validation method for smaller systems.

### Multivariate Gaussian Distribution

Used to sample walking speed and reaction time in the standalone behavior module.

### Mean, Variance, and Covariance

Used to define and validate the Gaussian agent trait distribution.

### Eigenvalues and Eigenvectors

Used to draw Gaussian confidence ellipses from the covariance matrix.

### Partial Differential Equations

Panic is modeled with a diffusion PDE.

### Laplacian Operator

Measures local differences in panic and drives diffusion.

### Finite Difference / Finite Volume Methods

Used to discretize and update the panic PDE on the grid.

### Neumann / No-Flux Boundary Condition

Walls prevent panic flux across blocked edges.

### CFL Stability

Controls the maximum stable time step for explicit diffusion.

### Convex Combination

Used to mix rational and uniform movement based on panic.

### Monte Carlo Simulation

Individual agent moves are sampled randomly. Repeated runs could estimate expected evacuation time more rigorously.

### Time-Series Visualization

Used in Phase 3 to study evolving heatmaps, panic curves, active-agent curves, and bottleneck migration.
