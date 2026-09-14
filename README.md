# TheSafeExit

TheSafeExit is a stochastic crowd evacuation simulation project. It models a building as a grid, moves agents using absorbing Markov chains, spreads panic with a wall-aware diffusion model, and visualizes evacuation bottlenecks over time.

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
- Sparse LU and dense SVD fundamental matrix solvers.
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
  solvers.py            Fundamental matrix solvers using LU and SVD
  heatmap.py            Static bottleneck heatmaps
  heatmap_sequence.py   Time-evolving heatmap grids, GIFs, migration tracking
  gaussian_sampler.py   Multivariate Gaussian behavior sampling
  panic.py              Panic diffusion PDE
  simulation.py         Coupled evacuation simulation

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
pytest
```

On Windows using the included virtual environment:

```bash
.\.venv\Scripts\python.exe -m pytest
```

For headless plotting/test runs, use Matplotlib's non-GUI backend:

```bash
$env:MPLBACKEND='Agg'; .\.venv\Scripts\python.exe -m pytest
```

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

So `N_t = (I - Q_t)^(-1)` is treated as a frozen-time bottleneck snapshot.

## Phase 3 Additions

The current simulation stores reduced heatmap snapshots instead of raw `N_t` matrices. Each snapshot stores:

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
