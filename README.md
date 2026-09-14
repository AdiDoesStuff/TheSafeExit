# TheSafeExit

TheSafeExit is a stochastic crowd evacuation simulation project. It represents a building as a grid, moves agents through the grid using absorbing Markov chains, models panic as a diffusion field, and visualizes evacuation bottlenecks with fundamental matrix heatmaps.

The project was built for a mathematics-focused intelligent systems setting, so the main emphasis is not just animation. The emphasis is the mathematical pipeline behind evacuation behavior: probability, Markov chains, linear algebra, PDE-based diffusion, numerical stability, and sparse solvers.

## Project Idea

In an emergency, people do not simply move in straight lines. They choose paths, crowd near bottlenecks, influence each other, and may become less rational as panic increases.

TheSafeExit models that idea using a grid-based floor plan:

- Walls are blocked cells.
- Walkable areas are transient states.
- Exits are absorbing states.
- Agents move probabilistically from cell to cell.
- Crowd density generates panic.
- Panic diffuses through passable space.
- Higher panic shifts movement from rational shortest-path behavior toward random movement.

This creates a feedback loop:

```text
agent positions -> crowd density -> panic field -> transition probabilities -> new agent positions
```

The result is a simulation that can report evacuation statistics and produce heatmaps showing where bottlenecks are likely to form.

## Main Features

- Grid-based floor plan representation.
- Simple room and L-shaped hallway generators.
- BFS reachability validation.
- Geodesic distance field computation from every cell to the nearest exit.
- Absorbing Markov chain construction with `Q` and `R` matrices.
- Uniform random-walk movement model.
- Rational shortest-path movement model.
- Panic-based convex mixing between rational and random movement.
- Panic diffusion using a wall-aware explicit numerical scheme.
- Sparse LU solver for the fundamental matrix.
- Dense SVD solver for validation and benchmarking.
- Fundamental matrix bottleneck heatmaps.
- Multivariate Gaussian agent trait sampler and visualization.
- Pytest suite covering floor plans, Markov matrices, solvers, panic, and simulation.

## Repository Structure

```text
src/
  floor_plan.py        Floor plan creation, validation, plotting, distance fields
  markov.py            State mapping and transition matrix construction
  solvers.py           Fundamental matrix solvers using LU and SVD
  heatmap.py           Bottleneck heatmap generation
  gaussian_sampler.py  Behavioral trait sampling
  panic.py             Panic diffusion PDE
  simulation.py        Full coupled evacuation simulation

tests/
  test_floor_plan.py
  test_markov.py
  test_solvers.py
  test_panic.py
  test_simulation.py

demo/
  mid_sem_demo.py
  demo_phase2.py
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

The tests verify core mathematical and implementation invariants, including reachability, row-stochastic transition matrices, panic stability, wall isolation, and LU/SVD agreement.

## Running Demos

Mid-semester demo:

```bash
python demo/mid_sem_demo.py
```

Phase 2 coupled simulation demo:

```bash
python demo/demo_phase2.py
```

Generated figures are saved in:

```text
demo/figures/
```

## Core Mathematical Model

The floor plan is converted into an absorbing Markov chain:

```text
P = [ Q  R ]
    [ 0  I ]
```

Where:

- `Q` represents transitions between walkable cells.
- `R` represents transitions from walkable cells to exits.
- Exit cells are absorbing.

For a fixed transition matrix, the fundamental matrix is:

```text
N = (I - Q)^(-1)
```

The row sums of `N` estimate expected time to absorption from each starting cell. The project maps those values back to the grid to produce bottleneck heatmaps.

In the coupled simulation, panic changes over time, so the transition matrix becomes `Q_t`. The simulation can compute frozen-time snapshots:

```text
N_t = (I - Q_t)^(-1)
```

These snapshots are useful for instantaneous bottleneck analysis.

## Current Scope

The main integrated simulation supports panic-driven Markov movement and evacuation tracking. The Gaussian behavioral sampler is implemented and demonstrated separately, but individual sampled traits are not yet wired into the main simulation loop.

See `PROJECT_EXPLANATION.md` for a thorough viva-preparation explanation of every module, the data flow, important invariants, limitations, and math concepts used.

