import pytest
import numpy as np
import scipy.sparse as sp
from src.floor_plan import (
    get_simple_room,
    get_hallway,
    compute_exit_distance_field,
    CELL_WALL,
    CELL_WALKABLE,
    CELL_EXIT,
)
from src.markov import (
    build_state_map,
    build_QR,
    build_uniform_QR,
    build_rational_QR,
    build_mixed_QR,
    get_markov_stats,
)



def test_markov_construction_simple_room():
    grid = get_simple_room(10, 10, exit_width=2)
    state_map, idx_map, n_transient, n_absorbing = build_state_map(grid)
    
    assert n_transient == 64  # (10-2)*(10-2) walkable cells = 64
    assert n_absorbing == 2
    assert len(state_map) == 66

    Q, R = build_QR(grid, state_map, n_transient, n_absorbing)
    assert Q.shape == (n_transient, n_transient)
    assert R.shape == (n_transient, n_absorbing)

    # Invariant: Each row sum of [Q | R] must equal 1.0 (stochastic matrix for transient states)
    row_sums = Q.sum(axis=1).A1 + R.sum(axis=1).A1
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_markov_small_hand_verifiable_grid():
    # 3x3 grid: 1 exit cell, 1 wall cell, 7 walkable cells
    grid = np.array([
        [CELL_WALKABLE, CELL_WALKABLE, CELL_EXIT],
        [CELL_WALKABLE, CELL_WALL,     CELL_WALKABLE],
        [CELL_WALKABLE, CELL_WALKABLE, CELL_WALKABLE]
    ], dtype=int)

    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    assert n_transient == 7
    assert n_absorbing == 1

    Q, R = build_QR(grid, state_map, n_transient, n_absorbing)
    row_sums = Q.sum(axis=1).A1 + R.sum(axis=1).A1
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_compute_exit_distance_field_and_rational_QR():
    # 5x5 room with exit at (2, 4)
    grid = np.ones((5, 5), dtype=int)
    grid[0, :] = CELL_WALL
    grid[-1, :] = CELL_WALL
    grid[:, 0] = CELL_WALL
    grid[:, -1] = CELL_WALL
    grid[2, 4] = CELL_EXIT

    dist = compute_exit_distance_field(grid)
    assert dist[2, 4] == 0
    assert dist[0, 0] == -1  # wall
    assert dist[2, 3] == 1  # 1 step to exit
    assert dist[1, 1] == 4  # (1,1) -> (1,2) -> (1,3) -> (2,3) -> (2,4) or similar 4 steps

    state_map, idx_map, n_transient, n_absorbing = build_state_map(grid)
    Q_rat, R_rat = build_rational_QR(grid, dist, state_map, n_transient, n_absorbing)

    # Invariant: Each row sum of [Q_rat | R_rat] must equal 1.0
    row_sums = Q_rat.sum(axis=1).A1 + R_rat.sum(axis=1).A1
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    # Cell (1, 1) has distance 4; all its rational moves must have distance 3
    i_11 = state_map[(1, 1)]
    q_row_11 = Q_rat.getrow(i_11)
    for j, p in zip(q_row_11.indices, q_row_11.data):
        assert p > 0
        dest_coord = idx_map[j]
        assert dist[dest_coord] == 3


def test_build_mixed_QR_convex_combination():
    grid = get_simple_room(8, 8, exit_width=2)
    dist = compute_exit_distance_field(grid)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)

    Q_unif, R_unif = build_uniform_QR(grid, state_map, n_transient, n_absorbing)
    Q_rat, R_rat = build_rational_QR(grid, dist, state_map, n_transient, n_absorbing)

    # 1. lambda = 0 reproduces rational QR exactly
    lam_0 = np.zeros(n_transient)
    Q_0, R_0 = build_mixed_QR(Q_rat, R_rat, Q_unif, R_unif, lam_0)
    np.testing.assert_allclose(Q_0.toarray(), Q_rat.toarray(), atol=1e-12)
    np.testing.assert_allclose(R_0.toarray(), R_rat.toarray(), atol=1e-12)

    # 2. lambda = 1 reproduces uniform QR exactly
    lam_1 = np.ones(n_transient)
    Q_1, R_1 = build_mixed_QR(Q_rat, R_rat, Q_unif, R_unif, lam_1)
    np.testing.assert_allclose(Q_1.toarray(), Q_unif.toarray(), atol=1e-12)
    np.testing.assert_allclose(R_1.toarray(), R_unif.toarray(), atol=1e-12)

    # 3. Arbitrary non-uniform random lambda in [0, 1] satisfies row-stochasticity
    rng = np.random.default_rng(42)
    lam_rand = rng.uniform(0.0, 1.0, size=n_transient)
    Q_mixed, R_mixed = build_mixed_QR(Q_rat, R_rat, Q_unif, R_unif, lam_rand)
    row_sums = Q_mixed.sum(axis=1).A1 + R_mixed.sum(axis=1).A1
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

