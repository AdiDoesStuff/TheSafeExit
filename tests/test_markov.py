import pytest
import numpy as np
import scipy.sparse as sp
from src.floor_plan import get_simple_room, get_hallway, CELL_WALL, CELL_WALKABLE, CELL_EXIT
from src.markov import build_state_map, build_QR, get_markov_stats


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
