import pytest
import numpy as np
import scipy.sparse as sp
from src.floor_plan import get_simple_room
from src.markov import build_state_map, build_QR
from src.solvers import solve_fundamental_lu, solve_fundamental_svd


def test_fundamental_matrix_lu_solver():
    grid = get_simple_room(10, 10, exit_width=2)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

    N_lu = solve_fundamental_lu(Q)
    assert N_lu.shape == (n_transient, n_transient)

    # Invariant 1: (I - Q) @ N = I => Residual ||(I - Q) @ N - I||_max < 1e-10
    I_dense = np.eye(n_transient)
    A_dense = I_dense - Q.toarray()
    residual = np.max(np.abs(A_dense @ N_lu - I_dense))
    assert residual < 1e-10

    # Invariant 2: Fundamental matrix entries N_ij must be non-negative
    assert np.all(N_lu >= -1e-12)


def test_lu_vs_svd_agreement():
    grid = get_simple_room(12, 12, exit_width=2)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

    N_lu = solve_fundamental_lu(Q)
    N_svd = solve_fundamental_svd(Q)

    max_diff = np.max(np.abs(N_lu - N_svd))
    assert max_diff < 1e-6
