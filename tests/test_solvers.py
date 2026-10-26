import pytest
import numpy as np
import scipy.sparse as sp
from src.floor_plan import get_simple_room
from src.markov import build_state_map, build_QR
from src.solvers import solve_fundamental_lu, solve_fundamental_svd, run_monte_carlo


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


def test_run_monte_carlo_convergence():
    # Construct a small hand-verifiable absorbing Markov chain:
    # State 0 (transient): transitions to State 1 with prob 1.0
    # State 1 (transient): transitions to Exit (absorbing) with prob 0.5, State 0 with prob 0.5
    # Closed-form:
    # (I - Q) = [[1.0, -1.0], [-0.5, 1.0]] => N = [[2.0, 2.0], [1.0, 2.0]]
    # Row sum from State 0 = 4.0 steps; Row sum from State 1 = 3.0 steps.
    Q = sp.csr_matrix([[0.0, 1.0], [0.5, 0.0]], dtype=np.float64)
    R = sp.csr_matrix([[0.0], [0.5]], dtype=np.float64)

    N = solve_fundamental_lu(Q)
    expected_0 = float(np.sum(N[0, :]))
    expected_1 = float(np.sum(N[1, :]))
    assert np.isclose(expected_0, 4.0)
    assert np.isclose(expected_1, 3.0)

    # Run Monte Carlo with M=5000
    res_0 = run_monte_carlo(Q, R, start_state_idx=0, M=5000, rng=42)
    assert len(res_0["trial_times"]) == 5000
    assert res_0["std_error"] > 0.0
    # Sample mean must be within 3.5 standard errors of the analytical expectation
    assert abs(res_0["mean"] - expected_0) < 3.5 * res_0["std_error"]

    res_1 = run_monte_carlo(Q, R, start_state_idx=1, M=5000, rng=42)
    assert abs(res_1["mean"] - expected_1) < 3.5 * res_1["std_error"]

