import numpy as np
import pytest
import scipy.sparse as sp

from src.floor_plan import (
    compute_exit_distance_field,
    get_hallway,
    get_room_with_pillar,
    get_simple_room,
    get_two_exit_room,
)
from src.markov import (
    build_QR,
    build_mixed_QR,
    build_rational_QR,
    build_state_map,
)
from src.solvers import (
    NonAbsorbingChainError,
    expected_time_residual,
    expected_times,
    matrix_condition_number,
    run_monte_carlo,
    solve_expected_time_dense,
    solve_expected_time_iterative,
    solve_expected_time_lu,
    solve_expected_time_sparse,
    solve_fundamental_lu,
    svd_rank_truncation_errors,
)


def _uniform_chain(grid):
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    Q, R = build_QR(grid, state_map, n_transient, n_absorbing)
    return Q, R


def _rational_chain(grid):
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    dist = compute_exit_distance_field(grid)
    Q, R = build_rational_QR(grid, dist, state_map, n_transient, n_absorbing)
    return Q, R


LAYOUTS = {
    "simple": get_simple_room(10, 10, exit_width=2),
    "two_exit": get_two_exit_room(10, 10, exit_width=2),
    "pillar": get_room_with_pillar(12, 12, exit_width=2, pillar_size=2),
    "hallway": get_hallway(15, 15, corridor_width=3, exit_width=1),
}


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_sparse_single_solve_residual_and_dense_agreement(name):
    grid = LAYOUTS[name]
    Q, _ = _uniform_chain(grid)

    report = solve_expected_time_sparse(Q)
    assert report.t.shape == (Q.shape[0],)
    assert report.residual < 1e-8
    assert np.all(np.isfinite(report.t))
    assert np.all(report.t >= 0)

    # The single solve must agree with the row sums of the full dense N.
    N = solve_fundamental_lu(Q)
    dense_row_sums = np.sum(N, axis=1)
    assert np.max(np.abs(report.t - dense_row_sums)) < 1e-8

    # The relative residual helper and the report agree.
    assert expected_time_residual(Q, report.t) == pytest.approx(report.residual)


def test_sparse_lu_and_dense_single_solve_agree():
    grid = get_simple_room(20, 20, exit_width=2)
    Q, _ = _uniform_chain(grid)

    sparse = solve_expected_time_sparse(Q)
    lu = solve_expected_time_lu(Q)
    dense = solve_expected_time_dense(Q)

    assert lu.residual < 1e-8
    assert dense.residual < 1e-8
    assert np.max(np.abs(sparse.t - lu.t)) < 1e-9
    assert np.max(np.abs(sparse.t - dense.t)) < 1e-9


def test_expected_times_convenience_matches_report():
    grid = get_simple_room(8, 8, exit_width=2)
    Q, _ = _uniform_chain(grid)
    assert np.allclose(expected_times(Q), solve_expected_time_sparse(Q).t, atol=0, rtol=0)


def test_cg_agrees_with_lu_on_uniform_chain_50x50():
    grid = get_simple_room(50, 50, exit_width=5)
    Q, _ = _uniform_chain(grid)

    reference = solve_expected_time_lu(Q)
    cg = solve_expected_time_iterative(Q, method="cg", rtol=1e-10)

    assert cg.method == "cg"
    assert cg.iterations > 0
    assert cg.residual < 1e-8
    assert np.max(np.abs(cg.t - reference.t)) < 1e-6


def test_bicgstab_agrees_with_lu_on_rational_chain_50x50():
    grid = get_simple_room(50, 50, exit_width=5)
    Q, _ = _rational_chain(grid)

    reference = solve_expected_time_lu(Q)
    result = solve_expected_time_iterative(Q, method="bicgstab", rtol=1e-10)

    assert result.iterations >= 0
    assert result.residual < 1e-8
    assert np.max(np.abs(result.t - reference.t)) < 1e-6


def test_bicgstab_on_mixed_chain():
    grid = get_simple_room(30, 30, exit_width=3)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)
    Q_u, R_u = build_QR(grid, state_map, n_transient, n_absorbing)
    dist = compute_exit_distance_field(grid)
    Q_r, R_r = build_rational_QR(grid, dist, state_map, n_transient, n_absorbing)
    gen = np.random.default_rng(7)
    lam = gen.uniform(0.0, 0.5, size=n_transient)
    Q_mix, R_mix = build_mixed_QR(Q_r, R_r, Q_u, R_u, lam)

    reference = solve_expected_time_sparse(Q_mix)
    result = solve_expected_time_iterative(Q_mix, method="bicgstab", rtol=1e-10)
    assert result.residual < 1e-8
    assert np.max(np.abs(result.t - reference.t)) < 1e-6


def test_iterative_solver_rejects_unknown_method():
    Q, _ = _uniform_chain(get_simple_room(8, 8, exit_width=2))
    with pytest.raises(ValueError, match="method"):
        solve_expected_time_iterative(Q, method="jacobi")


def test_condition_number_and_svd_truncation_are_sane():
    grid = get_simple_room(10, 10, exit_width=2)
    Q, _ = _rational_chain(grid)

    cond = matrix_condition_number(Q)
    assert np.isfinite(cond)
    assert cond > 1.0

    errors = svd_rank_truncation_errors(Q, ranks=(1, 2, 5, 10))
    assert len(errors) == 4
    assert all(e["relative_error"] >= 0 for e in errors)
    assert errors[-1]["relative_error"] <= errors[0]["relative_error"] + 1e-12
    # A full-rank reconstruction must reproduce the expected times.
    full_rank = svd_rank_truncation_errors(Q, ranks=(Q.shape[0],))[0]
    assert full_rank["relative_error"] < 1e-8


def test_monte_carlo_raises_on_closed_chain():
    Q = sp.eye(3, format="csr", dtype=float)
    R = sp.csr_matrix((3, 1), dtype=float)
    with pytest.raises(NonAbsorbingChainError, match="cannot reach"):
        run_monte_carlo(Q, R, start_state_idx=0, M=10, rng=1)


def test_monte_carlo_raises_on_two_state_closed_class():
    # 0 <-> 1, no route to an exit.
    Q = sp.csr_matrix([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    R = sp.csr_matrix((2, 1), dtype=float)
    with pytest.raises(NonAbsorbingChainError):
        run_monte_carlo(Q, R, start_state_idx=0, M=5, rng=2)


def test_monte_carlo_step_guard_raises_instead_of_hanging():
    # Absorption needs two steps from state 0.
    Q = sp.csr_matrix([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    R = sp.csr_matrix([[0.0], [1.0]], dtype=float)

    with pytest.raises(NonAbsorbingChainError, match="max_steps_per_trial"):
        run_monte_carlo(Q, R, start_state_idx=0, M=10, rng=3, max_steps_per_trial=1)

    res = run_monte_carlo(Q, R, start_state_idx=0, M=50, rng=3, max_steps_per_trial=10)
    assert np.all(res["trial_times"] == 2)
    assert res["mean"] == pytest.approx(2.0)


def test_monte_carlo_vectorised_matches_exact_mean():
    grid = get_simple_room(12, 12, exit_width=2)
    Q, R = _uniform_chain(grid)
    exact = solve_expected_time_sparse(Q).t

    start = int(Q.shape[0] // 2)
    res = run_monte_carlo(Q, R, start_state_idx=start, M=20000, rng=1234)
    assert abs(res["mean"] - exact[start]) < 4.0 * res["std_error"] + 1e-12
    assert res["trial_times"].shape == (20000,)


def test_monte_carlo_does_not_touch_global_numpy_state():
    Q = sp.csr_matrix([[0.0, 0.5], [0.0, 0.0]], dtype=float)
    R = sp.csr_matrix([[0.5], [1.0]], dtype=float)

    np.random.seed(1)
    a = run_monte_carlo(Q, R, 0, 200, rng=99)["trial_times"].copy()
    np.random.seed(2)
    b = run_monte_carlo(Q, R, 0, 200, rng=99)["trial_times"].copy()
    assert np.array_equal(a, b)


def test_expected_time_residual_validates_shape():
    Q, _ = _uniform_chain(get_simple_room(8, 8, exit_width=2))
    with pytest.raises(ValueError, match="shape"):
        expected_time_residual(Q, np.ones(3))
