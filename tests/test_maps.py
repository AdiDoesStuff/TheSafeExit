import numpy as np
import pytest

from src.floor_plan import get_room_with_barrier, get_simple_room
from src.heatmap import fundamental_to_heatmap, time_to_exit_map, expected_visits_map, blocked_moves_map, plot_heatmap
from src.markov import build_state_map, build_uniform_QR, build_rational_QR, build_mixed_QR
from src.solvers import fundamental_matrix_lu
from src.simulation import EvacuationSimulation


def _setup(grid):
    sm, its, nt, na = build_state_map(grid)
    Qu, Ru = build_uniform_QR(grid, sm, nt, na)
    Qr, Rr = build_rational_QR(grid, __import__("src.floor_plan", fromlist=["compute_exit_distance_field"]).compute_exit_distance_field(grid), sm, nt, na)
    return sm, nt, na, Qu, Ru, Qr, Rr


def test_rowsum_equals_old():
    grid = get_simple_room(10, 10, exit_width=2)
    sm, nt, na, Qu, Ru, Qr, Rr = _setup(grid)
    Q, R = build_mixed_QR(Qr, Rr, Qu, Ru, np.zeros(nt))
    N = fundamental_matrix_lu(Q)
    old = fundamental_to_heatmap(N, grid, sm, nt)
    new = time_to_exit_map(N, sm, grid)
    assert np.array_equal(old, new, equal_nan=True)


def test_s0_at_n_identity():
    grid = get_simple_room(10, 10, exit_width=2)
    sm, nt, na, Qu, Ru, Qr, Rr = _setup(grid)
    Q, R = build_mixed_QR(Qr, Rr, Qu, Ru, np.zeros(nt))
    N = fundamental_matrix_lu(Q)
    # uniform s0
    s0 = np.ones(nt) / nt
    visits = s0 @ N
    m = expected_visits_map(N, s0, sm, grid)
    # check sum
    assert abs((s0 @ N).sum() - (s0 @ N.sum(axis=1))) < 1e-9
    # check expected_visits_map places values
    vals = m[~np.isnan(m)]
    assert vals.size > 0


def test_expected_visits_default_uniform():
    grid = get_simple_room(8, 8, exit_width=2)
    sm, nt, na, Qu, Ru, Qr, Rr = _setup(grid)
    Q, R = build_mixed_QR(Qr, Rr, Qu, Ru, np.zeros(nt))
    N = fundamental_matrix_lu(Q)
    m = expected_visits_map(N, None, sm, grid)
    assert m.shape == grid.shape


def test_blocked_moves_map():
    grid = get_simple_room(8, 8, exit_width=2)
    bc = np.zeros(grid.shape, dtype=int)
    bc[2, 2] = 5
    m = blocked_moves_map(bc, grid)
    assert np.isnan(m[0, 0])  # wall? row0 is wall
    assert m[2, 2] == 5


def test_corridor_bottleneck():
    # room left, neck, room right with exit
    grid = np.full((9, 15), 0, dtype=int)
    grid[1:8, 1:6] = 1
    grid[1:8, 6:9] = 1  # neck width 3
    grid[1:8, 9:14] = 1
    grid[4, 14] = 2  # exit
    # walls border
    grid[0, :] = 0
    grid[8, :] = 0
    grid[:, 0] = 0
    grid[:, 14] = 0
    # fix exit on border
    grid[4, 14] = 2
    # ensure walkable neck
    # validate reachability
    from src.floor_plan import validate_reachability
    validate_reachability(grid)
    # run with capacity
    all_max = []
    for seed in range(5):
        sim = EvacuationSimulation(floor_plan=grid, n_agents=15, seed=seed, panic_enabled=False,
                                   cell_capacity=1)
        res = sim.run(max_steps=200)
        bc = res["blocked_counts"]
        # argmax among non-wall
        masked = np.where(grid == 0, -1, bc)
        flat = np.argmax(masked)
        r, c = np.unravel_index(flat, grid.shape)
        all_max.append((r, c))
    # print argmax for debugging if failure
    print("argmax cells:", all_max)
    # check at least one argmax in neck columns 6-8
    neck_hits = [1 for (r, c) in all_max if 6 <= c <= 8]
    assert sum(neck_hits) >= 1, f"argmax cells {all_max}"


def test_plot_functions_headless():
    grid = get_simple_room(8, 8, exit_width=2)
    sm, nt, na, Qu, Ru, Qr, Rr = _setup(grid)
    Q, R = build_mixed_QR(Qr, Rr, Qu, Ru, np.zeros(nt))
    N = fundamental_matrix_lu(Q)
    hm = time_to_exit_map(N, sm, grid)
    fig, ax = plot_heatmap(hm, grid)
    assert fig is not None