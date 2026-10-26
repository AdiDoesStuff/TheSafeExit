import math
import numpy as np
import pytest

from src.floor_plan import get_simple_room, get_two_exit_room, get_room_with_pillar, get_hallway
from src.markov import build_state_map, build_uniform_QR, build_rational_QR, build_mixed_QR
from src.solvers import fundamental_matrix_lu
from src.simulation import EvacuationSimulation
from src.ensemble import run_ensemble
from src.gaussian_sampler import sample_agents
from src.floor_plan import compute_exit_distance_field


LAYOUTS = {
    "simple": get_simple_room(8, 8, exit_width=2),
    "two_exit": get_two_exit_room(8, 8, exit_width=2),
    "pillar": get_room_with_pillar(9, 9, exit_width=2, pillar_size=2),
    "hallway": get_hallway(10, 10, corridor_width=2, exit_width=1),
}


def _chains(grid):
    sm, its, nt, na = build_state_map(grid)
    Qu, Ru = build_uniform_QR(grid, sm, nt, na)
    df = compute_exit_distance_field(grid)
    Qr, Rr = build_rational_QR(grid, df, sm, nt, na)
    return sm, nt, na, Qu, Ru, Qr, Rr


# A. Row-stochastic
def test_base_chains_row_stochastic():
    for name, grid in LAYOUTS.items():
        sm, nt, na, Qu, Ru, Qr, Rr = _chains(grid)
        for Q, R in ((Qu, Ru), (Qr, Rr)):
            Qa = Q.toarray(); Ra = R.toarray()
            rs = Qa.sum(axis=1) + Ra.sum(axis=1)
            assert np.allclose(rs, 1.0, atol=1e-12), name
            assert np.all(Qa >= -1e-12) and np.all(Ra >= -1e-12)


def test_mixed_chains_row_stochastic():
    grid = LAYOUTS["simple"]
    sm, nt, na, Qu, Ru, Qr, Rr = _chains(grid)
    rng = np.random.default_rng(0)
    lambdas = [0.0, 0.25, 1.0, rng.random(nt), np.array([-0.5]*nt), np.array([1.5]*nt)]
    for lam in lambdas:
        Q, R = build_mixed_QR(Qr, Rr, Qu, Ru, lam)
        assert Q.shape == (nt, nt)
        assert R.shape == (nt, na)
        Qa = Q.toarray(); Ra = R.toarray()
        rs = Qa.sum(axis=1) + Ra.sum(axis=1)
        assert np.allclose(rs, 1.0, atol=1e-12)
        assert np.all(Qa >= -1e-12) and np.all(Ra >= -1e-12)


def test_whole_run_invariants():
    for name in ("simple", "pillar", "hallway"):
        grid = LAYOUTS[name]
        sim = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=3, panic_enabled=True,
                                   check_invariants=True)
        res = sim.run(max_steps=30)
        assert res["evacuated_agents"] >= 0


# B. Capacity conservation
@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("cell_cap", [1, 2])
@pytest.mark.parametrize("exit_cap", [None, 1])
@pytest.mark.parametrize("panic", [False, True])
@pytest.mark.parametrize("traits", [False, True])
def test_capacity_conservation(seed, cell_cap, exit_cap, panic, traits):
    grid = get_simple_room(8, 8, exit_width=1)
    sim = EvacuationSimulation(floor_plan=grid, n_agents=6, seed=seed, panic_enabled=panic,
                               cell_capacity=cell_cap, exit_capacity=exit_cap,
                               use_traits=traits, check_invariants=True)
    res = sim.run(max_steps=40)
    hist = res["agent_position_history"]
    evac_counts = []
    for step in hist:
        occ = {}
        active = 0
        evacuated = 0
        for pos in step:
            if pos[0] == -1:
                evacuated += 1
            else:
                active += 1
                occ[tuple(pos)] = occ.get(tuple(pos), 0) + 1
                r, c = pos
                assert grid[r, c] == 1
        assert active + evacuated == 6
        for k, v in occ.items():
            assert v <= cell_cap
        evac_counts.append(evacuated)
    assert all(evac_counts[i] <= evac_counts[i + 1] for i in range(len(evac_counts) - 1))
    # exit capacity per step
    if exit_cap is not None:
        prev = 0
        for ec in evac_counts:
            assert (ec - prev) <= exit_cap
            prev = ec


# C. Seed reproducibility
def test_seed_reproducibility_variants():
    grid = LAYOUTS["simple"]
    variants = [
        dict(panic_enabled=True),
        dict(panic_enabled=True, cell_capacity=2, exit_capacity=1),
        dict(panic_enabled=True, cell_capacity=2, exit_capacity=1, use_traits=True),
    ]
    for v in variants:
        s1 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=11, **v)
        r1 = s1.run(max_steps=40)
        s2 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=11, **v)
        r2 = s2.run(max_steps=40)
        assert r1["agent_position_history"] == r2["agent_position_history"]
        assert sorted(s1.evacuation_times.values()) == sorted(s2.evacuation_times.values())
        assert np.array_equal(s1.blocked_counts, s2.blocked_counts)


def test_different_seeds_differ():
    grid = LAYOUTS["simple"]
    s1 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=1, panic_enabled=True)
    r1 = s1.run(max_steps=40)
    s2 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=2, panic_enabled=True)
    r2 = s2.run(max_steps=40)
    assert r1["agent_position_history"] != r2["agent_position_history"]


def test_global_numpy_state():
    grid = LAYOUTS["simple"]
    np.random.seed(123)
    r1 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=9, panic_enabled=True, use_traits=True).run(max_steps=30)
    np.random.seed(999)
    r2 = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=9, panic_enabled=True, use_traits=True).run(max_steps=30)
    assert r1["agent_position_history"] == r2["agent_position_history"]


def test_interleaving():
    grid = LAYOUTS["simple"]
    s_a = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=5, panic_enabled=True)
    r_a1 = s_a.run(max_steps=20)
    # another sim in between
    EvacuationSimulation(floor_plan=grid, n_agents=8, seed=99, panic_enabled=True).run(max_steps=20)
    s_b = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=5, panic_enabled=True)
    r_b = s_b.run(max_steps=20)
    assert r_a1["agent_position_history"] == r_b["agent_position_history"]


def test_trait_sampler_reproducible():
    a = sample_agents(n_agents=50, seed=7)
    b = sample_agents(n_agents=50, seed=7)
    assert np.array_equal(a, b)
    rng = np.random.default_rng(123)
    c = sample_agents(n_agents=50, rng=rng)
    rng2 = np.random.default_rng(123)
    d = sample_agents(n_agents=50, rng=rng2)
    assert np.array_equal(c, d)


def test_ensemble_reproducible():
    grid = LAYOUTS["simple"]
    a = run_ensemble(grid, n_runs=4, base_seed=2, max_steps=20, n_agents=5, panic_enabled=False)
    b = run_ensemble(grid, n_runs=4, base_seed=2, max_steps=20, n_agents=5, panic_enabled=False)
    assert np.array_equal(a.per_run["T90"], b.per_run["T90"])