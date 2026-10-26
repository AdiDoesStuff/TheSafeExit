import numpy as np
import pytest
import math

from src.floor_plan import get_simple_room
from src.simulation import EvacuationSimulation
from src.gaussian_sampler import sample_agents, sample_agents_truncated, truncate_traits


def test_traits_off_reproduces_baseline():
    grid = get_simple_room(12, 12, exit_width=2)
    s1 = EvacuationSimulation(floor_plan=grid, n_agents=20, seed=1000, panic_enabled=False, use_traits=False)
    r1 = s1.run(max_steps=200)
    s2 = EvacuationSimulation(floor_plan=grid, n_agents=20, seed=1000, panic_enabled=False)
    r2 = s2.run(max_steps=200)
    assert r1["agent_position_history"] == r2["agent_position_history"]


def test_traits_reproducible():
    grid = get_simple_room(10, 10, exit_width=2)
    s1 = EvacuationSimulation(floor_plan=grid, n_agents=12, seed=5, panic_enabled=True, use_traits=True)
    r1 = s1.run(max_steps=80)
    s2 = EvacuationSimulation(floor_plan=grid, n_agents=12, seed=5, panic_enabled=True, use_traits=True)
    r2 = s2.run(max_steps=80)
    assert np.array_equal(s1.speeds, s2.speeds)
    assert np.array_equal(s1.reactions, s2.reactions)
    assert r1["agent_position_history"] == r2["agent_position_history"]
    assert r1["blocked_counts"].tolist() == r2["blocked_counts"].tolist()


def test_no_move_before_delay():
    grid = get_simple_room(10, 10, exit_width=2)
    # one agent with reaction time large
    traits = (np.array([1.0]), np.array([1.0]))  # delay_steps = ceil(1.0/0.4)=3
    sim = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                               use_traits=True, traits=traits, initial_positions=[(5, 5)] if False else None)
    # place single agent at a known cell via initial_positions
    # find a transient cell
    from src.markov import build_state_map
    sm, _, nt, na = build_state_map(grid)
    # choose first transient coord
    coord = list(sm.keys())[0]
    sim = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                               use_traits=True, traits=traits, initial_positions=[coord])
    res = sim.run(max_steps=10)
    hist = res["agent_position_history"]
    init_pos = hist[0][0]
    # delay_steps=3: no move for first 3 steps (steps 0,1,2)
    for i in range(min(3, len(hist))):
        assert hist[i][0] == init_pos or hist[i][0] == list(coord), f"moved early at step {i}"


def test_trait_ordering():
    grid = get_simple_room(10, 10, exit_width=2)
    from src.markov import build_state_map
    sm, _, nt, na = build_state_map(grid)
    coord = list(sm.keys())[0]
    # fast agent
    sim_fast = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                                    use_traits=True, traits=(np.array([2.0]), np.array([0.0])),
                                    initial_positions=[coord])
    rf = sim_fast.run(max_steps=200)
    # slow agent
    sim_slow = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                                    use_traits=True, traits=(np.array([0.5]), np.array([0.0])),
                                    initial_positions=[coord])
    rs = sim_slow.run(max_steps=200)
    tf = max(sim_fast.evacuation_times.values()) if sim_fast.evacuation_times else 200
    ts = max(sim_slow.evacuation_times.values()) if sim_slow.evacuation_times else 200
    assert tf <= ts
    # delayed agent
    sim_del = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                                   use_traits=True, traits=(np.array([1.0]), np.array([0.8])),
                                   initial_positions=[coord])
    rd = sim_del.run(max_steps=200)
    td = max(sim_del.evacuation_times.values()) if sim_del.evacuation_times else 200
    sim_base = EvacuationSimulation(floor_plan=grid, n_agents=1, seed=1, panic_enabled=False,
                                    use_traits=True, traits=(np.array([1.0]), np.array([0.0])),
                                    initial_positions=[coord])
    rb = sim_base.run(max_steps=200)
    tb = max(sim_base.evacuation_times.values()) if sim_base.evacuation_times else 200
    assert td >= tb


def test_truncation():
    agents, n_trunc = sample_agents_truncated(n_agents=1000, seed=42)
    assert np.all(agents[:, 0] >= 0.3)
    assert np.all(agents[:, 0] <= 2.0)
    assert np.all(agents[:, 1] >= 0.0)
    assert isinstance(n_trunc, int)
    # truncate_traits alias
    raw = sample_agents(n_agents=100, seed=1)
    clipped, nt = truncate_traits(raw)
    assert clipped.shape == raw.shape


def test_global_numpy_state_independence():
    grid = get_simple_room(8, 8, exit_width=2)
    np.random.seed(123)
    r1 = EvacuationSimulation(floor_plan=grid, n_agents=10, seed=9, use_traits=True).run(max_steps=50)
    np.random.seed(999)
    r2 = EvacuationSimulation(floor_plan=grid, n_agents=10, seed=9, use_traits=True).run(max_steps=50)
    assert r1["agent_position_history"] == r2["agent_position_history"]
