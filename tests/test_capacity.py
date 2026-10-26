import numpy as np
import pytest

from src.floor_plan import get_simple_room
from src.simulation import EvacuationSimulation


def test_infinite_capacity_reproduces_baseline():
    grid = get_simple_room(12, 12, exit_width=2)
    for cap in (None, 10 ** 9):
        s1 = EvacuationSimulation(floor_plan=grid, n_agents=20, seed=1000, panic_enabled=False, cell_capacity=cap, exit_capacity=cap)
        r1 = s1.run(max_steps=200)
        s2 = EvacuationSimulation(floor_plan=grid, n_agents=20, seed=1000, panic_enabled=False)
        r2 = s2.run(max_steps=200)
        assert r1["agent_position_history"] == r2["agent_position_history"]


def test_cell_capacity_invariants():
    grid = get_simple_room(12, 12, exit_width=2)
    for seed in range(5):
        for panic in (False, True):
            sim = EvacuationSimulation(
                floor_plan=grid, n_agents=25, seed=seed, panic_enabled=panic,
                cell_capacity=1,
            )
            res = sim.run(max_steps=100)
            hist = res["agent_position_history"]
            evac_counts = []
            for step_positions in hist:
                occ = {}
                active = 0
                evacuated = 0
                for pos in step_positions:
                    if pos[0] == -1:
                        evacuated += 1
                    else:
                        active += 1
                        key = tuple(pos)
                        occ[key] = occ.get(key, 0) + 1
                        r, c = pos
                        assert grid[r, c] == 1, "active on non-walkable"
                assert active + evacuated == 25
                for k, v in occ.items():
                    assert v <= 1
                evac_counts.append(evacuated)
            assert all(evac_counts[i] <= evac_counts[i + 1] for i in range(len(evac_counts) - 1))


def test_exit_capacity_single_exit():
    grid = get_simple_room(12, 12, exit_width=1)
    sim = EvacuationSimulation(floor_plan=grid, n_agents=8, seed=1, panic_enabled=False, exit_capacity=1)
    res = sim.run(max_steps=200)
    hist = res["agent_position_history"]
    evac_counts = [sum(1 for p in step if p[0] == -1) for step in hist]
    prev = 0
    for ec in evac_counts:
        absorbed = ec - prev
        assert absorbed <= 1
        prev = ec
    assert res["total_steps"] >= 8


def test_overfull_initial_placement_raises():
    grid = get_simple_room(8, 8, exit_width=2)
    with pytest.raises(ValueError):
        EvacuationSimulation(floor_plan=grid, n_agents=1000, seed=1, cell_capacity=1)


def test_capacity_deterministic():
    grid = get_simple_room(10, 10, exit_width=2)
    r1 = EvacuationSimulation(floor_plan=grid, n_agents=15, seed=7, cell_capacity=2, exit_capacity=1).run(max_steps=100)
    r2 = EvacuationSimulation(floor_plan=grid, n_agents=15, seed=7, cell_capacity=2, exit_capacity=1).run(max_steps=100)
    assert r1["agent_position_history"] == r2["agent_position_history"]
    assert r1["blocked_counts"].tolist() == r2["blocked_counts"].tolist()
