import pytest
import numpy as np
from src.floor_plan import get_simple_room, get_hallway
from src.simulation import EvacuationSimulation


def test_simulation_initialization_and_step():
    grid = get_simple_room(10, 10, exit_width=2)
    n_agents = 20
    sim = EvacuationSimulation(grid, n_agents=n_agents, D=0.2, k=0.5, beta=0.4, seed=42)

    assert sim.n_agents == n_agents
    assert len(sim.agent_states) == n_agents
    assert np.all(sim.agent_states >= 0)  # all initially active in transient states

    # Perform a single step
    step_info = sim.step(record_heatmap=True)

    assert step_info["step"] == 1
    assert step_info["active_agents"] + step_info["evacuated_agents"] == n_agents
    assert step_info["mean_panic"] >= 0.0
    assert step_info["N_t"] is not None
    assert step_info["N_t"].shape == (sim.n_transient, sim.n_transient)


def test_simulation_run_full_evacuation():
    grid = get_simple_room(8, 8, exit_width=2)
    n_agents = 15
    sim = EvacuationSimulation(grid, n_agents=n_agents, D=0.2, k=0.5, beta=0.4, seed=123)

    results = sim.run(max_steps=200, record_heatmap=True, heatmap_interval=5)

    assert results["evacuated_agents"] == n_agents
    assert results["evacuation_rate"] == 1.0
    assert results["mean_evacuation_time"] > 0
    assert len(results["history"]) == results["total_steps"]
    assert len(results["snapshots"]) > 0


def test_custom_initial_positions():
    grid = get_simple_room(8, 8, exit_width=2)
    custom_coords = [(1, 1), (1, 2), (2, 1)]
    sim = EvacuationSimulation(grid, n_agents=3, initial_positions=custom_coords, seed=7)

    active_coords = sim.get_active_agent_coords()
    assert len(active_coords) == 3
    # Check that initial coordinates match
    coords_set = {tuple(c) for c in active_coords}
    assert coords_set == set(custom_coords)
