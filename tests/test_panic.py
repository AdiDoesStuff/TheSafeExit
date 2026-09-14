import pytest
import numpy as np
from src.floor_plan import get_simple_room, get_hallway, CELL_WALL, CELL_WALKABLE, CELL_EXIT
from src.panic import PanicField, UnstableTimestepError, compute_agent_density


def test_panic_field_initialization_and_auto_cfl():
    fp = get_simple_room(10, 10, exit_width=2)
    D = 0.2
    dx, dy = 1.0, 1.0
    field = PanicField(fp, D=D, dx=dx, dy=dy, safety_factor=0.9)

    # Theoretical limit: 0.5 / (0.2 * (1 + 1)) = 0.5 / 0.4 = 1.25
    expected_limit = 1.25
    expected_dt = 0.9 * expected_limit
    assert np.isclose(field.dt, expected_dt)
    assert field.is_stable() is True
    assert field.u.shape == (10, 10)
    assert field.total_panic() == 0.0


def test_panic_field_cfl_violation_raises_error():
    fp = get_simple_room(10, 10, exit_width=2)
    D = 0.2
    safe_dt = PanicField.max_stable_dt(D, safety_factor=0.9)
    unstable_dt = safe_dt / 0.9 * 1.5  # 1.5x theoretical limit

    # Check constructor CFL guard
    with pytest.raises(UnstableTimestepError, match="violates the CFL stability condition"):
        PanicField(fp, D=D, dt=unstable_dt)

    # Check step() method CFL guard
    field = PanicField(fp, D=D)
    with pytest.raises(UnstableTimestepError, match="step\\(\\) called with dt"):
        field.step(dt=unstable_dt)


def test_wall_isolation_and_neumann_no_flux():
    # 20x20 room with outer walls
    fp = get_simple_room(20, 20, exit_width=2)
    field = PanicField(fp, D=0.2, k=0.5)

    # Inject high agent density in center
    density = np.zeros_like(fp, dtype=float)
    density[9:11, 9:11] = 10.0

    for _ in range(30):
        field.step(density)

    # Invariant: panic must NEVER enter wall cells
    wall_mask = (fp == CELL_WALL)
    assert np.all(field.u[wall_mask] == 0.0)
    assert field.total_panic() > 0.0
    assert field.panic_at(10, 10) > 0.0


def test_disconnected_room_no_leakage():
    # Construct a 10x10 floor plan with a solid wall dividing left and right halves
    fp = np.full((10, 10), CELL_WALKABLE, dtype=int)
    fp[0, :] = CELL_WALL
    fp[-1, :] = CELL_WALL
    fp[:, 0] = CELL_WALL
    fp[:, -1] = CELL_WALL
    fp[:, 5] = CELL_WALL  # Solid middle wall dividing column 0..4 from 6..9

    field = PanicField(fp, D=0.25, k=1.0)

    # Inject panic ONLY on the left room
    density = np.zeros_like(fp, dtype=float)
    density[2:4, 2:4] = 5.0

    for _ in range(25):
        field.step(density)

    # Panic should diffuse on left side (cols 1..4)
    assert np.any(field.u[1:9, 1:5] > 0.0)

    # Panic must strictly be zero on the middle wall and the right room (cols 5..9)
    assert np.all(field.u[:, 5:] == 0.0)


def test_source_term_and_reset():
    fp = get_simple_room(10, 10, exit_width=1)
    k = 0.5
    field = PanicField(fp, D=0.2, k=k)

    # Single agent at (5, 5)
    density = np.zeros_like(fp, dtype=float)
    density[5, 5] = 2.0

    field.step(density)
    # After 1 step, panic should be generated at (5, 5) and slightly diffused
    assert field.panic_at(5, 5) > 0.0
    assert field.total_panic() > 0.0

    # Reset
    field.reset()
    assert field.total_panic() == 0.0
    assert np.all(field.u == 0.0)


def test_compute_agent_density():
    shape = (10, 10)
    # Heavy crowding scenario: multiple agents sharing the exact same cells
    # e.g., 5 agents in (2, 3), 3 agents in (5, 5), 1 in (8, 9) -> total 9 agents
    agent_coords = [
        (2, 3), (2, 3), (2, 3), (2, 3), (2, 3),
        (5, 5), (5, 5), (5, 5),
        (8, 9)
    ]
    density = compute_agent_density(agent_coords, shape)

    assert density.shape == shape
    # Must correctly accumulate all 5 agents on (2, 3), not overwrite with 1.0
    assert density[2, 3] == 5.0
    assert density[5, 5] == 3.0
    assert density[8, 9] == 1.0
    assert density.sum() == 9.0

    # Test equivalence with unbuffered in-place addition (np.add.at)
    coords_arr = np.array(agent_coords)
    ref_density = np.zeros(shape, dtype=float)
    np.add.at(ref_density, (coords_arr[:, 0], coords_arr[:, 1]), 1.0)
    np.testing.assert_array_equal(density, ref_density)

    # Empty agent list
    empty_density = compute_agent_density([], shape)
    assert empty_density.shape == shape
    assert empty_density.sum() == 0.0

