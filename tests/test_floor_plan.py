import pytest
import numpy as np
from src.floor_plan import get_simple_room, get_hallway, validate_reachability, CELL_WALL, CELL_WALKABLE, CELL_EXIT


def test_simple_room_creation():
    grid = get_simple_room(20, 20, exit_width=2)
    assert grid.shape == (20, 20)
    assert np.sum(grid == CELL_EXIT) == 2
    assert np.sum(grid == CELL_WALKABLE) > 0
    assert validate_reachability(grid) is True


def test_hallway_creation():
    grid = get_hallway(25, 25, corridor_width=3, exit_width=1)
    assert grid.shape == (25, 25)
    assert np.sum(grid == CELL_EXIT) >= 1
    assert validate_reachability(grid) is True


def test_unreachable_floor_plan_raises_error():
    grid = get_simple_room(10, 10, exit_width=1)
    # Inject an isolated walkable cell completely surrounded by walls at (2, 2)
    grid[1, 1:4] = CELL_WALL
    grid[3, 1:4] = CELL_WALL
    grid[2, 1] = CELL_WALL
    grid[2, 3] = CELL_WALL
    grid[2, 2] = CELL_WALKABLE

    with pytest.raises(ValueError, match="Floor plan reachability validation failed"):
        validate_reachability(grid)
