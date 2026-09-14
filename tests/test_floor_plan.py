import pytest
import numpy as np
from src.floor_plan import (
    get_simple_room,
    get_two_exit_room,
    get_room_with_pillar,
    get_hallway,
    validate_reachability,
    CELL_WALL,
    CELL_WALKABLE,
    CELL_EXIT,
)


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


def test_two_exit_room():
    grid = get_two_exit_room(20, 20, exit_width=2)
    assert grid.shape == (20, 20)
    # Two exits of width 2 -> 4 exit cells total
    assert np.sum(grid == CELL_EXIT) == 4
    assert validate_reachability(grid) is True


def test_room_with_pillar():
    grid = get_room_with_pillar(20, 20, exit_width=2, pillar_size=4)
    assert grid.shape == (20, 20)
    assert np.sum(grid == CELL_EXIT) == 2
    # Pillar is 4x4 wall block in center
    assert np.all(grid[8:12, 8:12] == CELL_WALL)
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


