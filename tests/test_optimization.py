import pytest
import numpy as np
from src.floor_plan import get_simple_room, validate_reachability, CELL_EXIT
from src.optimization import (
    build_grid_with_exit_on_wall,
    worst_case_evacuation_time,
    optimize_exit_placement,
)


def test_worst_case_evacuation_time_matches_known_grid():
    # Baseline 20x20 simple room with 2-cell exit centered on right wall
    grid = get_simple_room(20, 20, exit_width=2)
    j_val = worst_case_evacuation_time(grid)

    # In a 20x20 room with exit at rows 9..10 on col 19, the farthest walkable cells
    # are (1, 1) and (18, 1) at distance |1 - 9| + |1 - 19| = 8 + 18 = 26 steps.
    assert np.isclose(j_val, 26.0)


def test_optimizer_improves_on_bad_seed():
    # Manually place exit near a corner (position 0 along right wall)
    grid_bad = build_grid_with_exit_on_wall(
        wall="right", position=0, rows=20, cols=20, exit_width=2
    )
    validate_reachability(grid_bad)
    j_bad = worst_case_evacuation_time(grid_bad)

    # Distance from (18, 1) to exit at rows 1..2 on col 19 is |18 - 2| + 18 = 34 steps
    assert j_bad > 26.0

    # Running optimizer starting from bad seeds
    opt_res = optimize_exit_placement(
        rows=20,
        cols=20,
        exit_width=2,
        initial_wall_positions={"right": 0.0, "left": 0.0, "top": 0.0, "bottom": 0.0},
    )

    assert opt_res["best_J"] < j_bad
    assert np.isclose(opt_res["best_J"], 26.0)
    assert opt_res["best_position"] == 8  # Centered position in 18-cell interior


def test_all_four_walls_produce_valid_grids():
    walls = ["top", "bottom", "left", "right"]
    for wall in walls:
        grid = build_grid_with_exit_on_wall(
            wall=wall, position=8, rows=20, cols=20, exit_width=2
        )
        assert validate_reachability(grid)
        exit_cells = list(zip(*np.where(grid == CELL_EXIT)))
        assert len(exit_cells) == 2


def test_invalid_parameters():
    with pytest.raises(ValueError, match="Invalid wall"):
        build_grid_with_exit_on_wall(wall="ceiling", position=0, rows=20, cols=20, exit_width=2)

    with pytest.raises(ValueError, match="Room dimensions"):
        build_grid_with_exit_on_wall(wall="top", position=0, rows=4, cols=4, exit_width=2)

    with pytest.raises(ValueError, match="exit_width"):
        build_grid_with_exit_on_wall(wall="top", position=0, rows=10, cols=10, exit_width=20)
