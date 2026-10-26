"""
optimization.py — Phase 5: Gradient-Based Exit Placement Optimization (CO2)

Provides deterministic, closed-form exit placement optimization based on
minimizing worst-case expected evacuation time over floor-plan boundary walls.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.optimize import minimize, approx_fprime

from src.floor_plan import (
    CELL_WALL,
    CELL_WALKABLE,
    CELL_EXIT,
    validate_reachability,
    compute_exit_distance_field,
)
from src.markov import build_state_map, build_rational_QR
from src.solvers import fundamental_matrix_lu


def build_grid_with_exit_on_wall(
    wall: str,
    position: int,
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    obstacles: Optional[Sequence[Tuple[int, int]]] = None,
) -> np.ndarray:
    """
    Constructs a rectangular room layout with outer walls and an exit opening of width
    `exit_width` placed along the specified boundary wall.

    Parameters:
        wall: Boundary wall name in {"top", "bottom", "left", "right"}.
        position: 0-indexed integer starting coordinate along the wall's interior
                  (excluding the two corner cells).
        rows, cols: Dimensions of the floor plan grid (minimum 5x5).
        exit_width: Width (in cells) of the exit opening.
        obstacles: Optional list of (row, col) coordinates to mark as interior walls.

    Returns:
        grid: 2D numpy array with cell encodings (0=Wall, 1=Walkable, 2=Exit).
    """
    wall_norm = wall.strip().lower()
    if wall_norm not in {"top", "bottom", "left", "right"}:
        raise ValueError(f"Invalid wall '{wall}'. Must be one of 'top', 'bottom', 'left', 'right'.")

    if rows < 5 or cols < 5:
        raise ValueError("Room dimensions must be at least 5x5.")

    # Determine interior span along the chosen wall
    if wall_norm in {"top", "bottom"}:
        interior_len = cols - 2
    else:
        interior_len = rows - 2

    if exit_width < 1 or exit_width > interior_len:
        raise ValueError(
            f"exit_width {exit_width} must be between 1 and interior wall length {interior_len}."
        )

    # Maximum starting index ensuring exit does not overlap boundary corner cells
    max_pos = interior_len - exit_width
    pos_int = int(np.clip(round(position), 0, max_pos))

    # Base grid: fully walkable interior surrounded by boundary walls
    grid = np.full((rows, cols), CELL_WALKABLE, dtype=int)
    grid[0, :] = CELL_WALL
    grid[-1, :] = CELL_WALL
    grid[:, 0] = CELL_WALL
    grid[:, -1] = CELL_WALL

    # Place interior obstacle walls if provided
    if obstacles is not None:
        for r, c in obstacles:
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] = CELL_WALL

    # Place exit cells along the specified wall
    if wall_norm == "top":
        col_start = 1 + pos_int
        grid[0, col_start : col_start + exit_width] = CELL_EXIT
    elif wall_norm == "bottom":
        col_start = 1 + pos_int
        grid[-1, col_start : col_start + exit_width] = CELL_EXIT
    elif wall_norm == "left":
        row_start = 1 + pos_int
        grid[row_start : row_start + exit_width, 0] = CELL_EXIT
    elif wall_norm == "right":
        row_start = 1 + pos_int
        grid[row_start : row_start + exit_width, -1] = CELL_EXIT

    return grid


def worst_case_evacuation_time(grid: np.ndarray) -> float:
    """
    Computes the exact closed-form worst-case expected evacuation time across all
    walkable cells on the floor plan under rational gradient descent.

    Since Phase 4 proved that rational steepest descent has zero variance in absorption time
    (step count is deterministic and equals geodesic distance), this evaluates the closed-form
    spectral objective:
        J = max_i ( (I - Q_rational)^(-1) 1 )_i

    Parameters:
        grid: 2D numpy array representing the floor plan.

    Returns:
        J: Maximum expected steps to evacuation from any walkable starting cell.
    """
    validate_reachability(grid)
    dist_field = compute_exit_distance_field(grid)
    state_map, _, n_transient, n_absorbing = build_state_map(grid)

    Q_rational, R_rational = build_rational_QR(
        grid, dist_field, state_map, n_transient, n_absorbing
    )
    N_rational = fundamental_matrix_lu(Q_rational)

    # Row sums: expected absorption time for each transient state
    row_sums = np.sum(N_rational, axis=1)
    return float(np.max(row_sums))


def optimize_exit_placement(
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    initial_wall_positions: Optional[Dict[str, float]] = None,
    obstacles: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """
    Optimizes exit placement along room boundary walls using BFGS gradient descent
    with numerical finite-difference gradient estimation.

    Parameters:
        rows, cols: Dimensions of the room.
        exit_width: Width of the exit.
        initial_wall_positions: Optional dict specifying starting position for each wall.
                                Defaults to center position along each wall.
        obstacles: Optional list of (row, col) obstacle wall cells.

    Returns:
        results: Dictionary containing the overall best configuration, rounded integer
                 grid, and per-wall optimization trajectories.
    """
    walls = ["top", "bottom", "left", "right"]
    wall_results: Dict[str, Any] = {}

    best_wall = None
    best_actual_J = float("inf")
    best_position = 0
    best_grid = None
    raw_best_position = 0.0
    raw_best_J = float("inf")

    for wall in walls:
        interior_len = (cols - 2) if wall in {"top", "bottom"} else (rows - 2)
        max_pos = interior_len - exit_width

        if initial_wall_positions and wall in initial_wall_positions:
            x0_val = float(np.clip(initial_wall_positions[wall], 0.0, float(max_pos)))
        else:
            x0_val = float(max_pos / 2.0)

        history_J: List[float] = []
        history_pos: List[float] = []

        # Discrete integer evaluator
        def f_discrete(i: int) -> float:
            i_clamped = int(np.clip(i, 0, max_pos))
            test_grid = build_grid_with_exit_on_wall(
                wall=wall,
                position=i_clamped,
                rows=rows,
                cols=cols,
                exit_width=exit_width,
                obstacles=obstacles,
            )
            return worst_case_evacuation_time(test_grid)

        # Piecewise-linear continuous relaxation allowing line-search and gradient propagation
        def continuous_objective(p: np.ndarray | float) -> float:
            p_val = float(np.asarray(p).item())
            p_clamped = float(np.clip(p_val, 0.0, float(max_pos)))
            i0 = int(np.floor(p_clamped))
            i1 = min(i0 + 1, max_pos)
            alpha = p_clamped - i0
            if i0 == i1 or np.isclose(alpha, 0.0):
                return f_discrete(i0)
            return (1.0 - alpha) * f_discrete(i0) + alpha * f_discrete(i1)

        # Record initial point
        j0 = continuous_objective(x0_val)
        history_pos.append(x0_val)
        history_J.append(j0)

        # Iteration callback
        def record_callback(xk: np.ndarray) -> None:
            pos_k = float(np.asarray(xk).item())
            jk = continuous_objective(pos_k)
            history_pos.append(pos_k)
            history_J.append(jk)

        # Boundary-aware finite difference gradient estimation:
        # Backward difference at upper bound, forward difference at lower bound,
        # central difference in interior.
        def gradient_approx(p: np.ndarray) -> np.ndarray:
            p_val = float(np.asarray(p).item())
            if p_val >= max_pos - 0.5:
                grad = (continuous_objective(p_val) - continuous_objective(p_val - 1.0)) / 1.0
            elif p_val <= 0.5:
                grad = (continuous_objective(p_val + 1.0) - continuous_objective(p_val)) / 1.0
            else:
                grad = (continuous_objective(p_val + 0.5) - continuous_objective(p_val - 0.5)) / 1.0
            return np.array([grad])

        opt_res = minimize(
            fun=continuous_objective,
            x0=[x0_val],
            method="BFGS",
            jac=gradient_approx,
            callback=record_callback,
            options={"disp": False, "maxiter": 25},
        )

        raw_pos = float(opt_res.x[0])
        raw_j = float(opt_res.fun)

        rounded_pos = int(np.clip(round(raw_pos), 0, max_pos))
        rounded_grid = build_grid_with_exit_on_wall(
            wall=wall,
            position=rounded_pos,
            rows=rows,
            cols=cols,
            exit_width=exit_width,
            obstacles=obstacles,
        )
        actual_j = worst_case_evacuation_time(rounded_grid)

        # Ensure final state is recorded in history
        if not np.isclose(history_pos[-1], raw_pos, atol=1e-3):
            history_pos.append(raw_pos)
            history_J.append(raw_j)

        wall_results[wall] = {
            "wall": wall,
            "x0": x0_val,
            "raw_position": raw_pos,
            "raw_J": raw_j,
            "rounded_position": rounded_pos,
            "actual_J": actual_j,
            "nit": int(opt_res.nit),
            "success": bool(opt_res.success),
            "history_pos": history_pos,
            "history_J": history_J,
            "grid": rounded_grid,
        }

        if actual_j < best_actual_J:
            best_actual_J = actual_j
            best_wall = wall
            best_position = rounded_pos
            best_grid = rounded_grid
            raw_best_position = raw_pos
            raw_best_J = raw_j

    return {
        "best_wall": best_wall,
        "best_position": best_position,
        "best_J": best_actual_J,
        "raw_best_position": raw_best_position,
        "raw_best_J": raw_best_J,
        "best_grid": best_grid,
        "walls": wall_results,
    }
