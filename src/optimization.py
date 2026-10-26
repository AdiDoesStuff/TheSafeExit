"""
optimization.py — Exit placement: exhaustive search and a heuristic baseline.

Two objectives are used here:

* ``rational_max_distance`` — the maximum BFS distance from a walkable cell to
  the nearest exit. For the rational (steepest-descent) chain this equals the
  worst-case expected absorption time exactly, and it is cheap to evaluate.
* ensemble T90 — a stochastic check of a shortlisted design.

The exhaustive search in :func:`exact_exit_search` enumerates every legal
boundary position and is therefore guaranteed optimal for the chosen discrete
objective on the enumerated set. :func:`optimize_exit_placement` (BFGS over a
rounded interpolation) is retained as a *heuristic search* baseline so E5 can
report where it misses the true optimum.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from src.floor_plan import (
    CELL_EXIT,
    CELL_WALKABLE,
    CELL_WALL,
    compute_exit_distance_field,
    validate_reachability,
)
from src.markov import build_rational_QR, build_state_map
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


def rational_max_distance(grid: np.ndarray) -> float:
    """
    Worst-case BFS distance from a walkable cell to the nearest exit.

    For the rational (steepest-descent) chain, every step reduces the BFS
    distance by exactly one, so the absorption time is deterministic and equals
    this distance. This is the cheap objective used to enumerate exit positions.
    """
    validate_reachability(grid)
    dist_field = compute_exit_distance_field(grid)
    walkable = grid == CELL_WALKABLE
    if not np.any(walkable):
        raise ValueError("Floor plan has no walkable cells.")
    return float(np.max(dist_field[walkable]))


def worst_case_evacuation_time(grid: np.ndarray) -> float:
    """
    Exact worst-case expected evacuation time under rational gradient descent.

    Since rational steepest descent has zero variance in absorption time (the
    step count is deterministic and equals geodesic distance), this evaluates
    the spectral objective

        J = max_i ((I - Q_rational)^(-1) 1)_i

    and is retained as the reference check for :func:`rational_max_distance`.
    It forms a dense fundamental matrix and is therefore only appropriate for
    small grids.
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
    Heuristic exit search: BFGS over a rounded continuous interpolation.

    This is *not* guaranteed to find the optimum. The decision variable is a
    small discrete set of legal wall positions, so the exact objective is
    piecewise constant; BFGS treats a linear interpolation of that objective
    as smooth and can stop at a local plateau. It is kept as the comparison
    baseline for E5; use :func:`exact_exit_search` for a guaranteed optimum
    over the enumerated positions.

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


# Backwards-compatible explicit name for the heuristic.
heuristic_exit_search = optimize_exit_placement


# --------------------------------------------------------------------------- #
# 2.4 Exact exit search
# --------------------------------------------------------------------------- #
_DEFAULT_WALL_ORDER = ("top", "bottom", "left", "right")


def enumerate_exit_positions(
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    obstacles: Optional[Sequence[Tuple[int, int]]] = None,
    walls: Sequence[str] = _DEFAULT_WALL_ORDER,
) -> List[Dict[str, Any]]:
    """
    Enumerate every legal exit position on the requested boundary walls.

    A position is legal when the opening lies strictly between the corner
    cells, and the resulting grid has no unreachable walkable cell. Each
    candidate is scored by :func:`rational_max_distance`.

    The returned list is ordered deterministically: by score, then wall order,
    then position.
    """
    candidates: List[Dict[str, Any]] = []
    wall_order = [str(w).strip().lower() for w in walls]
    for wall in wall_order:
        if wall not in _DEFAULT_WALL_ORDER:
            raise ValueError(f"Invalid wall '{wall}'.")
        interior_len = (cols - 2) if wall in {"top", "bottom"} else (rows - 2)
        if exit_width < 1 or exit_width > interior_len:
            raise ValueError(
                f"exit_width {exit_width} must be between 1 and interior wall "
                f"length {interior_len}."
            )
        max_pos = interior_len - exit_width
        for position in range(max_pos + 1):
            grid = build_grid_with_exit_on_wall(
                wall=wall,
                position=position,
                rows=rows,
                cols=cols,
                exit_width=exit_width,
                obstacles=obstacles,
            )
            try:
                validate_reachability(grid)
            except ValueError:
                # An obstacle can isolate the opening; such a position is not
                # a legal design and is simply omitted.
                continue
            candidates.append(
                {
                    "wall": wall,
                    "position": int(position),
                    "grid": grid,
                    "max_distance": rational_max_distance(grid),
                }
            )

    if not candidates:
        raise ValueError("No legal exit positions exist for this layout.")

    order = {wall: i for i, wall in enumerate(_DEFAULT_WALL_ORDER)}
    candidates.sort(
        key=lambda c: (c["max_distance"], order[c["wall"]], c["position"])
    )
    return candidates


def _candidate_public(candidate: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "wall": candidate["wall"],
        "position": int(candidate["position"]),
        "max_distance": float(candidate["max_distance"]),
    }
    if "grid" in candidate:
        out["grid"] = candidate["grid"]
    if "ensemble" in candidate:
        out["ensemble"] = candidate["ensemble"]
    if "ensemble_rank" in candidate:
        out["ensemble_rank"] = int(candidate["ensemble_rank"])
    return out


def exact_exit_search(
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    obstacles: Optional[Sequence[Tuple[int, int]]] = None,
    walls: Sequence[str] = _DEFAULT_WALL_ORDER,
    *,
    ensemble_top_k: int = 5,
    n_runs: int = 100,
    base_seed: int = 0,
    seeds: Optional[Sequence[int]] = None,
    max_steps: int = 600,
    sim_kwargs: Optional[Dict[str, Any]] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Exhaustive exit placement search with an optional ensemble re-score.

    Stage 1 enumerates every legal boundary position and scores it by the
    rational worst-case BFS distance. The first candidate is therefore the true
    minimum of that objective on the enumerated set.

    Stage 2 re-scores the best ``ensemble_top_k`` candidates by the median T90
    of an ensemble with common random numbers (the same seeds for every
    candidate). ``best_by_ensemble`` is the winner of that second stage; it can
    differ from ``best`` because the ensemble includes panic, capacity and
    stochastic effects that the rational objective ignores.

    Parameters
    ----------
    walls:
        Boundary walls to search. Defaults to all four.
    ensemble_top_k:
        Number of stage-1 finalists to re-score. Set to 0 to skip stage 2.
    seeds:
        Explicit seeds. Defaults to ``range(base_seed, base_seed + n_runs)``.
    sim_kwargs:
        Extra ``EvacuationSimulation`` kwargs for stage 2 (for example
        ``{"panic_enabled": False, "cell_capacity": 1, "n_agents": 40}``).
    """
    if ensemble_top_k < 0:
        raise ValueError("ensemble_top_k must be non-negative.")
    if n_runs <= 0:
        raise ValueError("n_runs must be positive.")

    candidates = enumerate_exit_positions(
        rows=rows,
        cols=cols,
        exit_width=exit_width,
        obstacles=obstacles,
        walls=walls,
    )
    best = candidates[0]
    objective_value = float(best["max_distance"])

    n_finalists = min(int(ensemble_top_k), len(candidates))
    finalists = candidates[:n_finalists]
    if seeds is None:
        seeds = list(range(int(base_seed), int(base_seed) + int(n_runs)))
    seeds = [int(s) for s in seeds]
    if len(seeds) != int(n_runs):
        raise ValueError("len(seeds) must equal n_runs")

    best_by_ensemble: Optional[Dict[str, Any]] = None
    if finalists:
        from src.ensemble import run_ensemble

        kwargs = dict(sim_kwargs or {})
        for candidate in finalists:
            result = run_ensemble(
                candidate["grid"],
                n_runs=int(n_runs),
                base_seed=int(base_seed),
                seeds=seeds,
                max_steps=int(max_steps),
                **kwargs,
            )
            summary = result.summary()
            t90 = summary["T90"]
            candidate["ensemble"] = {
                "T90_median": t90["median"],
                "T90_bootstrap_ci": t90["bootstrap_ci"],
                "T100_median": summary["T100"]["median"],
                "fraction_finished": summary["fraction_finished"],
                "fraction_evacuated_mean": summary["fraction_evacuated_mean"],
                "n_runs": int(n_runs),
                "base_seed": int(base_seed),
            }
            if progress is not None:
                progress(candidate)

        # Rank by T90 (NaN last), then by stage-1 score for a stable tie-break.
        scored = sorted(
            finalists,
            key=lambda c: (
                np.isnan(c["ensemble"]["T90_median"]),
                c["ensemble"]["T90_median"]
                if not np.isnan(c["ensemble"]["T90_median"])
                else float("inf"),
                c["max_distance"],
                c["wall"],
                c["position"],
            ),
        )
        for rank, candidate in enumerate(scored, start=1):
            candidate["ensemble_rank"] = rank
        best_by_ensemble = scored[0]

    return {
        "objective": "min_max_rational_bfs_distance",
        "rows": int(rows),
        "cols": int(cols),
        "exit_width": int(exit_width),
        "obstacles": list(obstacles) if obstacles is not None else None,
        "n_candidates": len(candidates),
        "candidates": [_candidate_public(c) for c in candidates],
        "best": _candidate_public(best),
        "best_score": objective_value,
        "optimal_for_objective": True,
        "best_by_ensemble": _candidate_public(best_by_ensemble)
        if best_by_ensemble is not None
        else None,
        "ensemble_top_k": int(ensemble_top_k),
        "n_runs": int(n_runs) if finalists else 0,
        "base_seed": int(base_seed),
        "max_steps": int(max_steps),
        "sim_kwargs": dict(sim_kwargs or {}),
    }
