import numpy as np
import pytest

from src.floor_plan import get_simple_room
from src.optimization import (
    build_grid_with_exit_on_wall,
    enumerate_exit_positions,
    exact_exit_search,
    heuristic_exit_search,
    optimize_exit_placement,
    rational_max_distance,
    worst_case_evacuation_time,
)


def test_rational_max_distance_equals_spectral_objective():
    for grid in (
        get_simple_room(10, 10, exit_width=2),
        build_grid_with_exit_on_wall("right", 0, rows=12, cols=12, exit_width=2),
        build_grid_with_exit_on_wall("top", 3, rows=11, cols=14, exit_width=1),
    ):
        assert rational_max_distance(grid) == pytest.approx(
            worst_case_evacuation_time(grid), abs=1e-9
        )


def test_enumerate_counts_and_hand_checked_optimum_7x7_width1():
    candidates = enumerate_exit_positions(rows=7, cols=7, exit_width=1)
    # Five interior positions on each of four walls.
    assert len(candidates) == 20
    # For a 5x5 interior and one exit cell, the centre position is optimal and
    # the worst distance is 5 (horizontal) + 2 (vertical half-span) = 7.
    assert candidates[0]["max_distance"] == pytest.approx(7.0)
    assert candidates[1]["max_distance"] >= candidates[0]["max_distance"]
    # Deterministic ordering: score, then top/bottom/left/right, then position.
    assert (candidates[0]["wall"], candidates[0]["position"]) == ("top", 2)


def test_enumerate_hand_checked_optimum_7x7_width2():
    candidates = enumerate_exit_positions(rows=7, cols=7, exit_width=2)
    # Four walls x four legal starts (interior length 5, width 2).
    assert len(candidates) == 16
    assert candidates[0]["max_distance"] == pytest.approx(7.0)
    # Positions 1 and 2 both centre a two-cell opening around the middle row.
    assert candidates[0]["wall"] == "top"
    assert candidates[0]["position"] in (1, 2)


def test_exact_search_returns_true_minimum_without_ensemble():
    result = exact_exit_search(rows=7, cols=7, exit_width=1, ensemble_top_k=0)
    assert result["optimal_for_objective"] is True
    assert result["n_candidates"] == 20
    assert result["best_score"] == pytest.approx(7.0)
    assert result["best"]["max_distance"] == min(
        c["max_distance"] for c in result["candidates"]
    )
    assert result["best_by_ensemble"] is None


def test_exact_search_ensemble_rescore_uses_common_seeds():
    result = exact_exit_search(
        rows=7,
        cols=7,
        exit_width=1,
        ensemble_top_k=3,
        n_runs=6,
        base_seed=100,
        max_steps=80,
        sim_kwargs={"n_agents": 6, "panic_enabled": False},
    )
    assert result["best_by_ensemble"] is not None
    assert result["n_candidates"] == 20
    assert result["best_by_ensemble"]["ensemble"]["n_runs"] == 6
    assert result["best_by_ensemble"]["ensemble"]["base_seed"] == 100
    ranked = sorted(
        [c for c in result["candidates"] if "ensemble" in c],
        key=lambda c: c["ensemble_rank"],
    )
    assert len(ranked) == 3
    assert [c["ensemble_rank"] for c in ranked] == [1, 2, 3]
    # Same seeds for each finalist: the ensemble block records them.
    for candidate in ranked:
        assert candidate["ensemble"]["n_runs"] == 6


def test_exact_search_rejects_bad_ensemble_top_k():
    with pytest.raises(ValueError, match="non-negative"):
        exact_exit_search(rows=7, cols=7, exit_width=1, ensemble_top_k=-1)


def test_heuristic_alias_is_the_bfgs_baseline():
    assert heuristic_exit_search is optimize_exit_placement
    assert "heuristic" in (optimize_exit_placement.__doc__ or "").lower()


def test_bfgs_heuristic_is_not_worse_than_naive_corner():
    # The heuristic should at least improve the deliberately bad corner seed.
    bad = build_grid_with_exit_on_wall("right", 0, rows=12, cols=12, exit_width=2)
    result = optimize_exit_placement(
        rows=12,
        cols=12,
        exit_width=2,
        initial_wall_positions={"right": 0.0, "left": 0.0, "top": 0.0, "bottom": 0.0},
    )
    assert result["best_J"] <= rational_max_distance(bad)
