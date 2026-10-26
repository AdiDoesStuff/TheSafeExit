"""
TheSafeExit — Stochastic Crowd Evacuation Modeling package.
"""
from src.floor_plan import (
    compute_exit_distance_field,
    get_simple_room,
    get_two_exit_room,
    get_room_with_pillar,
    get_room_with_barrier,
    get_hallway,
    validate_reachability,
    plot_floor_plan,
)
from src.markov import build_state_map, build_QR, build_uniform_QR, build_rational_QR, build_mixed_QR
from src.panic import PanicField, UnstableTimestepError, compute_agent_density
from src.simulation import EvacuationSimulation
from src.heatmap_sequence import (
    global_colorscale,
    plot_heatmap_grid,
    animate_heatmap_sequence,
    track_bottleneck_migration,
)

from src.solvers import (
    solve_fundamental_lu,
    solve_fundamental_svd,
    fundamental_matrix_lu,
    fundamental_matrix_svd,
    benchmark_solvers,
    run_monte_carlo,
)
from src.optimization import (
    build_grid_with_exit_on_wall,
    worst_case_evacuation_time,
    optimize_exit_placement,
)
from src.ensemble import run_ensemble, EnsembleResult, paired_difference

__version__ = "0.1.0"
__all__ = [
    "compute_exit_distance_field",
    "get_simple_room",
    "get_two_exit_room",
    "get_room_with_pillar",
    "get_room_with_barrier",
    "get_hallway",
    "validate_reachability",
    "plot_floor_plan",
    "build_state_map",
    "build_QR",
    "build_uniform_QR",
    "build_rational_QR",
    "build_mixed_QR",
    "PanicField",
    "UnstableTimestepError",
    "compute_agent_density",
    "EvacuationSimulation",
    "global_colorscale",
    "plot_heatmap_grid",
    "animate_heatmap_sequence",
    "track_bottleneck_migration",
    "solve_fundamental_lu",
    "solve_fundamental_svd",
    "fundamental_matrix_lu",
    "fundamental_matrix_svd",
    "benchmark_solvers",
    "run_monte_carlo",
    "build_grid_with_exit_on_wall",
    "worst_case_evacuation_time",
    "optimize_exit_placement",
    "run_ensemble",
    "EnsembleResult",
    "paired_difference",
]


