"""
TheSafeExit — Stochastic Crowd Evacuation Modeling package.
"""
from src.floor_plan import (
    compute_exit_distance_field,
    get_simple_room,
    get_two_exit_room,
    get_room_with_pillar,
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

__version__ = "0.1.0"
__all__ = [
    "compute_exit_distance_field",
    "get_simple_room",
    "get_two_exit_room",
    "get_room_with_pillar",
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
]


