"""
TheSafeExit — Stochastic Crowd Evacuation Modeling package.
"""
from src.floor_plan import compute_exit_distance_field
from src.markov import build_state_map, build_QR, build_uniform_QR, build_rational_QR, build_mixed_QR
from src.panic import PanicField, UnstableTimestepError, compute_agent_density
from src.simulation import EvacuationSimulation

__version__ = "0.1.0"
__all__ = [
    "compute_exit_distance_field",
    "build_state_map",
    "build_QR",
    "build_uniform_QR",
    "build_rational_QR",
    "build_mixed_QR",
    "PanicField",
    "UnstableTimestepError",
    "compute_agent_density",
    "EvacuationSimulation",
]

