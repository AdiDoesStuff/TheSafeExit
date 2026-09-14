"""
TheSafeExit — Stochastic Crowd Evacuation Modeling package.
"""
from src.panic import PanicField, UnstableTimestepError, compute_agent_density

__version__ = "0.1.0"
__all__ = [
    "PanicField",
    "UnstableTimestepError",
    "compute_agent_density",
]
