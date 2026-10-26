"""Single source of truth for experiment settings and seeds.

The seed list is explicit and recorded beside every output. Experiments never
draw from the global NumPy random state; they either pass an integer seed to
``numpy.random.default_rng`` or pass a ``Generator`` instance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

SEEDS = list(range(1000, 1200))
DEFAULT_MAX_STEPS = 600


@dataclass
class ExperimentConfig:
    """Configuration for the full E1-E7 suite."""

    out_dir: str = "experiments/out"
    seeds: List[int] = field(default_factory=lambda: list(SEEDS))
    max_steps: int = DEFAULT_MAX_STEPS
    quick: bool = False

    # E1 — exact vs Monte Carlo
    e1_m_values: Tuple[int, ...] = (1_000, 10_000, 100_000)
    e1_n_cells: int = 10

    # E2 — solver scaling
    e2_sizes: Tuple[int, ...] = (20, 40, 60, 80, 100, 150)
    e2_dense_memory_cap_mb: float = 512.0
    e2_iter_rtol: float = 1e-10

    # E3 — panic and capacity
    e3_runs: int = 100
    e3_agents: int = 40
    e3_panic_decay: float = 0.15

    # E4 — density sweep
    e4_densities: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
                                       0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
    e4_runs: int = 40
    e4_panic_decay: float = 0.15

    # E5 — exact vs heuristic exit search
    e5_runs: int = 60
    e5_top_k: int = 5

    # E6 — traits
    e6_runs: int = 80
    e6_agents: int = 60
    e6_correlations: Tuple[float, ...] = (-0.75, 0.0, 0.75)

    # E7 — robustness / Kendall tau
    e7_runs: int = 30
    e7_agents: int = 30

    def seeds_for(self, n: int) -> List[int]:
        n = int(n)
        if n < 1:
            raise ValueError("n must be positive")
        if n > len(self.seeds):
            # Extend deterministically rather than silently reusing seeds.
            start = self.seeds[-1] + 1 if self.seeds else 0
            self.seeds.extend(range(start, start + (n - len(self.seeds))))
        return list(self.seeds[:n])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def quick_config(cls, **overrides: Any) -> "ExperimentConfig":
        """A fast configuration for smoke tests and development runs."""
        cfg = cls(
            quick=True,
            e1_m_values=(200, 2_000, 20_000),
            e1_n_cells=5,
            e2_sizes=(20, 40, 60),
            e3_runs=12,
            e4_runs=8,
            e5_runs=10,
            e6_runs=10,
            e7_runs=6,
        )
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise ValueError(f"unknown config field: {key}")
            setattr(cfg, key, value)
        return cfg


DEFAULT_CONFIG = ExperimentConfig()
