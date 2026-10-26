"""Named floor plans shared by the experiments."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from src.floor_plan import (
    get_hallway,
    get_room_with_barrier,
    get_room_with_pillar,
    get_simple_room,
    get_two_exit_room,
)
from src.optimization import build_grid_with_exit_on_wall


def e1_layouts() -> Dict[str, np.ndarray]:
    """Small fixed chains with moderate expected absorption times."""
    return {
        "simple_8x8": get_simple_room(8, 8, exit_width=2),
        "pillar_9x9": get_room_with_pillar(9, 9, exit_width=2, pillar_size=2),
        "two_exit_9x9": get_two_exit_room(9, 9, exit_width=2),
    }


def e3_layouts() -> Dict[str, np.ndarray]:
    """The two slide layouts: open room and room with a central pillar."""
    return {
        "simple_20x20": get_simple_room(20, 20, exit_width=2),
        "pillar_20x20": get_room_with_pillar(20, 20, exit_width=2, pillar_size=4),
    }


def e4_layout() -> np.ndarray:
    """The narrow L-shaped hallway used for the density sweep."""
    return get_hallway(25, 25, corridor_width=3, exit_width=1)


def e5_cases() -> List[Dict[str, Any]]:
    """Five memorable exit-placement problems for exact-vs-heuristic."""
    pillar_obstacles = [
        (r, c) for r in range(6, 8) for c in range(6, 8)
    ]
    barrier_obstacles = [(r, 7) for r in range(1, 9)]
    return [
        {"name": "simple_12x12", "rows": 12, "cols": 12, "exit_width": 2, "obstacles": None},
        {"name": "simple_16x16", "rows": 16, "cols": 16, "exit_width": 2, "obstacles": None},
        {"name": "rect_14x10", "rows": 14, "cols": 10, "exit_width": 2, "obstacles": None},
        {"name": "pillar_14x14", "rows": 14, "cols": 14, "exit_width": 2, "obstacles": pillar_obstacles},
        {"name": "barrier_14x14", "rows": 14, "cols": 14, "exit_width": 2, "obstacles": barrier_obstacles},
    ]


def e7_configurations() -> Dict[str, np.ndarray]:
    """
    Four candidate designs whose ranking is tested against parameter sweeps.

    The mix includes two exit positions and two obstacle layouts so the ranking
    is about designs, not only one room with one exit.
    """
    return {
        "simple_center": get_simple_room(12, 12, exit_width=2),
        "simple_corner": build_grid_with_exit_on_wall(
            "right", position=0, rows=12, cols=12, exit_width=2
        ),
        "pillar_center": get_room_with_pillar(12, 12, exit_width=2, pillar_size=2),
        "barrier_center": get_room_with_barrier(12, 12, exit_width=2),
    }
