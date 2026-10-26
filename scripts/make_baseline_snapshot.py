import os
import sys
import json
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.floor_plan import get_simple_room, get_room_with_pillar, get_hallway
from src.simulation import EvacuationSimulation

SEEDS = [1000, 1001]
N_AGENTS = 20
MAX_STEPS = 200


def _sha256_history(history):
    arr = np.asarray(history, dtype=np.int64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main():
    layouts = {
        "simple_room": get_simple_room(12, 12, exit_width=2),
        "l_hallway": get_hallway(15, 15, corridor_width=2, exit_width=1),
        "pillar_room": get_room_with_pillar(15, 15, exit_width=2, pillar_size=3),
    }

    cases = []
    for layout_name, grid in layouts.items():
        for panic in (False, True):
            for seed in SEEDS:
                args = {
                    "floor_plan": grid.tolist(),
                    "n_agents": N_AGENTS,
                    "seed": seed,
                    "panic_enabled": panic,
                }
                sim = EvacuationSimulation(
                    floor_plan=np.array(grid, dtype=int),
                    n_agents=N_AGENTS,
                    seed=seed,
                    panic_enabled=panic,
                )
                result = sim.run(max_steps=MAX_STEPS)
                times = sorted(sim.evacuation_times.values())
                cases.append({
                    "layout": layout_name,
                    "panic_enabled": panic,
                    "seed": seed,
                    "args": {
                        "floor_plan": grid.tolist(),
                        "n_agents": N_AGENTS,
                        "seed": seed,
                        "panic_enabled": panic,
                        "max_steps": MAX_STEPS,
                    },
                    "evacuation_times_sorted": [int(x) for x in times],
                    "n_evacuated": int(len(times)),
                    "position_history_sha256": _sha256_history(result["agent_position_history"]),
                })

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_snapshot.json")
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f, indent=2)
    print(f"wrote {out_path} with {len(cases)} cases")


if __name__ == "__main__":
    main()
