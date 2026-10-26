import json
import os
import hashlib

import numpy as np

from src.simulation import EvacuationSimulation


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "baseline_snapshot.json")


def _sha256_history(history):
    arr = np.asarray(history, dtype=np.int64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_baseline_regression():
    with open(DATA_PATH) as f:
        data = json.load(f)
    for case in data["cases"]:
        args = case["args"]
        grid = np.array(args["floor_plan"], dtype=int)
        sim = EvacuationSimulation(
            floor_plan=grid,
            n_agents=args["n_agents"],
            seed=args["seed"],
            panic_enabled=args["panic_enabled"],
        )
        result = sim.run(max_steps=args["max_steps"])
        times = sorted(sim.evacuation_times.values())
        assert times == case["evacuation_times_sorted"], (case["layout"], args["seed"])
        assert len(times) == case["n_evacuated"]
        assert _sha256_history(result["agent_position_history"]) == case["position_history_sha256"]
