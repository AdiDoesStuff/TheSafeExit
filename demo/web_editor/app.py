"""
app.py — TheSafeExit Web Editor Backend
Flask API server bridging the HTML grid editor to the Python analysis pipeline.

Endpoints:
    POST /api/analyze      — Full analysis: reachability, distance field, Markov chain,
                             fundamental matrix, expected evacuation times.
    POST /api/monte_carlo  — Monte Carlo sampling for a given start cell.
    POST /api/optimize     — Exit-placement BFGS optimizer (runs on server; may take seconds).
    GET  /                 — Serves the editor HTML page.
"""

from __future__ import annotations

import json
import sys
import os
import traceback

# Ensure the project root is on the Python path so `src` imports resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.floor_plan import (
    CELL_WALL,
    CELL_WALKABLE,
    CELL_EXIT,
    validate_reachability,
    compute_exit_distance_field,
)
from src.markov import build_state_map, build_QR, build_rational_QR
from src.solvers import fundamental_matrix_lu, run_monte_carlo
from src.simulation import EvacuationSimulation
from src.heatmap_sequence import global_colorscale

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
EDITOR_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=EDITOR_DIR)
CORS(app)  # Allow cross-origin requests during development


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_grid(payload: dict) -> np.ndarray:
    """Convert the JSON grid payload (list-of-lists) into a numpy int array."""
    raw = payload.get("grid")
    if raw is None:
        raise ValueError("Request body must include a 'grid' field.")
    grid = np.array(raw, dtype=int)
    if grid.ndim != 2:
        raise ValueError(f"'grid' must be a 2-D array, got shape {grid.shape}.")
    rows, cols = grid.shape
    if rows < 5 or cols < 5:
        raise ValueError(f"Grid must be at least 5x5, got {rows}x{cols}.")
    unique_vals = set(np.unique(grid).tolist())
    valid_vals = {CELL_WALL, CELL_WALKABLE, CELL_EXIT}
    bad = unique_vals - valid_vals
    if bad:
        raise ValueError(
            f"Grid contains invalid cell values: {bad}. Use 0 (wall), 1 (walkable), 2 (exit)."
        )
    return grid


def _numpy_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(EDITOR_DIR, "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full pipeline analysis on a custom grid.

    Request JSON:
        {
          "grid": [[int, ...], ...],
          "chain": "rational" | "uniform"   (default: "rational")
        }

    Response JSON:
        {
          "ok": true,
          "rows": int, "cols": int,
          "n_transient": int, "n_absorbing": int,
          "distance_field": [[int, ...]],
          "expected_times": [[float|null, ...]],
          "worst_case_time": float,
          "mean_time": float,
          "chain": str
        }
    """
    try:
        payload = request.get_json(force=True)
        grid = _parse_grid(payload)
        chain = payload.get("chain", "rational").strip().lower()
        if chain not in {"rational", "uniform"}:
            chain = "rational"

        rows, cols = grid.shape

        validate_reachability(grid)
        dist_field = compute_exit_distance_field(grid)
        state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)

        if chain == "rational":
            Q, R = build_rational_QR(grid, dist_field, state_map, n_transient, n_absorbing)
        else:
            Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

        N = fundamental_matrix_lu(Q)
        row_sums = np.sum(N, axis=1)

        expected_times = [[None] * cols for _ in range(rows)]
        for idx, t_val in enumerate(row_sums):
            r, c = index_to_state[idx]
            expected_times[r][c] = round(float(t_val), 4)

        worst_case = float(np.max(row_sums))
        mean_time = float(np.mean(row_sums))

        return jsonify({
            "ok": True,
            "rows": rows,
            "cols": cols,
            "n_transient": n_transient,
            "n_absorbing": n_absorbing,
            "distance_field": _numpy_safe(dist_field),
            "expected_times": expected_times,
            "worst_case_time": round(worst_case, 4),
            "mean_time": round(mean_time, 4),
            "chain": chain,
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 400


@app.route("/api/monte_carlo", methods=["POST"])
def monte_carlo():
    """
    Monte Carlo absorption-time sampling for a single start cell.

    Request JSON:
        {
          "grid": [[int, ...]],
          "start_row": int,
          "start_col": int,
          "trials": int,
          "chain": "rational" | "uniform"
        }

    Response JSON:
        {
          "ok": true,
          "mean": float, "std_error": float, "trials": int,
          "chain": str, "analytic_mean": float
        }
    """
    try:
        payload = request.get_json(force=True)
        grid = _parse_grid(payload)
        chain = payload.get("chain", "rational").strip().lower()
        if chain not in {"rational", "uniform"}:
            chain = "rational"

        start_row = int(payload.get("start_row", 0))
        start_col = int(payload.get("start_col", 0))
        M = min(int(payload.get("trials", 1000)), 50000)

        rows, cols = grid.shape
        if not (0 <= start_row < rows and 0 <= start_col < cols):
            raise ValueError(f"start ({start_row},{start_col}) is outside the grid.")
        if grid[start_row, start_col] != CELL_WALKABLE:
            raise ValueError(
                f"Cell ({start_row},{start_col}) is not a walkable cell "
                f"(value={grid[start_row, start_col]}). Click a white cell."
            )

        validate_reachability(grid)
        dist_field = compute_exit_distance_field(grid)
        state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)

        if chain == "rational":
            Q, R = build_rational_QR(grid, dist_field, state_map, n_transient, n_absorbing)
        else:
            Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

        start_idx = state_map[(start_row, start_col)]
        mc_result = run_monte_carlo(Q, R, start_idx, M, rng=42)

        N = fundamental_matrix_lu(Q)
        analytic_mean = float(np.sum(N[start_idx, :]))

        return jsonify({
            "ok": True,
            "mean": round(mc_result["mean"], 4),
            "std_error": round(mc_result["std_error"], 6),
            "trials": M,
            "chain": chain,
            "analytic_mean": round(analytic_mean, 4),
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 400


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """
    Exit-placement BFGS optimizer on a custom obstacle layout.

    Request JSON:
        {
          "rows": int, "cols": int, "exit_width": int,
          "obstacles": [[r, c], ...]
        }

    Response JSON:
        {
          "ok": true,
          "best_wall": str, "best_position": int, "best_J": float,
          "wall_results": { wall: { position, J, nit, success } },
          "best_grid": [[int, ...]]
        }
    """
    try:
        from src.optimization import optimize_exit_placement

        payload = request.get_json(force=True)
        rows = int(payload.get("rows", 20))
        cols = int(payload.get("cols", 20))
        exit_width = int(payload.get("exit_width", 2))
        obstacles_raw = payload.get("obstacles", [])
        obstacles = [tuple(o) for o in obstacles_raw] if obstacles_raw else None

        if rows < 5 or cols < 5:
            raise ValueError("rows and cols must each be at least 5.")

        result = optimize_exit_placement(
            rows=rows,
            cols=cols,
            exit_width=exit_width,
            obstacles=obstacles,
        )

        wall_summary = {}
        for w, wr in result["walls"].items():
            wall_summary[w] = {
                "position": wr["rounded_position"],
                "J": round(wr["actual_J"], 4),
                "nit": wr["nit"],
                "success": wr["success"],
            }

        return jsonify({
            "ok": True,
            "best_wall": result["best_wall"],
            "best_position": result["best_position"],
            "best_J": round(result["best_J"], 4),
            "wall_results": wall_summary,
            "best_grid": _numpy_safe(result["best_grid"]),
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 400


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Runs dynamic panic-coupled evacuation simulation (EvacuationSimulation)
    and returns a time-series of frozen fundamental matrix heatmaps (N_t) and panic fields (u_t).

    Request JSON:
        {
          "grid": [[0, 1, ...], ...],
          "num_agents" / "n_agents": int (default 30),
          "beta": float (default 0.4),
          "D": float (default 0.3),
          "k": float (default 0.5),
          "max_steps": int (default 60),
          "snapshot_every": int (default 5),
          "seed": int (optional)
        }
    """
    try:
        payload = request.get_json(force=True)
        grid = _parse_grid(payload)
        rows, cols = grid.shape

        # Guard against runaway compute (simulation involves LU per snapshot)
        if rows > 40 or cols > 40:
            return jsonify({
                "ok": False,
                "error": f"Grid size ({rows}x{cols}) exceeds the 40x40 cap for dynamic panic simulation."
            }), 400

        # Validate reachability
        validate_reachability(grid)

        # Parse simulation parameters
        raw_agents = payload.get("num_agents") or payload.get("n_agents") or 30
        num_agents = max(1, min(500, int(raw_agents)))

        beta = float(payload.get("beta", 0.4))
        D = float(payload.get("D", 0.3))
        k = float(payload.get("k", 0.5))
        max_steps = max(5, min(200, int(payload.get("max_steps", 60))))
        snapshot_every = max(1, min(50, int(payload.get("snapshot_every", 5))))
        seed_val = payload.get("seed")
        seed = int(seed_val) if seed_val is not None else 42

        # Compute static rational baseline J* for comparison
        dist_field = compute_exit_distance_field(grid)
        state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)
        Q_rat, R_rat = build_rational_QR(grid, dist_field, state_map, n_transient, n_absorbing)
        N_rat = fundamental_matrix_lu(Q_rat)
        t_rat = np.sum(N_rat, axis=1)
        rational_J = float(np.max(t_rat))

        # Instantiate and run dynamic EvacuationSimulation
        sim = EvacuationSimulation(
            grid,
            n_agents=num_agents,
            beta=beta,
            D=D,
            k=k,
            panic_enabled=True,
            seed=seed,
        )

        max_snapshots = 30
        sim_result = sim.run(
            max_steps=max_steps,
            snapshot_every=snapshot_every,
            max_snapshots=max_snapshots,
        )

        snapshots = sim_result["snapshots"]

        # Ensure at least one snapshot frame
        if not snapshots:
            step_summary = sim.step(record_snapshot=True)
            if step_summary["snapshot"] is not None:
                snapshots = [step_summary["snapshot"]]

        # Global color scale for consistent frame-to-frame rendering across time series
        vmin, vmax = global_colorscale(snapshots)

        def _snap_to_json(arr):
            if arr is None:
                return None
            res = []
            for r in range(rows):
                row_vals = []
                for c in range(cols):
                    v = arr[r, c]
                    if np.isnan(v):
                        row_vals.append(None)
                    else:
                        row_vals.append(round(float(v), 4))
                res.append(row_vals)
            return res

        frames = []
        max_panic_seen = 0.0

        for snap in snapshots:
            h_2d = snap["heatmap_2d"]
            p_2d = snap.get("panic_2d")

            # Calculate worst-case and mean for walkable cells in this snapshot
            valid_h = h_2d[~np.isnan(h_2d) & (grid == CELL_WALKABLE)]
            worst_case = float(np.max(valid_h)) if len(valid_h) > 0 else 0.0
            mean_val = float(np.mean(valid_h)) if len(valid_h) > 0 else 0.0

            if p_2d is not None:
                valid_p = p_2d[grid == CELL_WALKABLE]
                if len(valid_p) > 0:
                    max_panic_seen = max(max_panic_seen, float(np.max(valid_p)))

            frame = {
                "t": snap["t"],
                "heatmap": _snap_to_json(h_2d),
                "panic": _snap_to_json(p_2d),
                "worst_case": round(worst_case, 4),
                "mean": round(mean_val, 4),
                "mean_panic": round(snap.get("mean_panic", 0.0), 4),
                "active_agents": snap.get("active_agents", 0),
            }
            frames.append(frame)

        final_worst = frames[-1]["worst_case"] if frames else rational_J
        final_mean = frames[-1]["mean"] if frames else float(np.mean(t_rat))
        mean_evac = sim_result["mean_evacuation_time"]

        return jsonify({
            "ok": True,
            "frames": frames,
            "vmin": round(float(vmin), 4),
            "vmax": round(float(vmax), 4),
            "panic_max": round(float(max_panic_seen), 4) if max_panic_seen > 0 else 1.0,
            "rational_J": round(float(rational_J), 4),
            "n_t_semantics": "frozen-chain diagnostic, not a forecast",
            "n_agents_start": num_agents,
            "final_stats": {
                "total_steps": sim_result["total_steps"],
                "evacuated_agents": sim_result["evacuated_agents"],
                "evacuation_rate": round(float(sim_result["evacuation_rate"]), 4),
                "mean_evacuation_time": round(float(mean_evac), 4) if mean_evac != float("inf") else None,
                "final_worst_case": round(float(final_worst), 4),
                "final_mean": round(float(final_mean), 4),
            }
        })

    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "trace": traceback.format_exc()}), 400


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  TheSafeExit Web Editor -> http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
