"""
app.py — TheSafeExit Web Editor Backend

Flask API server bridging the HTML grid editor to the analysis pipeline.

Endpoints
---------
GET  /                    Editor page.
GET  /api/health          Liveness and configuration probe.
POST /api/analyze         Reachability, distance field, expected times from one
                          sparse solve, with residual and solve time.
POST /api/monte_carlo     Monte Carlo sampling for one start cell, against the
                          exact fixed-chain value.
POST /api/optimize        Exit placement: exhaustive optimum vs BFGS heuristic.
POST /api/simulate        Coupled panic simulation with optional decay,
                          capacity and traits.
POST /api/ensemble        Common-seed ensemble: T50/T90/T100 with bootstrap
                          intervals, used for the panic on/off comparison.

The server binds to 127.0.0.1 only. The heavy endpoints (/api/simulate and
/api/ensemble) additionally refuse requests that do not originate from the
local machine.
"""

from __future__ import annotations

import math
import os
import sys
import time
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
    compute_exit_distance_field,
    validate_reachability,
)
from src.markov import build_state_map, build_QR, build_rational_QR
from src.solvers import solve_expected_time_sparse, run_monte_carlo
from src.simulation import EvacuationSimulation
from src.ensemble import run_ensemble
from src.heatmap_sequence import global_colorscale
from src.gaussian_sampler import SIGMA_DEFAULT

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
EDITOR_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=EDITOR_DIR)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# Only the local editor origins may call the API cross-origin. The page itself
# is served same-origin, so this is purely for local tooling.
_ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "DEMO_CORS_ORIGINS",
        "http://127.0.0.1:5050,http://localhost:5050",
    ).split(",")
    if o.strip()
]
CORS(app, resources={r"/api/*": {"origins": _ALLOWED_ORIGINS}})

_LOCAL_ADDRS = {"127.0.0.1", "::1", "localhost"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fail(exc: Exception, status: int = 400):
    """Return a JSON error and log the traceback server-side only."""
    app.logger.warning("request failed: %s", exc)
    app.logger.debug("%s", traceback.format_exc())
    return jsonify({"ok": False, "error": str(exc) or exc.__class__.__name__}), status


def _require_local():
    """Return an error response when the caller is not on this machine."""
    if request.remote_addr not in _LOCAL_ADDRS:
        return _fail(ValueError("This endpoint is only available from localhost."), 403)
    return None


def _parse_grid(payload: dict) -> np.ndarray:
    """Convert the JSON grid payload (list-of-lists) into a numpy int array."""
    raw = payload.get("grid")
    if raw is None:
        raise ValueError("Request body must include a 'grid' field.")
    try:
        grid = np.array(raw, dtype=int)
    except (TypeError, ValueError):
        raise ValueError("'grid' must be a rectangular 2-D array of integers.")
    if grid.ndim != 2:
        raise ValueError(f"'grid' must be a 2-D array, got shape {grid.shape}.")
    rows, cols = grid.shape
    if rows < 5 or cols < 5:
        raise ValueError(f"Grid must be at least 5x5, got {rows}x{cols}.")
    if rows > 80 or cols > 80:
        raise ValueError(f"Grid must be at most 80x80, got {rows}x{cols}.")
    unique_vals = set(np.unique(grid).tolist())
    valid_vals = {CELL_WALL, CELL_WALKABLE, CELL_EXIT}
    bad = unique_vals - valid_vals
    if bad:
        raise ValueError(
            f"Grid contains invalid cell values: {sorted(bad)}. "
            "Use 0 (wall), 1 (walkable), 2 (exit)."
        )
    return grid


def _float(payload: dict, key: str, default: float, lo: float, hi: float) -> float:
    raw = payload.get(key, default)
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not np.isfinite(val):
        raise ValueError(f"'{key}' must be finite.")
    if val < lo or val > hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return val


def _int(payload: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = payload.get(key, default)
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be an integer.")
    if val < lo or val > hi:
        raise ValueError(f"'{key}' must be between {lo} and {hi}.")
    return val


def _optional_capacity(payload: dict, key: str) -> int | None:
    raw = payload.get(key)
    if raw in (None, "", 0, "0", False):
        return None
    try:
        val = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a positive integer or null.")
    if val < 1 or val > 20:
        raise ValueError(f"'{key}' must be between 1 and 20.")
    return val


def _trait_sigma(rho: float) -> np.ndarray:
    """Covariance for (speed, reaction time) at a given correlation.

    Standard deviations are taken from SIGMA_DEFAULT (0.20 m/s and 0.10 s), so
    rho = +0.75 reproduces the default covariance used by the experiments.
    """
    sd = np.sqrt(np.diag(SIGMA_DEFAULT))
    cov = rho * sd[0] * sd[1]
    return np.array([[sd[0] ** 2, cov], [cov, sd[1] ** 2]], dtype=float)


def _sim_kwargs(payload: dict) -> dict:
    """Build EvacuationSimulation kwargs from optional scenario fields.

    Defaults preserve the original behaviour: no decay, no capacity, no traits.
    """
    kwargs: dict = {}

    panic_on = payload.get("panic_on", True)
    if not isinstance(panic_on, bool):
        raise ValueError("'panic_on' must be true or false.")
    kwargs["panic_enabled"] = panic_on

    if "panic_decay" in payload:
        decay = _float(payload, "panic_decay", 0.0, 0.0, 0.9)
        if decay > 0:
            kwargs["panic_decay"] = decay

    cell_capacity = _optional_capacity(payload, "cell_capacity")
    if cell_capacity is not None:
        kwargs["cell_capacity"] = cell_capacity
    exit_capacity = _optional_capacity(payload, "exit_capacity")
    if exit_capacity is not None:
        kwargs["exit_capacity"] = exit_capacity

    traits = payload.get("traits", False)
    if not isinstance(traits, bool):
        raise ValueError("'traits' must be true or false.")
    if traits:
        rho = _float(payload, "trait_rho", 0.75, -0.95, 0.95)
        kwargs["use_traits"] = True
        kwargs["trait_sigma"] = _trait_sigma(rho)

    return kwargs


def _numpy_safe(obj):
    """Recursively convert numpy types to native Python for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_numpy_safe(x) for x in obj]
    return obj


def _quantiles(times, n_agents: int) -> dict:
    """T50/T90/T100 of the observed evacuation times (None when censored)."""
    ordered = sorted(times)
    out = {}
    for name, p in (("T50", 0.5), ("T90", 0.9), ("T100", 1.0)):
        m = int(math.ceil(p * n_agents))
        out[name] = float(ordered[m - 1]) if len(ordered) >= m else None
    return out


def _blocked_summary(blocked: np.ndarray, top: int = 5) -> list:
    """Largest blocked-move cells as (row, col, count), for the UI readout."""
    arr = np.asarray(blocked)
    if arr.size == 0 or arr.max() <= 0:
        return []
    flat = np.dstack(np.unravel_index(np.argsort(arr.ravel())[::-1][:top], arr.shape))[0]
    return [
        {"row": int(r), "col": int(c), "count": int(arr[r, c])}
        for r, c in flat
        if arr[r, c] > 0
    ]


def _build_chain(grid: np.ndarray, chain: str):
    validate_reachability(grid)
    dist_field = compute_exit_distance_field(grid)
    state_map, index_to_state, n_transient, n_absorbing = build_state_map(grid)
    if chain == "rational":
        Q, R = build_rational_QR(grid, dist_field, state_map, n_transient, n_absorbing)
    else:
        Q, R = build_QR(grid, state_map, n_transient, n_absorbing)
    return {
        "dist_field": dist_field,
        "state_map": state_map,
        "index_to_state": index_to_state,
        "n_transient": n_transient,
        "n_absorbing": n_absorbing,
        "Q": Q,
        "R": R,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(EDITOR_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "app": "TheSafeExit web editor",
        "version": "1.0",
        "port": int(os.environ.get("PORT", 5050)),
        "cors_origins": _ALLOWED_ORIGINS,
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Full pipeline analysis on a custom grid.

    Request JSON:
        {"grid": [[int, ...], ...], "chain": "rational" | "uniform"}

    Response JSON:
        ok, rows, cols, n_transient, n_absorbing, distance_field,
        expected_times, worst_case_time, mean_time, chain,
        numerics: {method, residual, seconds, n_states}
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        grid = _parse_grid(payload)
        chain = str(payload.get("chain", "rational")).strip().lower()
        if chain not in {"rational", "uniform"}:
            chain = "rational"

        rows, cols = grid.shape
        built = _build_chain(grid, chain)
        report = solve_expected_time_sparse(built["Q"])
        if report.residual >= 1e-8:
            return jsonify({
                "ok": False,
                "error": (
                    f"The linear solve did not converge: relative residual "
                    f"{report.residual:.2e} (needs < 1e-8)."
                ),
            }), 422

        expected_times = [[None] * cols for _ in range(rows)]
        for idx, t_val in enumerate(report.t):
            r, c = built["index_to_state"][idx]
            expected_times[r][c] = round(float(t_val), 4)

        return jsonify({
            "ok": True,
            "rows": rows,
            "cols": cols,
            "n_transient": built["n_transient"],
            "n_absorbing": built["n_absorbing"],
            "distance_field": _numpy_safe(built["dist_field"]),
            "expected_times": expected_times,
            "worst_case_time": round(float(np.max(report.t)), 4),
            "mean_time": round(float(np.mean(report.t)), 4),
            "chain": chain,
            "numerics": {
                "method": report.method,
                "residual": float(report.residual),
                "seconds": float(report.seconds),
                "n_states": int(report.n),
            },
        })

    except Exception as exc:
        return _fail(exc)


@app.route("/api/monte_carlo", methods=["POST"])
def monte_carlo():
    """
    Monte Carlo absorption-time sampling for a single start cell.

    Request JSON:
        {grid, start_row, start_col, trials, chain}

    Response JSON:
        ok, mean, std_error, trials, chain, analytic_mean, analytic_seconds,
        difference, relative_error, n_transient
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
        grid = _parse_grid(payload)
        chain = str(payload.get("chain", "rational")).strip().lower()
        if chain not in {"rational", "uniform"}:
            chain = "rational"

        rows, cols = grid.shape
        start_row = _int(payload, "start_row", 0, 0, rows - 1)
        start_col = _int(payload, "start_col", 0, 0, cols - 1)
        M = _int(payload, "trials", 2000, 10, 50000)

        if grid[start_row, start_col] != CELL_WALKABLE:
            raise ValueError(
                f"Cell ({start_row},{start_col}) is not a walkable cell "
                f"(value={grid[start_row, start_col]}). Click a white cell."
            )

        built = _build_chain(grid, chain)
        start_idx = built["state_map"][(start_row, start_col)]
        mc_result = run_monte_carlo(built["Q"], built["R"], start_idx, M, rng=42)
        analytic = solve_expected_time_sparse(built["Q"])

        analytic_mean = float(analytic.t[start_idx])
        mean = float(mc_result["mean"])
        diff = mean - analytic_mean

        return jsonify({
            "ok": True,
            "mean": round(mean, 4),
            "std_error": round(float(mc_result["std_error"]), 6),
            "trials": M,
            "chain": chain,
            "analytic_mean": round(analytic_mean, 4),
            "analytic_seconds": float(analytic.seconds),
            "difference": round(diff, 4),
            "relative_error": round(abs(diff) / analytic_mean, 5) if analytic_mean else None,
            "n_transient": int(built["n_transient"]),
        })

    except Exception as exc:
        return _fail(exc)


@app.route("/api/optimize", methods=["POST"])
def optimize():
    """
    Exit placement on a custom obstacle layout.

    Stage 1 enumerates every legal boundary position and minimises the rational
    worst-case BFS distance (guaranteed optimal on the enumerated set). The BFGS
    search is run as a heuristic baseline for comparison, and is skipped on
    large grids where it would take too long for an interactive call.

    Request JSON:
        {rows, cols, exit_width, obstacles: [[r, c], ...]}

    Response JSON:
        ok, exact: {wall, position, score, n_candidates, seconds},
        heuristic: {wall, position, J, seconds} | null,
        gap_steps: float | null, notes: [str],
        best_wall / best_position / best_J / wall_results / best_grid
        (heuristic, kept for backwards compatibility), exact_grid
    """
    try:
        from src.optimization import exact_exit_search, heuristic_exit_search

        payload = request.get_json(force=True, silent=True) or {}
        rows = _int(payload, "rows", 20, 5, 60)
        cols = _int(payload, "cols", 20, 5, 60)
        exit_width = _int(payload, "exit_width", 2, 1, 10)
        obstacles_raw = payload.get("obstacles") or []
        obstacles = [tuple(int(v) for v in o) for o in obstacles_raw if len(o) == 2]

        notes = []

        t0 = time.perf_counter()
        exact = exact_exit_search(
            rows=rows,
            cols=cols,
            exit_width=exit_width,
            obstacles=obstacles or None,
            ensemble_top_k=0,  # stage 1 only: instant, deterministic
        )
        exact_seconds = time.perf_counter() - t0
        best_exact = exact["best"]

        heuristic_result = None
        heuristic_seconds = None
        if rows * cols <= 1600:
            t0 = time.perf_counter()
            heuristic_result = heuristic_exit_search(
                rows=rows,
                cols=cols,
                exit_width=exit_width,
                obstacles=obstacles or None,
            )
            heuristic_seconds = time.perf_counter() - t0
        else:
            notes.append(
                "BFGS skipped: the heuristic needs many dense solves and is "
                "limited to grids up to 40x40 in the editor."
            )

        response = {
            "ok": True,
            "exact": {
                "wall": best_exact["wall"],
                "position": best_exact["position"],
                "score": float(exact["best_score"]),
                "n_candidates": int(exact["n_candidates"]),
                "seconds": round(exact_seconds, 4),
                "objective": exact["objective"],
            },
            "heuristic": None,
            "gap_steps": None,
            "notes": notes,
            "exact_grid": _numpy_safe(best_exact["grid"]),
        }

        if heuristic_result is not None:
            wall_summary = {}
            for w, wr in heuristic_result["walls"].items():
                wall_summary[w] = {
                    "position": wr["rounded_position"],
                    "J": round(wr["actual_J"], 4),
                    "nit": int(wr["nit"]),
                    "success": bool(wr["success"]),
                }
            response.update({
                "heuristic": {
                    "wall": heuristic_result["best_wall"],
                    "position": heuristic_result["best_position"],
                    "J": round(float(heuristic_result["best_J"]), 4),
                    "seconds": round(heuristic_seconds, 4),
                },
                "gap_steps": round(
                    float(heuristic_result["best_J"]) - float(exact["best_score"]), 4
                ),
                # Backwards-compatible fields.
                "best_wall": heuristic_result["best_wall"],
                "best_position": heuristic_result["best_position"],
                "best_J": round(float(heuristic_result["best_J"]), 4),
                "wall_results": wall_summary,
                "best_grid": _numpy_safe(heuristic_result["best_grid"]),
            })

        return jsonify(response)

    except Exception as exc:
        return _fail(exc)


@app.route("/api/simulate", methods=["POST"])
def simulate():
    """
    Coupled panic simulation with frozen-chain heatmap snapshots.

    Request JSON (only ``grid`` is required):
        grid, n_agents (30), beta (0.4), D (0.3), k (0.5), max_steps (60),
        snapshot_every (5), seed (42), panic_on (true), panic_decay (0),
        cell_capacity (null), exit_capacity (null), traits (false),
        trait_rho (0.75)

    Response JSON: frames, color scale, rational_J, final_stats (now including
    T50/T90/T100, blocked_total and a blocked-cell summary), scenario echo.
    """
    gate = _require_local()
    if gate is not None:
        return gate
    try:
        payload = request.get_json(force=True, silent=True) or {}
        grid = _parse_grid(payload)
        rows, cols = grid.shape

        if rows > 40 or cols > 40:
            raise ValueError(
                f"Grid size ({rows}x{cols}) exceeds the 40x40 cap for the dynamic "
                "panic simulation. Use Analyze for larger grids."
            )
        validate_reachability(grid)

        raw_agents = payload.get("num_agents") or payload.get("n_agents") or 30
        num_agents = max(1, min(300, int(raw_agents)))

        beta = _float(payload, "beta", 0.4, 0.0, 5.0)
        D = _float(payload, "D", 0.3, 0.01, 1.0)
        k = _float(payload, "k", 0.5, 0.0, 20.0)
        max_steps = _int(payload, "max_steps", 60, 5, 200)
        snapshot_every = _int(payload, "snapshot_every", 5, 1, 50)
        seed = _int(payload, "seed", 42, 0, 2**32 - 1)

        scenario = _sim_kwargs(payload)

        # Static rational baseline for comparison.
        built = _build_chain(grid, "rational")
        baseline = solve_expected_time_sparse(built["Q"])
        rational_J = float(np.max(baseline.t))

        sim = EvacuationSimulation(
            grid,
            n_agents=num_agents,
            beta=beta,
            D=D,
            k=k,
            seed=seed,
            **scenario,
        )

        t0 = time.perf_counter()
        sim_result = sim.run(max_steps=max_steps, snapshot_every=snapshot_every, max_snapshots=30)
        seconds = time.perf_counter() - t0

        snapshots = sim_result["snapshots"]
        if not snapshots:
            step_summary = sim.step(record_snapshot=True)
            if step_summary["snapshot"] is not None:
                snapshots = [step_summary["snapshot"]]

        vmin, vmax = global_colorscale(snapshots)

        def _snap_to_json(arr):
            if arr is None:
                return None
            out = []
            for r in range(rows):
                row_vals = []
                for c in range(cols):
                    v = arr[r, c]
                    row_vals.append(None if np.isnan(v) else round(float(v), 4))
                out.append(row_vals)
            return out

        frames = []
        max_panic_seen = 0.0
        for snap in snapshots:
            h_2d = snap["heatmap_2d"]
            p_2d = snap.get("panic_2d")

            valid_h = h_2d[~np.isnan(h_2d) & (grid == CELL_WALKABLE)]
            worst_case = float(np.max(valid_h)) if len(valid_h) > 0 else 0.0
            mean_val = float(np.mean(valid_h)) if len(valid_h) > 0 else 0.0

            if p_2d is not None:
                valid_p = p_2d[grid == CELL_WALKABLE]
                if len(valid_p) > 0:
                    max_panic_seen = max(max_panic_seen, float(np.max(valid_p)))

            frames.append({
                "t": snap["t"],
                "heatmap": _snap_to_json(h_2d),
                "panic": _snap_to_json(p_2d),
                "worst_case": round(worst_case, 4),
                "mean": round(mean_val, 4),
                "mean_panic": round(snap.get("mean_panic", 0.0), 4),
                "active_agents": snap.get("active_agents", 0),
            })

        final_worst = frames[-1]["worst_case"] if frames else rational_J
        final_mean = frames[-1]["mean"] if frames else float(np.mean(baseline.t))
        mean_evac = sim_result["mean_evacuation_time"]
        quantiles = _quantiles(sim.evacuation_times.values(), num_agents)

        blocked_total = int(np.sum(sim.blocked_counts))

        return jsonify({
            "ok": True,
            "frames": frames,
            "vmin": round(float(vmin), 4),
            "vmax": round(float(vmax), 4),
            "panic_max": round(float(max_panic_seen), 4) if max_panic_seen > 0 else 1.0,
            "rational_J": round(float(rational_J), 4),
            "n_t_semantics": "frozen-chain diagnostic, not a forecast",
            "n_agents_start": num_agents,
            "scenario": {
                "panic_on": scenario.get("panic_enabled", True),
                "panic_decay": scenario.get("panic_decay", 0.0),
                "cell_capacity": scenario.get("cell_capacity"),
                "exit_capacity": scenario.get("exit_capacity"),
                "traits": scenario.get("use_traits", False),
                "trait_rho": payload.get("trait_rho") if scenario.get("use_traits") else None,
            },
            "final_stats": {
                "total_steps": sim_result["total_steps"],
                "evacuated_agents": sim_result["evacuated_agents"],
                "evacuation_rate": round(float(sim_result["evacuation_rate"]), 4),
                "mean_evacuation_time": (
                    round(float(mean_evac), 4) if mean_evac != float("inf") else None
                ),
                "T50": quantiles["T50"],
                "T90": quantiles["T90"],
                "T100": quantiles["T100"],
                "blocked_total": blocked_total,
                "blocked_top": _blocked_summary(sim.blocked_counts),
                "truncated_traits": int(getattr(sim, "truncated_traits", 0)),
                "final_worst_case": round(float(final_worst), 4),
                "final_mean": round(float(final_mean), 4),
                "seconds": round(seconds, 3),
            },
        })

    except Exception as exc:
        return _fail(exc)


@app.route("/api/ensemble", methods=["POST"])
def ensemble():
    """
    Common-seed ensemble of the coupled simulation.

    Request JSON:
        grid, n_runs (20), n_agents (40), seed (1000), max_steps (600),
        panic_on (true), panic_decay (0.15), cell_capacity (1),
        exit_capacity (1), traits (false), trait_rho (0.75)

    Response JSON:
        ok, stat: {T50|T90|T100: {median, lo, hi, n_censored}}, fraction_finished,
        fraction_evacuated_mean, blocked_total, truncated_traits_total, n_runs,
        seconds, scenario.
    """
    gate = _require_local()
    if gate is not None:
        return gate
    try:
        payload = request.get_json(force=True, silent=True) or {}
        grid = _parse_grid(payload)

        n_runs = _int(payload, "n_runs", 20, 1, 50)
        n_agents = _int(payload, "n_agents", 40, 1, 300)
        seed = _int(payload, "seed", 1000, 0, 2**32 - 1)
        max_steps = _int(payload, "max_steps", 600, 10, 1200)

        scenario = _sim_kwargs(payload)
        scenario["n_agents"] = n_agents
        # The editor defaults to a capacity model; both capacities are opt-in.
        scenario.setdefault("cell_capacity", None)
        scenario.setdefault("exit_capacity", None)

        t0 = time.perf_counter()
        result = run_ensemble(
            grid,
            n_runs=n_runs,
            base_seed=seed,
            max_steps=max_steps,
            **scenario,
        )
        summary = result.summary()
        seconds = time.perf_counter() - t0

        stats = {}
        for name in ("T50", "T90", "T100"):
            entry = summary[name]
            lo, hi = entry["bootstrap_ci"]
            stats[name] = {
                "median": None if math.isnan(entry["median"]) else entry["median"],
                "lo": None if math.isnan(lo) else lo,
                "hi": None if math.isnan(hi) else hi,
                "n_censored": entry["n_censored"],
            }

        return jsonify({
            "ok": True,
            "stats": stats,
            "fraction_finished": summary["fraction_finished"],
            "fraction_evacuated_mean": summary["fraction_evacuated_mean"],
            "blocked_total": summary.get("blocked_total", 0),
            "truncated_traits_total": summary.get("truncated_traits_total", 0),
            "n_runs": result.n_runs,
            "seconds": round(seconds, 3),
            "scenario": {
                "panic_on": scenario.get("panic_enabled", True),
                "panic_decay": scenario.get("panic_decay", 0.0),
                "cell_capacity": scenario.get("cell_capacity"),
                "exit_capacity": scenario.get("exit_capacity"),
                "traits": scenario.get("use_traits", False),
                "trait_rho": payload.get("trait_rho") if scenario.get("use_traits") else None,
                "n_agents": n_agents,
                "seed": seed,
                "max_steps": max_steps,
            },
        })

    except Exception as exc:
        return _fail(exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print()
    print("  TheSafeExit Web Editor")
    print(f"  ->  http://127.0.0.1:{port}")
    print("  Local only. Press Ctrl+C to stop.")
    print()
    app.run(host="127.0.0.1", port=port, debug=False)
