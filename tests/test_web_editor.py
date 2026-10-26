"""Endpoint tests for the TheSafeExit web editor.

Flask is an optional dependency (``requirements-web.txt``). When it is not
installed these tests skip instead of failing the headless suite.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

pytest.importorskip("flask")
pytest.importorskip("flask_cors")

import numpy as np

from src.floor_plan import get_simple_room

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
APP_PATH = os.path.join(PROJECT_ROOT, "demo", "web_editor", "app.py")


@pytest.fixture(scope="module")
def client():
    spec = importlib.util.spec_from_file_location("safexit_editor_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.app.config.update(TESTING=True)
    return module.app.test_client()


@pytest.fixture(scope="module")
def small_grid():
    return get_simple_room(10, 10, exit_width=2).tolist()


def test_health(client):
    body = client.get("/api/health").get_json()
    assert body["ok"] is True
    assert body["app"]


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower()


def test_analyze_reports_numerics(client, small_grid):
    body = client.post("/api/analyze", json={"grid": small_grid}).get_json()
    assert body["ok"] is True
    assert body["n_transient"] > 0
    assert body["worst_case_time"] > 0
    numerics = body["numerics"]
    assert numerics["method"] == "spsolve"
    assert numerics["residual"] < 1e-8
    assert numerics["n_states"] == body["n_transient"]


def test_heavy_endpoints_reject_non_local_callers(client, small_grid):
    """The two expensive endpoints are restricted to this machine."""
    resp = client.post(
        "/api/simulate",
        json={"grid": small_grid},
        environ_base={"REMOTE_ADDR": "10.0.0.5"},
    )
    assert resp.status_code == 403
    assert resp.get_json()["ok"] is False

    resp = client.post(
        "/api/ensemble",
        json={"grid": small_grid, "n_runs": 1, "n_agents": 5},
        environ_base={"REMOTE_ADDR": "10.0.0.5"},
    )
    assert resp.status_code == 403
    assert resp.get_json()["ok"] is False


def test_analyze_rejects_bad_grid(client):
    resp = client.post("/api/analyze", json={"grid": [[9, 9], [9, 9]]})
    body = resp.get_json()
    assert resp.status_code == 400
    assert body["ok"] is False
    assert "error" in body
    assert "trace" not in body


def test_analyze_requires_grid(client):
    resp = client.post("/api/analyze", json={})
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_monte_carlo_matches_exact(client, small_grid):
    body = client.post(
        "/api/monte_carlo",
        json={"grid": small_grid, "start_row": 5, "start_col": 2, "trials": 2000},
    ).get_json()
    assert body["ok"] is True
    assert body["trials"] == 2000
    assert body["analytic_mean"] > 0
    # The rational chain is deterministic, so sampling must reproduce it.
    assert body["mean"] == body["analytic_mean"]
    assert body["relative_error"] == 0


def test_monte_carlo_rejects_wall_start(client, small_grid):
    resp = client.post(
        "/api/monte_carlo",
        json={"grid": small_grid, "start_row": 0, "start_col": 0, "trials": 100},
    )
    assert resp.status_code == 400


def test_optimize_returns_exact_and_heuristic(client):
    body = client.post(
        "/api/optimize",
        json={"rows": 10, "cols": 10, "exit_width": 2, "obstacles": []},
    ).get_json()
    assert body["ok"] is True
    exact = body["exact"]
    heuristic = body["heuristic"]
    assert exact["n_candidates"] > 0
    assert exact["score"] == pytest.approx(body["exact"]["score"])
    # The exhaustive search is optimal on the enumerated set, so no heuristic
    # result may be strictly better than it by more than floating-point noise.
    assert heuristic["J"] >= exact["score"] - 1e-9
    assert body["gap_steps"] == pytest.approx(heuristic["J"] - exact["score"])
    assert body["exact_grid"] is not None


def test_simulate_legacy_payload_still_works(client, small_grid):
    body = client.post(
        "/api/simulate",
        json={"grid": small_grid, "num_agents": 12, "max_steps": 40},
    ).get_json()
    assert body["ok"] is True
    assert body["n_agents_start"] == 12
    assert body["frames"]
    scenario = body["scenario"]
    # Defaults must preserve the original behaviour (off switches off).
    assert scenario["panic_on"] is True
    assert scenario["panic_decay"] == 0
    assert scenario["cell_capacity"] is None
    assert scenario["exit_capacity"] is None
    assert scenario["traits"] is False
    for key in ("T50", "T90", "T100", "blocked_total"):
        assert key in body["final_stats"]


def test_simulate_scenario_options_reach_the_engine(client, small_grid):
    body = client.post(
        "/api/simulate",
        json={
            "grid": small_grid,
            "num_agents": 20,
            "max_steps": 60,
            "panic_decay": 0.15,
            "cell_capacity": 1,
            "exit_capacity": 1,
            "traits": True,
            "trait_rho": -0.5,
        },
    ).get_json()
    assert body["ok"] is True
    scenario = body["scenario"]
    assert scenario["panic_decay"] == pytest.approx(0.15)
    assert scenario["cell_capacity"] == 1
    assert scenario["exit_capacity"] == 1
    assert scenario["traits"] is True
    assert scenario["trait_rho"] == pytest.approx(-0.5)
    # Capacity is the only source of blocked moves.
    assert body["final_stats"]["blocked_total"] > 0


def test_simulate_rejects_oversized_grid(client):
    grid = np.ones((45, 45), dtype=int).tolist()
    resp = client.post("/api/simulate", json={"grid": grid})
    body = resp.get_json()
    assert resp.status_code == 400
    assert "40x40" in body["error"]


def test_ensemble_reports_quantiles(client, small_grid):
    body = client.post(
        "/api/ensemble",
        json={
            "grid": small_grid,
            "n_runs": 3,
            "n_agents": 15,
            "seed": 1000,
            "max_steps": 200,
            "panic_decay": 0.15,
            "cell_capacity": 1,
            "exit_capacity": 1,
        },
    ).get_json()
    assert body["ok"] is True
    assert body["n_runs"] == 3
    for key in ("T50", "T90", "T100"):
        stat = body["stats"][key]
        assert "median" in stat and "lo" in stat and "hi" in stat
    assert 0.0 <= body["fraction_finished"] <= 1.0
    assert body["blocked_total"] >= 0


def test_ensemble_off_switch_removes_blocking(client, small_grid):
    body = client.post(
        "/api/ensemble",
        json={"grid": small_grid, "n_runs": 2, "n_agents": 10, "max_steps": 200, "seed": 7},
    ).get_json()
    assert body["ok"] is True
    assert body["blocked_total"] == 0
    assert body["scenario"]["cell_capacity"] is None
