"""
test_panic_web_integration.py - Regression tests for panic-coupled simulation integration.

Tests:
1. test_zero_panic_control:
   - When k=0 or beta=0, panic coupling must produce lambda_t = 0 everywhere.
   - At every recorded snapshot, J(t) must match the static rational J* exactly (< 1e-6).
2. test_saturation_consistency:
   - Under heavy density and extreme coupling (high k, high beta), lambda_t saturates to 1.0
     across all transient cells (min lambda >= 0.999).
   - The worst-case J from the saturated snapshot matches the static pure-uniform J*
     within a tight tolerance (< 1.0%).
"""

import pytest
import numpy as np

from src.floor_plan import get_simple_room, CELL_WALKABLE
from src.markov import build_state_map, build_rational_QR, build_uniform_QR
from src.solvers import fundamental_matrix_lu
from src.heatmap import fundamental_to_heatmap
from src.simulation import EvacuationSimulation


def _compute_static_worst_case(floor_plan: np.ndarray, mode: str = "rational") -> float:
    state_map, _, n_transient, n_absorbing = build_state_map(floor_plan)
    if mode == "rational":
        from src.floor_plan import compute_exit_distance_field
        dist = compute_exit_distance_field(floor_plan)
        Q, _ = build_rational_QR(floor_plan, dist, state_map, n_transient, n_absorbing)
    elif mode == "uniform":
        Q, _ = build_uniform_QR(floor_plan, state_map, n_transient, n_absorbing)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    N = fundamental_matrix_lu(Q)
    h_2d = fundamental_to_heatmap(N, floor_plan, state_map, n_transient)
    valid_h = h_2d[~np.isnan(h_2d) & (floor_plan == CELL_WALKABLE)]
    return float(np.max(valid_h))


def test_zero_panic_control_k_zero():
    """
    Variant A: k=0 (zero source gain).
    Asserts J(t) at every snapshot equals static rational J* to within 1e-6.
    """
    grid = get_simple_room(10, 10, exit_width=2)
    rational_J = _compute_static_worst_case(grid, mode="rational")

    sim = EvacuationSimulation(
        floor_plan=grid,
        n_agents=30,
        D=0.2,
        k=0.0,      # zero panic source
        beta=0.4,
        seed=42,
        panic_enabled=True,
    )

    results = sim.run(max_steps=40, snapshot_every=2, max_snapshots=20)
    assert len(results["snapshots"]) > 0, "No snapshots recorded"

    for snap in results["snapshots"]:
        t = snap["t"]
        h_2d = snap["heatmap_2d"]
        valid_h = h_2d[~np.isnan(h_2d) & (grid == CELL_WALKABLE)]
        snap_J = float(np.max(valid_h)) if len(valid_h) > 0 else 0.0

        diff = abs(snap_J - rational_J)
        print(f"[k=0] Snapshot t={t:02d}: snap_J={snap_J:.6f}, rational_J={rational_J:.6f}, diff={diff:.2e}")
        assert diff < 1e-6, (
            f"Zero-panic (k=0) control failed at t={t}: snap_J={snap_J} != rational_J={rational_J} (diff={diff})"
        )


def test_zero_panic_control_beta_zero():
    """
    Variant B: beta=0 (zero panic sensitivity).
    Even with nonzero panic field u(x,t) > 0, lambda = clip(beta * u, 0, 1) = 0 everywhere.
    Asserts J(t) at every snapshot equals static rational J* to within 1e-6.
    """
    grid = get_simple_room(10, 10, exit_width=2)
    rational_J = _compute_static_worst_case(grid, mode="rational")

    sim = EvacuationSimulation(
        floor_plan=grid,
        n_agents=30,
        D=0.2,
        k=2.0,      # high panic source
        beta=0.0,   # zero sensitivity -> lambda=0
        seed=123,
        panic_enabled=True,
    )

    results = sim.run(max_steps=40, snapshot_every=2, max_snapshots=20)
    assert len(results["snapshots"]) > 0, "No snapshots recorded"

    for snap in results["snapshots"]:
        t = snap["t"]
        h_2d = snap["heatmap_2d"]
        valid_h = h_2d[~np.isnan(h_2d) & (grid == CELL_WALKABLE)]
        snap_J = float(np.max(valid_h)) if len(valid_h) > 0 else 0.0

        diff = abs(snap_J - rational_J)
        print(f"[beta=0] Snapshot t={t:02d}: snap_J={snap_J:.6f}, rational_J={rational_J:.6f}, diff={diff:.2e}")
        assert diff < 1e-6, (
            f"Zero-panic (beta=0) control failed at t={t}: snap_J={snap_J} != rational_J={rational_J} (diff={diff})"
        )


def test_saturation_consistency():
    """
    SATURATION-CONSISTENCY TEST:
    Under extreme panic (high density, high k, high beta, no decay), lambda saturates
    to 1.0 across transient cells (min lambda >= 0.999).
    The worst-case J from the saturated snapshot must match the static pure-uniform J*
    to within tight tolerance (< 1.0%).
    """
    # Small compact room with single exit
    grid = get_simple_room(6, 6, exit_width=1)
    uniform_J = _compute_static_worst_case(grid, mode="uniform")

    # Place high number of agents with extreme k and beta to guarantee full saturation across room
    sim = EvacuationSimulation(
        floor_plan=grid,
        n_agents=50,
        D=0.5,
        k=20.0,
        beta=100.0,
        seed=999,
        panic_enabled=True,
    )

    # Step simulation forward and check saturation
    final_snap = None
    min_lambda_final = 0.0

    for step in range(1, 25):
        step_info = sim.step(record_snapshot=True)
        # Compute current lambda vector
        panic_vec = sim.panic_field.u[sim._transient_rows, sim._transient_cols]
        lam_vec = np.clip(sim.beta * panic_vec, 0.0, 1.0)
        min_lam = float(np.min(lam_vec))
        mean_lam = float(np.mean(lam_vec))

        if step_info["snapshot"] is not None:
            final_snap = step_info["snapshot"]
            min_lambda_final = min_lam

        if min_lam >= 0.999:
            # Saturation achieved
            break

    print(f"\n[Saturation Test] Reached min_lambda = {min_lambda_final:.6f}, target >= 0.999")
    assert min_lambda_final >= 0.999, (
        f"Failed to achieve full saturation across all transient cells: min lambda = {min_lambda_final:.6f}"
    )

    # Check J(t) at saturation vs uniform_J
    h_2d = final_snap["heatmap_2d"]
    valid_h = h_2d[~np.isnan(h_2d) & (grid == CELL_WALKABLE)]
    sat_J = float(np.max(valid_h))

    rel_diff = abs(sat_J - uniform_J) / uniform_J
    abs_diff = abs(sat_J - uniform_J)

    print(f"[Saturation Test] sat_J = {sat_J:.6f}, uniform_J = {uniform_J:.6f}")
    print(f"[Saturation Test] abs_diff = {abs_diff:.6e}, rel_diff = {rel_diff * 100:.4f}%")

    assert rel_diff < 0.01, (
        f"Saturated J={sat_J:.4f} diverges from pure uniform J*={uniform_J:.4f} by {rel_diff*100:.2f}% (> 1%)"
    )
