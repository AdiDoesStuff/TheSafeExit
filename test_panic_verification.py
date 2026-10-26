"""
test_panic_verification.py
==========================
Standalone physical/numerical verification of src/panic.py.

Run from the project root:
    python test_panic_verification.py

Prints PASS / FAIL for each check with the actual computed values.
No pytest required.
"""

from __future__ import annotations
import sys
import os
import textwrap

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.panic import PanicField, UnstableTimestepError, compute_agent_density, CELL_WALL, CELL_WALKABLE, CELL_EXIT

# ── Result tracking ───────────────────────────────────────────────────────────
results: list[tuple[str, bool]] = []

def report(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    bar = "─" * 70
    print(f"\n{bar}")
    print(f"  [{status}]  {name}")
    if detail:
        for line in textwrap.wrap(detail, width=68, initial_indent="  ", subsequent_indent="         "):
            print(line)
    print(bar)
    results.append((name, passed))


def make_room(rows: int = 15, cols: int = 15) -> np.ndarray:
    """Empty walled room: walkable interior, wall boundary."""
    g = np.ones((rows, cols), dtype=int)
    g[0, :] = CELL_WALL
    g[-1, :] = CELL_WALL
    g[:, 0] = CELL_WALL
    g[:, -1] = CELL_WALL
    return g


# ==============================================================================
# CHECK 1  —  MASS CONSERVATION (no source, closed boundaries)
# ==============================================================================
def check_mass_conservation():
    print("\n" + "=" * 70)
    print("  CHECK 1 — MASS CONSERVATION (no source, closed Neumann boundaries)")
    print("=" * 70)

    rows, cols = 15, 15
    fp = make_room(rows, cols)

    # Spike of panic exactly in the centre walkable cell
    pf = PanicField(fp, D=0.5, k=0.0)
    cr, cc = rows // 2, cols // 2
    pf.u[cr, cc] = 100.0

    mass_0 = pf.total_panic()
    print(f"  Initial mass (spike at ({cr},{cc})):  {mass_0:.10f}")

    # Run 200 steps with zero density (no source)
    for _ in range(200):
        pf.step(density=None)   # pure diffusion

    mass_final = pf.total_panic()
    rel_err = abs(mass_final - mass_0) / mass_0

    print(f"  Final mass after 200 steps:       {mass_final:.10f}")
    print(f"  Absolute change:                  {abs(mass_final - mass_0):.2e}")
    print(f"  Relative error:                   {rel_err:.2e}")

    # NOTE: The implementation calls np.clip(u_new, 0.0, None) AFTER the FTCS
    # update. For a stable dt and a purely-positive initial field with no
    # source, u should never want to go negative, so clip has no effect and
    # conservation should hold to floating-point precision (~1e-14).
    TOL = 1e-8
    passed = rel_err < TOL
    report(
        "Mass conservation",
        passed,
        f"rel_err = {rel_err:.2e}  (tolerance = {TOL:.0e}). "
        + ("Conservation holds." if passed
           else "LEAKAGE DETECTED — check _laplacian flux accumulation."),
    )


# ==============================================================================
# CHECK 2  —  ISOTROPIC SPREADING (symmetry about the centre)
# ==============================================================================
def check_isotropy():
    print("\n" + "=" * 70)
    print("  CHECK 2 — ISOTROPIC SPREADING (90-degree rotational symmetry)")
    print("=" * 70)

    # Use an odd-sized square room so the centre is a single cell
    N = 21
    fp = make_room(N, N)

    pf = PanicField(fp, D=0.5, k=0.0)
    cr, cc = N // 2, N // 2          # exact centre
    pf.u[cr, cc] = 1000.0

    # Run enough steps for panic to reach the walls but before boundary
    # reflections create strong asymmetries (50 steps is moderate)
    for _ in range(50):
        pf.step(density=None)

    u = pf.u.copy()

    # Four 90-degree rotations should all give the same field.
    # The centre of the grid aligns exactly for odd N.
    max_asym = 0.0
    for k_rot in [1, 2, 3]:
        u_rot = np.rot90(u, k=k_rot)
        asym = np.max(np.abs(u - u_rot))
        max_asym = max(max_asym, asym)
        print(f"  Max |u - rot90^{k_rot}(u)|:  {asym:.2e}")

    # Also check reflection symmetry (left-right, up-down)
    asym_lr = np.max(np.abs(u - np.fliplr(u)))
    asym_ud = np.max(np.abs(u - np.flipud(u)))
    max_asym = max(max_asym, asym_lr, asym_ud)
    print(f"  Max |u - fliplr(u)|:      {asym_lr:.2e}")
    print(f"  Max |u - flipud(u)|:      {asym_ud:.2e}")
    print(f"  Overall max asymmetry:    {max_asym:.2e}")
    print(f"  Peak panic value:         {u.max():.6f}")

    # On a square uniform-spacing grid with a centred spike, the conductance
    # discretisation should produce the same left/right/up/down fluxes ⟹
    # the field should be 4-fold symmetric to floating-point precision.
    TOL = 1e-10
    passed = max_asym < TOL
    report(
        "Isotropic spreading",
        passed,
        f"max asymmetry = {max_asym:.2e}  (tolerance = {TOL:.0e}). "
        + ("Perfectly isotropic." if passed
           else "ASYMMETRY DETECTED — check h_cond / v_cond flux directions in _laplacian."),
    )


# ==============================================================================
# CHECK 3  —  CFL GUARD ACTUALLY GUARDS
# ==============================================================================
def check_cfl_guard():
    print("\n" + "=" * 70)
    print("  CHECK 3 — CFL GUARD")
    print("=" * 70)

    fp = make_room(10, 10)
    D, dx, dy = 0.5, 1.0, 1.0

    # Theoretical limit (no safety factor): dt_lim = 0.5 / (D*(1/dx^2+1/dy^2))
    dt_lim_raw = 0.5 / (D * (1.0 / dx**2 + 1.0 / dy**2))
    print(f"  D={D}, dx={dx}, dy={dy}")
    print(f"  Theoretical CFL limit (safety_factor=1.0): dt_lim = {dt_lim_raw:.6f}")

    # ── 3a. dt that exceeds the safety-factor-wrapped limit should raise ──
    # PanicField.max_stable_dt uses safety_factor=0.9 by default.
    # We construct with safety_factor=1.0 so safe_dt = dt_lim_raw,
    # then request dt slightly above it.
    dt_bad = dt_lim_raw * 1.001
    print(f"  Trying dt={dt_bad:.6f} (1.001× limit)…")
    raised_correctly = False
    err_msg = ""
    try:
        # Pass safety_factor=1.0 so the guard threshold equals the true CFL limit
        PanicField(fp, D=D, dx=dx, dy=dy, k=0.0, dt=dt_bad, safety_factor=1.0)
    except UnstableTimestepError as e:
        raised_correctly = True
        err_msg = str(e)[:80]
    print(f"  UnstableTimestepError raised: {raised_correctly}")
    if raised_correctly:
        print(f"  Error: {err_msg}…")

    # ── 3b. dt just below the limit should NOT raise and should stay bounded ──
    dt_good = dt_lim_raw * 0.90   # 90 % of the hard limit
    print(f"\n  Trying dt={dt_good:.6f} (0.90× limit)…")
    no_raise = False
    stays_bounded = False
    try:
        pf_good = PanicField(fp, D=D, dx=dx, dy=dy, k=0.0, dt=dt_good, safety_factor=1.0)
        pf_good.u[5, 5] = 1.0
        max_vals = []
        for _ in range(200):
            pf_good.step(density=None)
            max_vals.append(pf_good.u.max())
        no_raise = True
        # Should stay finite and bounded (no blow-up, no oscillation to values >> initial)
        stays_bounded = np.isfinite(max_vals[-1]) and max_vals[-1] < 1.1  # within 10 % of initial
        print(f"  No exception raised:          {no_raise}")
        print(f"  Final max(u)={max_vals[-1]:.6f}  initial=1.0  (should stay ≤ 1.0 for pure diffusion)")
    except UnstableTimestepError as e:
        print(f"  UNEXPECTED raise: {e}")

    passed = raised_correctly and no_raise and stays_bounded
    report(
        "CFL guard",
        passed,
        f"Bad dt raised={raised_correctly}, good dt no-raise={no_raise}, "
        f"stays_bounded={stays_bounded}.",
    )


# ==============================================================================
# CHECK 4  —  SOURCE TERM PROPORTIONALITY  (S = k * rho * dt)
# ==============================================================================
def check_source_term():
    print("\n" + "=" * 70)
    print("  CHECK 4 — SOURCE TERM PROPORTIONALITY  (delta_u = k * rho * dt)")
    print("=" * 70)

    fp = make_room(10, 10)

    # Use D=0 to isolate the source from diffusion.
    # PanicField requires D > 0 for the CFL calc, so use tiny D and large dx
    # so that diffusion contributes negligibly in a single step.
    D_tiny = 1e-10   # effectively zero diffusion
    k = 2.0
    pf = PanicField(fp, D=D_tiny, k=k)

    density = np.zeros((10, 10))
    density[3, 3] = 5.0    # 5 agents at (3,3)
    density[7, 7] = 3.0    # 3 agents at (7,7)

    u_before = pf.u.copy()
    pf.step(density=density)
    u_after = pf.u.copy()

    delta_u = u_after - u_before
    expected_delta = k * density * pf.dt   # S * dt, ignoring D*lap which is ~0

    max_residual = np.max(np.abs(delta_u - expected_delta))
    print(f"  k={k},  dt={pf.dt:.6f}")
    print(f"  density[3,3]={density[3,3]},  delta_u[3,3]={delta_u[3,3]:.8f}")
    print(f"  expected    ={k*density[3,3]*pf.dt:.8f}   (k*rho*dt)")
    print(f"  density[7,7]={density[7,7]},  delta_u[7,7]={delta_u[7,7]:.8f}")
    print(f"  expected    ={k*density[7,7]*pf.dt:.8f}   (k*rho*dt)")
    print(f"  Max residual across all cells: {max_residual:.2e}")

    # ── Linearity: doubling k ──
    pf2 = PanicField(fp, D=D_tiny, k=2*k)
    pf2.step(density=density)
    delta_2k = pf2.u.copy()

    pf3 = PanicField(fp, D=D_tiny, k=k)
    pf3.step(density=density)
    delta_1k = pf3.u.copy()

    doubling_k_ratio = np.max(np.abs(delta_2k)) / np.max(np.abs(delta_1k + 1e-30))
    print(f"\n  Doubling k ratio (should be ~2.0): {doubling_k_ratio:.6f}")

    # ── Linearity: doubling density ──
    pf4 = PanicField(fp, D=D_tiny, k=k)
    pf4.step(density=2*density)
    delta_2rho = pf4.u.copy()

    doubling_rho_ratio = np.max(np.abs(delta_2rho)) / np.max(np.abs(delta_1k + 1e-30))
    print(f"  Doubling rho ratio (should be ~2.0): {doubling_rho_ratio:.6f}")

    TOL = 1e-8
    passed = (max_residual < TOL
              and abs(doubling_k_ratio - 2.0) < 1e-6
              and abs(doubling_rho_ratio - 2.0) < 1e-6)

    report(
        "Source term proportionality",
        passed,
        f"max_residual={max_residual:.2e}, "
        f"double-k ratio={doubling_k_ratio:.6f}, "
        f"double-rho ratio={doubling_rho_ratio:.6f}.",
    )


# ==============================================================================
# CHECK 5  —  WALL-AWARE NO-FLUX (internal walls isolate rooms)
# ==============================================================================
def check_wall_isolation():
    print("\n" + "=" * 70)
    print("  CHECK 5 — WALL-AWARE NO-FLUX (internal walls)")
    print("=" * 70)

    # ── 5a. Fully sealed barrier: room B must stay at zero ──
    # 15×15 grid. Vertical barrier at col=7 (interior: rows 1..13), sealing
    # left half (cols 1-6) from right half (cols 8-13) completely.
    R, C = 15, 15
    fp_sealed = np.ones((R, C), dtype=int)
    fp_sealed[0, :] = CELL_WALL
    fp_sealed[-1, :] = CELL_WALL
    fp_sealed[:, 0] = CELL_WALL
    fp_sealed[:, -1] = CELL_WALL
    fp_sealed[:, 7] = CELL_WALL          # full-height internal barrier

    pf_sealed = PanicField(fp_sealed, D=0.5, k=1.0)
    # Inject panic only in the left half (room A)
    pf_sealed.u[3:7, 1:7] = 10.0
    mass_room_a_0 = pf_sealed.u[:, 1:7].sum()
    mass_room_b_0 = pf_sealed.u[:, 8:].sum()
    print(f"  [Sealed] t=0: room A mass={mass_room_a_0:.4f},  room B mass={mass_room_b_0:.4f}")

    density_sealed = np.zeros((R, C))
    density_sealed[3:7, 1:7] = 2.0   # agents only on left

    for _ in range(100):
        pf_sealed.step(density=density_sealed)

    mass_room_b_100 = pf_sealed.u[:, 8:].sum()
    wall_held = pf_sealed.u[:, 7].sum() == 0.0   # wall cells always zero
    print(f"  [Sealed] t=100: room B mass={mass_room_b_100:.8f}  (must be 0)")
    print(f"  Wall cells stay at zero: {wall_held}")

    sealed_ok = (mass_room_b_100 == 0.0) and wall_held

    # ── 5b. Gap: panic CAN reach room B through the opening ──
    fp_gap = np.ones((R, C), dtype=int)
    fp_gap[0, :] = CELL_WALL
    fp_gap[-1, :] = CELL_WALL
    fp_gap[:, 0] = CELL_WALL
    fp_gap[:, -1] = CELL_WALL
    fp_gap[:, 7] = CELL_WALL
    fp_gap[7, 7] = CELL_WALKABLE    # single-cell gap at row 7

    pf_gap = PanicField(fp_gap, D=0.5, k=0.0)
    pf_gap.u[3:6, 1:7] = 10.0   # panic only in room A at t=0

    for _ in range(300):
        pf_gap.step(density=None)   # no source — pure diffusion through gap

    mass_room_b_gap = pf_gap.u[:, 8:].sum()
    print(f"\n  [Gap] t=300: room B mass={mass_room_b_gap:.6f}  (must be > 0)")

    gap_ok = mass_room_b_gap > 0.0

    passed = sealed_ok and gap_ok
    report(
        "Wall-aware no-flux",
        passed,
        f"Sealed: room B mass={mass_room_b_100:.2e} (should be 0.0). "
        f"Gap: room B mass={mass_room_b_gap:.4f} (should be >0). "
        + ("Both correct." if passed
           else "FAILURE — check h_cond/v_cond construction for internal walls."),
    )


# ==============================================================================
# CHECK 6  —  MONOTONICITY (no decay term, panic is permanent)
# ==============================================================================
def check_monotonicity():
    print("\n" + "=" * 70)
    print("  CHECK 6 — MONOTONICITY (panic must never decrease with active source)")
    print("=" * 70)

    R, C = 12, 12
    fp = make_room(R, C)

    pf = PanicField(fp, D=0.3, k=1.0)

    # Constant density everywhere inside
    density = np.zeros((R, C))
    density[1:R-1, 1:C-1] = 1.0

    prev = pf.u.copy()
    decreases_found = 0
    min_delta = 0.0

    for step in range(200):
        pf.step(density=density)
        # Every cell's panic should be >= previous value
        # (source is positive, clip prevents negatives, no decay term exists)
        delta = pf.u - prev
        bad = delta[fp != CELL_WALL]   # only check non-wall cells
        if bad.min() < -1e-12:
            decreases_found += 1
            min_delta = min(min_delta, bad.min())
        prev = pf.u.copy()

    total_final = pf.total_panic()
    print(f"  Steps with any cell decreasing: {decreases_found} / 200")
    print(f"  Minimum delta (most negative):  {min_delta:.2e}")
    print(f"  Final total panic:              {total_final:.4f}  (should be >> 0 with constant source)")

    # With constant positive source and sealed boundaries, every step must
    # add panic — no cell should ever decrease. The clip(0) ensures this
    # even if diffusion would momentarily pull a cell below zero (unlikely
    # for a stable dt, but clip is the safety net).
    passed = decreases_found == 0 and total_final > 0.0
    report(
        "Monotonicity (no decay)",
        passed,
        f"Steps with decreases: {decreases_found}/200, "
        f"min_delta={min_delta:.2e}, "
        f"final_mass={total_final:.4f}.",
    )


# ==============================================================================
# CHECK 7  —  compute_agent_density SANITY CHECK
# ==============================================================================
def check_agent_density():
    print("\n" + "=" * 70)
    print("  CHECK 7 — compute_agent_density SANITY CHECK")
    print("=" * 70)

    shape = (10, 10)
    # Manual setup:
    #   2 agents at (2, 3)
    #   1 agent  at (5, 7)
    #   3 agents at (0, 0)   (corner — also tests boundary handling)
    agent_coords = [
        (2, 3), (2, 3),       # two at same cell
        (5, 7),
        (0, 0), (0, 0), (0, 0),
    ]
    expected = np.zeros(shape)
    expected[2, 3] = 2
    expected[5, 7] = 1
    expected[0, 0] = 3

    density = compute_agent_density(agent_coords, shape)

    # Check exact integer counts
    max_err = np.max(np.abs(density - expected))
    total_agents = int(density.sum())

    print(f"  density[2,3] = {density[2,3]:.1f}  (expected 2)")
    print(f"  density[5,7] = {density[5,7]:.1f}  (expected 1)")
    print(f"  density[0,0] = {density[0,0]:.1f}  (expected 3)")
    print(f"  Total agents counted: {total_agents}  (expected 6)")
    print(f"  Max abs error across all cells: {max_err:.2e}")

    # Test empty agent list
    density_empty = compute_agent_density([], shape)
    all_zero = np.all(density_empty == 0.0)
    print(f"  Empty agent list → all-zero density: {all_zero}")

    # Test array input (not just list-of-tuples)
    arr_coords = np.array([(2, 3), (5, 7)], dtype=int)
    density_arr = compute_agent_density(arr_coords, shape)
    array_ok = density_arr[2, 3] == 1 and density_arr[5, 7] == 1

    passed = (max_err == 0.0 and total_agents == 6 and all_zero and array_ok)
    report(
        "compute_agent_density",
        passed,
        f"max_err={max_err:.2e}, total={total_agents}/6, "
        f"empty_ok={all_zero}, array_input_ok={array_ok}.",
    )


# ==============================================================================
# BONUS CHECK  —  CLIP INTERACTION WITH CONSERVATION
# ==============================================================================
# The np.clip(u_new, 0.0, None) in step() technically breaks strict mass
# conservation IF u ever wants to go negative (which can happen in the FTCS
# scheme near steep gradients at the stability boundary). This check verifies
# that for a STABLE, well-posed run the clip never fires, so conservation
# in checks 1-2 is genuine and not just masked by the clip.
def check_clip_never_fires():
    print("\n" + "=" * 70)
    print("  BONUS — CLIP NEVER FIRES for stable dt / positive initial field")
    print("=" * 70)

    fp = make_room(15, 15)
    pf = PanicField(fp, D=0.5, k=0.0)   # auto-selects safe dt
    pf.u[7, 7] = 100.0

    clip_fires_count = 0
    for _ in range(200):
        u_before_clip = pf.u + pf.dt * pf.D * pf._laplacian(pf.u)
        went_negative = np.any(u_before_clip < 0)
        if went_negative:
            clip_fires_count += 1
        pf.step(density=None)

    print(f"  Steps where pre-clip u < 0: {clip_fires_count} / 200")
    print(f"  (If > 0, clip is masking conservation; consider reducing dt)")

    passed = clip_fires_count == 0
    report(
        "Clip never fires (stable run)",
        passed,
        f"{clip_fires_count}/200 steps had pre-clip negatives. "
        + ("No masking — conservation in checks 1/2 is genuine." if passed
           else "CLIP IS ACTIVE — conservation holds only because clip removes negatives, "
                "not because the scheme is truly conservative. Use a smaller dt."),
    )


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("  TheSafeExit — PanicField Physical/Numerical Verification")
    print("#" * 70)

    check_mass_conservation()
    check_isotropy()
    check_cfl_guard()
    check_source_term()
    check_wall_isolation()
    check_monotonicity()
    check_agent_density()
    check_clip_never_fires()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "#" * 70)
    print("  SUMMARY")
    print("#" * 70)
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    for name, ok in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker}  {name}")
    print(f"\n  {n_pass} / {len(results)} checks passed")

    if n_fail > 0:
        print(f"\n  *** {n_fail} check(s) FAILED — see output above for details ***")
        print("""
  Diagnostic guide:
    Check 1 fails  → _laplacian flux is leaking/gaining at boundary rows/cols.
                     Look at lap_h[:, :-1] += / lap_h[:, 1:] -= accumulation.
    Check 2 fails  → Asymmetry in h_cond vs v_cond, or a transposed index.
    Check 3 fails  → max_stable_dt formula or the > vs >= guard in __init__.
    Check 4 fails  → Source wiring: 'source = self.k * density' (panic.py:217).
    Check 5 fails  → h_cond/v_cond construction; check passable_mask logical AND.
    Check 6 fails  → A decay (gamma) term was added; check panic.py:219.
    Check 7 fails  → flat_indices = coords[:,0]*cols + coords[:,1] in bincount.
    Bonus   fails  → dt is too close to the CFL limit; clip is masking negatives.
""")
        sys.exit(1)
    else:
        print("\n  All checks passed. PanicField is physically and numerically correct.")
        sys.exit(0)
