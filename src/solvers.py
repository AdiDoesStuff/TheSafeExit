"""
solvers.py — Linear algebra for absorbing Markov chains.

The primary quantity needed by the evacuation analysis is the expected time to
absorption from every transient state:

    t = (I - Q)^(-1) 1

This module provides four ways to compute it:

* ``solve_expected_time_sparse`` — one sparse direct solve (2.1),
* ``solve_expected_time_lu``     — sparse LU factorisation (kept for reuse),
* ``solve_expected_time_dense``  — dense direct solve, for small grids,
* ``solve_expected_time_iterative`` — CG / BiCGSTAB / GMRES (2.2).

``solve_fundamental_lu`` and ``solve_fundamental_svd`` remain available for the
cases that genuinely need the full matrix N (the small-grid maps of Phase 1).
The single-solve helpers deliberately never form a dense n x n matrix.

All routines accept explicit random generators where randomness is involved.
No function reads or writes the global NumPy random state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.floor_plan import get_simple_room
from src.markov import build_QR, build_state_map


# --------------------------------------------------------------------------- #
# Errors and result containers
# --------------------------------------------------------------------------- #
class NonAbsorbingChainError(RuntimeError):
    """Raised when a chain can never reach an absorbing state (or is too slow)."""


@dataclass
class SolveReport:
    """Outcome of one linear solve for the expected-time vector."""

    t: np.ndarray
    method: str
    residual: float
    seconds: float
    iterations: int = 0
    n: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "residual": float(self.residual),
            "seconds": float(self.seconds),
            "iterations": int(self.iterations),
            "n": int(self.n),
        }


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _as_square_sparse(Q: Any) -> sp.csr_matrix:
    Q_csr = sp.csr_matrix(Q, dtype=np.float64)
    if Q_csr.ndim != 2 or Q_csr.shape[0] != Q_csr.shape[1]:
        raise ValueError(f"Q must be square, got shape {Q_csr.shape}")
    if Q_csr.shape[0] == 0:
        raise ValueError("Q must contain at least one transient state")
    return Q_csr


def _system_matrix(Q: Any) -> sp.csc_matrix:
    """Return A = I - Q in CSC format."""
    Q_csr = _as_square_sparse(Q)
    n = Q_csr.shape[0]
    A = (sp.eye(n, format="csc", dtype=np.float64) - Q_csr.tocsc()).tocsc()
    return A


def expected_time_residual(Q: Any, t: np.ndarray) -> float:
    """
    Relative residual of ``(I - Q) t = 1``:

        ||(I - Q) t - 1||_2 / ||1||_2

    The denominator is sqrt(n), so the value is scale-free.
    """
    A = _system_matrix(Q)
    b = np.ones(A.shape[0], dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    if t.shape != b.shape:
        raise ValueError(f"t has shape {t.shape}, expected {b.shape}")
    return float(np.linalg.norm(A @ t - b) / np.linalg.norm(b))


# --------------------------------------------------------------------------- #
# Full fundamental matrix (small grids / maps only)
# --------------------------------------------------------------------------- #
def solve_fundamental_lu(Q: sp.csr_matrix) -> np.ndarray:
    """
    Solve ``(I - Q) N = I`` with a sparse LU factorisation.

    The result is dense by definition (N is n x n). Use
    :func:`solve_expected_time_sparse` when only N @ 1 is needed.
    """
    n = Q.shape[0]
    A_csc = _system_matrix(Q)

    lu_factor = spla.splu(A_csc)
    identity = np.eye(n, dtype=np.float64)
    N = lu_factor.solve(identity)
    return N


def solve_fundamental_svd(Q: sp.csr_matrix) -> np.ndarray:
    """
    Solve ``(I - Q) N = I`` with a dense SVD.

    ``A = U S V^T`` so ``A^-1 = V S^-1 U^T``. This is O(n^3) and intended for
    small or ill-conditioned systems; see :func:`matrix_condition_number` and
    :func:`svd_rank_truncation_errors` for when the SVD is informative.
    """
    n = Q.shape[0]
    A_dense = _system_matrix(Q).toarray()

    U, s, Vt = sla.svd(A_dense)

    # A singular value below this threshold is treated as zero. For a valid
    # absorbing chain A is non-singular, so this only affects numerically
    # singular inputs.
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T


# Aliases for convenience and standardized naming
fundamental_matrix_lu = solve_fundamental_lu
fundamental_matrix_svd = solve_fundamental_svd


# --------------------------------------------------------------------------- #
# 2.1 Sparse single solve: t = (I - Q)^(-1) 1
# --------------------------------------------------------------------------- #
def solve_expected_time_lu(Q: Any) -> SolveReport:
    """Expected times via one sparse LU factorisation and one RHS."""
    A = _system_matrix(Q)
    n = A.shape[0]
    b = np.ones(n, dtype=np.float64)

    t0 = time.perf_counter()
    lu = spla.splu(A)
    t = lu.solve(b)
    seconds = time.perf_counter() - t0

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    return SolveReport(
        t=t,
        method="sparse_lu",
        residual=expected_time_residual(Q, t),
        seconds=seconds,
        iterations=1,
        n=n,
    )


def solve_expected_time_sparse(Q: Any) -> SolveReport:
    """
    Expected times via a single sparse direct solve (``spsolve``).

    This is the Phase 2 replacement for forming ``N``: one solve of
    ``(I - Q) t = 1`` rather than an n x n inverse.
    """
    A = _system_matrix(Q)
    n = A.shape[0]
    b = np.ones(n, dtype=np.float64)

    t0 = time.perf_counter()
    t = spla.spsolve(A, b)
    seconds = time.perf_counter() - t0

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    return SolveReport(
        t=t,
        method="spsolve",
        residual=expected_time_residual(Q, t),
        seconds=seconds,
        iterations=1,
        n=n,
    )


def solve_expected_time_dense(Q: Any) -> SolveReport:
    """Expected times via a dense direct solve, for small comparison grids."""
    A = _system_matrix(Q)
    n = A.shape[0]
    b = np.ones(n, dtype=np.float64)

    t0 = time.perf_counter()
    t = sla.solve(A.toarray(), b, assume_a="gen")
    seconds = time.perf_counter() - t0

    t = np.asarray(t, dtype=np.float64).reshape(-1)
    return SolveReport(
        t=t,
        method="dense_solve",
        residual=expected_time_residual(Q, t),
        seconds=seconds,
        iterations=1,
        n=n,
    )


def expected_times(Q: Any) -> np.ndarray:
    """Convenience wrapper returning only the expected-time vector."""
    return solve_expected_time_sparse(Q).t


# --------------------------------------------------------------------------- #
# 2.2 Iterative solvers
# --------------------------------------------------------------------------- #
def _ilu_preconditioner(
    A: sp.spmatrix,
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0,
) -> Optional[spla.LinearOperator]:
    """Build an incomplete-LU preconditioner, or return None if it fails."""
    try:
        ilu = spla.spilu(
            A.tocsc(),
            drop_tol=drop_tol,
            fill_factor=fill_factor,
            permc_spec="COLAMD",
        )
        return spla.LinearOperator(A.shape, matvec=ilu.solve)
    except Exception:
        return None


def solve_expected_time_iterative(
    Q: Any,
    *,
    method: str = "cg",
    rtol: float = 1e-10,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
    preconditioner: str = "auto",
    drop_tol: float = 1e-4,
    fill_factor: float = 10.0,
    allow_fallback: bool = True,
) -> SolveReport:
    """
    Expected times with an iterative Krylov solver.

    Parameters
    ----------
    method:
        ``"cg"`` (requires symmetric positive definite ``I - Q``; use it for
        the uniform random walk), ``"bicgstab"`` or ``"gmres"`` (use these for
        rational or panic-mixed chains, whose Q is not symmetric).
    preconditioner:
        ``"auto"`` uses ILU for BiCGSTAB/GMRES and no preconditioner for CG,
        ``"ilu"`` forces incomplete LU, ``"none"``/``None`` disables it.
    allow_fallback:
        If BiCGSTAB/GMRES fails to reach a small residual, retry with GMRES
        (preconditioned). The returned method name records the fallback.

    Returns
    -------
    SolveReport with ``iterations`` (callback count) and ``residual``.
    """
    method_norm = str(method).strip().lower()
    if method_norm not in {"cg", "bicgstab", "gmres"}:
        raise ValueError("method must be one of 'cg', 'bicgstab', 'gmres'")

    A = _system_matrix(Q)
    n = A.shape[0]
    b = np.ones(n, dtype=np.float64)
    if maxiter is None:
        maxiter = max(1000, 10 * n)

    precond_mode = "auto" if preconditioner is None else str(preconditioner).lower()
    M = None
    if precond_mode in {"ilu", "auto"} and (precond_mode == "ilu" or method_norm != "cg"):
        M = _ilu_preconditioner(A, drop_tol=drop_tol, fill_factor=fill_factor)

    def _run(solver_name: str) -> Tuple[np.ndarray, int, int]:
        iterations = {"count": 0}

        def callback(_xk: np.ndarray) -> None:
            iterations["count"] += 1

        common = dict(
            A=A,
            b=b,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            M=M,
            callback=callback,
        )
        if solver_name == "cg":
            x, info = spla.cg(**common)
        elif solver_name == "bicgstab":
            x, info = spla.bicgstab(**common)
        else:
            # Restart matters for GMRES; a modest restart keeps memory bounded.
            x, info = spla.gmres(restart=min(200, max(20, n)), **common)
        return np.asarray(x, dtype=np.float64).reshape(-1), int(info), iterations["count"]

    t0 = time.perf_counter()
    t, info, iterations = _run(method_norm)
    seconds = time.perf_counter() - t0
    residual = expected_time_residual(Q, t)

    used_method = method_norm
    if (info != 0 or residual > 1e-8) and allow_fallback and method_norm != "gmres":
        # A failed BiCGSTAB is common on strongly non-symmetric rational chains;
        # GMRES with the same ILU is the documented fallback in the plan.
        if M is None:
            M = _ilu_preconditioner(A, drop_tol=drop_tol, fill_factor=fill_factor)
        t0 = time.perf_counter()
        t, info, iterations = _run("gmres")
        seconds = time.perf_counter() - t0
        residual = expected_time_residual(Q, t)
        used_method = f"{method_norm}->gmres"

    if not np.all(np.isfinite(t)):
        raise NonAbsorbingChainError(
            "Iterative solver produced non-finite expected times; the chain may "
            "have a non-absorbing closed class."
        )

    return SolveReport(
        t=t,
        method=used_method,
        residual=float(residual),
        seconds=float(seconds),
        iterations=int(iterations),
        n=n,
    )


# --------------------------------------------------------------------------- #
# 2.6 SVD diagnostics
# --------------------------------------------------------------------------- #
def matrix_condition_number(
    Q: Any,
    *,
    exact: Optional[bool] = None,
    exact_limit: int = 2500,
) -> float:
    """
    Condition number (2-norm) of ``I - Q``.

    Dense SVD is exact but O(n^3); above ``exact_limit`` the extremal singular
    values are estimated with ``svds``. A large value warns that solving
    ``(I - Q) t = 1`` is sensitive to perturbation and that SVD-based or
    iterative methods need care.
    """
    A = _system_matrix(Q)
    n = A.shape[0]
    if exact is None:
        exact = n <= exact_limit

    if exact:
        return float(np.linalg.cond(A.toarray(), 2))

    try:
        s_max = float(
            spla.svds(A, k=1, which="LM", return_singular_vectors=False)[0]
        )
        s_min = float(
            spla.svds(A, k=1, which="SM", return_singular_vectors=False)[0]
        )
        if s_min <= 0:
            return float("inf")
        return s_max / s_min
    except Exception:
        return float("nan")


def svd_rank_truncation_errors(
    Q: Any,
    ranks: Sequence[int] = (1, 2, 5, 10, 20),
) -> List[Dict[str, float]]:
    """
    Rank-truncated SVD approximation error for the expected-time solve.

    For each rank r, reconstruct ``A_r = U_r S_r V_r^T``, solve for the
    expected times with that approximation, and report

        relative_error = ||t_r - t_exact|| / ||t_exact||.

    This quantifies the "SVD helps for small/ill-conditioned systems" claim:
    the smallest rank that keeps the error small is the effective rank of the
    problem.
    """
    A = _system_matrix(Q).toarray()
    n = A.shape[0]
    b = np.ones(n, dtype=np.float64)
    U, s, Vt = sla.svd(A, full_matrices=False)
    t_exact = sla.solve(A, b, assume_a="gen")
    denom = float(np.linalg.norm(t_exact))
    if denom == 0.0:
        denom = 1.0

    out: List[Dict[str, float]] = []
    for rank in ranks:
        r = int(rank)
        if r < 1:
            raise ValueError("ranks must be positive integers")
        r = min(r, n)
        A_r = (U[:, :r] * s[:r]) @ Vt[:r, :]
        t_r, *_ = np.linalg.lstsq(A_r, b, rcond=None)
        rel_error = float(np.linalg.norm(t_r - t_exact) / denom)
        rel_residual = float(np.linalg.norm(A_r @ t_r - b) / np.linalg.norm(b))
        out.append(
            {
                "rank": float(r),
                "relative_error": rel_error,
                "relative_residual": rel_residual,
                "sigma_min_kept": float(s[r - 1]),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Benchmarking helper (kept for the existing demos)
# --------------------------------------------------------------------------- #
def benchmark_solvers(
    sizes: list = [10, 20, 30, 50, 100], max_svd_size: int = 50
) -> Dict[str, Any]:
    """
    Benchmark LU (full N) vs SVD across grid sizes.

    SVD is capped at ``max_svd_size`` because of its O(n^3) cost. This helper
    predates the Phase 2 single-solve API; E2 uses the newer functions so that
    no dense N is formed at large sizes.
    """
    results: Dict[str, Any] = {
        "sizes": [],
        "n_states": [],
        "lu_times": [],
        "svd_times": [],
        "max_diffs": [],
        "notes": [],
    }

    for sz in sizes:
        grid = get_simple_room(sz, sz, exit_width=max(1, sz // 10))
        state_map, _, n_transient, n_absorbing = build_state_map(grid)
        Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

        results["sizes"].append(sz)
        results["n_states"].append(n_transient)

        t0 = time.perf_counter()
        N_lu = solve_fundamental_lu(Q)
        t_lu = time.perf_counter() - t0
        results["lu_times"].append(t_lu)

        if sz <= max_svd_size:
            t0 = time.perf_counter()
            N_svd = solve_fundamental_svd(Q)
            t_svd = time.perf_counter() - t0
            results["svd_times"].append(t_svd)

            max_diff = float(np.max(np.abs(N_lu - N_svd)))
            results["max_diffs"].append(max_diff)
            results["notes"].append(f"LU vs SVD max diff = {max_diff:.2e}")
        else:
            results["svd_times"].append(None)
            results["max_diffs"].append(None)
            results["notes"].append(
                "SVD skipped (O(n^3) cap reached at n > 2500 states)"
            )

    return results


# --------------------------------------------------------------------------- #
# 2.3 Monte Carlo with an absorption guard
# --------------------------------------------------------------------------- #
def _reaches_absorbing(Q: sp.csr_matrix, R: sp.csr_matrix, start: int) -> bool:
    """
    Graph reachability check: can ``start`` reach a state with positive
    probability of absorption? This catches closed transient classes before any
    trial is simulated.
    """
    n = Q.shape[0]
    row_absorbs = np.asarray(R.sum(axis=1)).ravel() > 0.0
    if row_absorbs[start]:
        return True

    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    stack = [start]
    csr = Q.tocsr()
    while stack:
        s = int(stack.pop())
        if row_absorbs[s]:
            return True
        for j in csr.indices[csr.indptr[s] : csr.indptr[s + 1]]:
            j = int(j)
            if j != s and not seen[j]:
                seen[j] = True
                stack.append(j)
    return False


def _build_row_samplers(
    Q: sp.spmatrix, R: sp.spmatrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-state categorical samplers into rectangular arrays."""
    n_transient = Q.shape[0]
    P = sp.hstack([Q, R], format="csr")
    max_deg = 0
    raw: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(n_transient):
        start_ptr, end_ptr = P.indptr[s], P.indptr[s + 1]
        idx = P.indices[start_ptr:end_ptr].astype(np.int64)
        probs = P.data[start_ptr:end_ptr].astype(np.float64)
        if idx.size == 0:
            raise NonAbsorbingChainError(
                f"Transient state {s} has no outgoing transitions."
            )
        if np.any(probs < -1e-12):
            raise ValueError(f"Transient state {s} has a negative transition probability.")
        probs = np.clip(probs, 0.0, None)
        p_sum = float(probs.sum())
        if p_sum <= 0.0:
            raise NonAbsorbingChainError(
                f"Transient state {s} has zero outgoing probability; it cannot reach an exit."
            )
        probs = probs / p_sum
        cum = np.cumsum(probs)
        cum[-1] = 1.0
        raw.append((idx, cum))
        max_deg = max(max_deg, idx.size)

    cdf_pad = np.ones((n_transient, max_deg), dtype=np.float64)
    tgt_pad = np.zeros((n_transient, max_deg), dtype=np.int64)
    deg = np.zeros(n_transient, dtype=np.int64)
    for s, (idx, cum) in enumerate(raw):
        deg[s] = idx.size
        cdf_pad[s, : idx.size] = cum
        tgt_pad[s, : idx.size] = idx
    return cdf_pad, tgt_pad, deg


def run_monte_carlo(
    Q: sp.spmatrix,
    R: sp.spmatrix,
    start_state_idx: int,
    M: int,
    rng: Any = None,
    *,
    max_steps_per_trial: int = 100_000,
) -> Dict[str, Any]:
    """
    Run M independent Monte Carlo trials on a fixed absorbing chain [Q | R].

    The trials are vectorised across agents: one categorical draw per active
    trial per step. This keeps M = 1e5 practical for E1 while preserving the
    same statistical contract as the original scalar loop.

    Parameters
    ----------
    max_steps_per_trial:
        Hard guard against a non-absorbing (or effectively non-absorbing)
        chain. Exceeding it raises :class:`NonAbsorbingChainError` instead of
        hanging.

    Returns
    -------
    dict with ``trial_times`` (shape M), ``mean`` and ``std_error``.
    """
    Q_csr = sp.csr_matrix(Q, dtype=np.float64)
    R_csr = sp.csr_matrix(R, dtype=np.float64)
    n_transient = Q_csr.shape[0]
    if R_csr.shape[0] != n_transient:
        raise ValueError(
            f"R has {R_csr.shape[0]} rows but Q has {n_transient} transient states."
        )
    if not (0 <= start_state_idx < n_transient):
        raise ValueError(
            f"start_state_idx {start_state_idx} is out of bounds for n_transient={n_transient}."
        )
    if isinstance(M, bool) or not isinstance(M, (int, np.integer)) or M <= 0:
        raise ValueError("M must be a positive integer.")
    if (
        isinstance(max_steps_per_trial, bool)
        or not isinstance(max_steps_per_trial, (int, np.integer))
        or max_steps_per_trial <= 0
    ):
        raise ValueError("max_steps_per_trial must be a positive integer.")

    if not _reaches_absorbing(Q_csr, R_csr, int(start_state_idx)):
        raise NonAbsorbingChainError(
            f"State {start_state_idx} cannot reach any absorbing state. "
            "Check that Q has no closed transient class and that R has a "
            "positive row on every absorbing path."
        )

    if rng is None or isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        gen = rng
    else:
        raise TypeError("rng must be None, an integer seed, or a numpy Generator")

    cdf_pad, tgt_pad, deg = _build_row_samplers(Q_csr, R_csr)

    curr = np.full(int(M), int(start_state_idx), dtype=np.int64)
    steps = np.zeros(int(M), dtype=np.int64)
    active = np.ones(int(M), dtype=bool)
    iterations = 0

    while np.any(active):
        iterations += 1
        if iterations > max_steps_per_trial:
            raise NonAbsorbingChainError(
                f"Monte Carlo trials exceeded max_steps_per_trial="
                f"{max_steps_per_trial} without absorption. The chain is likely "
                "non-absorbing or numerically degenerate."
            )
        rows = curr[active]
        u = gen.random(rows.size)
        idx = np.sum(u[:, None] > cdf_pad[rows], axis=1)
        idx = np.minimum(idx, deg[rows] - 1)
        nxt = tgt_pad[rows, idx]
        steps[active] += 1
        curr[active] = nxt
        active = curr < n_transient

    mean_val = float(np.mean(steps))
    std_val = float(np.std(steps, ddof=1)) if M > 1 else 0.0
    std_error = std_val / np.sqrt(M) if M > 0 else 0.0

    return {
        "trial_times": steps,
        "mean": mean_val,
        "std_error": float(std_error),
    }
