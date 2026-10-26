import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla
from typing import Dict, Any, Tuple
from src.floor_plan import get_simple_room
from src.markov import build_state_map, build_QR


def solve_fundamental_lu(Q: sp.csr_matrix) -> np.ndarray:
    """
    Solves the linear system (I - Q) N = I for the Fundamental Matrix N using sparse LU decomposition.
    
    Sparse LU (via scipy.sparse.linalg.splu) factorizes A = (I - Q) into L and U triangular matrices,
    allowing efficient forward/backward substitution rather than explicit matrix inversion.
    
    Returns:
        N: Dense numpy ndarray of shape (n_transient, n_transient)
    """
    n = Q.shape[0]
    I_sparse = sp.eye(n, format='csc', dtype=np.float64)
    A_csc = (I_sparse - Q).tocsc()

    # Perform SuperLU sparse LU decomposition
    lu_factor = spla.splu(A_csc)

    # Solve A N = I by passing identity matrix I
    I_dense = np.eye(n, dtype=np.float64)
    N = lu_factor.solve(I_dense)
    return N


def solve_fundamental_svd(Q: sp.csr_matrix) -> np.ndarray:
    """
    Solves (I - Q) N = I using dense Singular Value Decomposition (SVD).
    
    A = U S V^T  =>  A^(-1) = V S^(-1) U^T
    
    Note: Dense SVD requires converting (I - Q) to a dense array and performs O(n^3) operations.
    It is intended for small to medium matrix sizes (e.g. n <= 2500).
    """
    n = Q.shape[0]
    A_dense = (sp.eye(n, dtype=np.float64) - Q).toarray()

    U, s, Vt = sla.svd(A_dense)
    
    # Invert singular values (s > 1e-12 for non-singular A)
    s_inv = np.where(s > 1e-12, 1.0 / s, 0.0)
    
    # N = V * diag(1/s) * U^T
    N = (Vt.T * s_inv) @ U.T
    return N


# Aliases for convenience and standardized naming
fundamental_matrix_lu = solve_fundamental_lu
fundamental_matrix_svd = solve_fundamental_svd



def benchmark_solvers(sizes: list = [10, 20, 30, 50, 100], max_svd_size: int = 50) -> Dict[str, Any]:
    """
    Benchmarks solve time (in seconds) for LU vs SVD solvers across grid sizes.
    
    Caps SVD solver execution at max_svd_size (default 50x50 grid = ~2500 states)
    due to O(n^3) computational complexity of dense SVD.
    Runs LU up to 100x100 grid (10,000 states).
    """
    results = {
        "sizes": [],
        "n_states": [],
        "lu_times": [],
        "svd_times": [],
        "max_diffs": [],
        "notes": []
    }

    for sz in sizes:
        grid = get_simple_room(sz, sz, exit_width=max(1, sz // 10))
        state_map, _, n_transient, n_absorbing = build_state_map(grid)
        Q, R = build_QR(grid, state_map, n_transient, n_absorbing)

        results["sizes"].append(sz)
        results["n_states"].append(n_transient)

        # 1. Measure LU Solve Time
        t0 = time.perf_counter()
        N_lu = solve_fundamental_lu(Q)
        t_lu = time.perf_counter() - t0
        results["lu_times"].append(t_lu)

        # 2. Measure SVD Solve Time (if grid size <= max_svd_size)
        if sz <= max_svd_size:
            t0 = time.perf_counter()
            N_svd = solve_fundamental_svd(Q)
            t_svd = time.perf_counter() - t0
            results["svd_times"].append(t_svd)

            # Measure agreement between LU and SVD solutions
            max_diff = np.max(np.abs(N_lu - N_svd))
            results["max_diffs"].append(max_diff)
            results["notes"].append(f"LU vs SVD max diff = {max_diff:.2e}")
        else:
            results["svd_times"].append(None)
            results["max_diffs"].append(None)
            results["notes"].append("SVD skipped (O(n^3) cap reached at n > 2500 states)")

def run_monte_carlo(
    Q: sp.spmatrix,
    R: sp.spmatrix,
    start_state_idx: int,
    M: int,
    rng: Any = None,
) -> Dict[str, Any]:
    """
    Runs M independent Monte Carlo trials on a fixed absorbing Markov chain [Q | R].

    Parameters:
        Q: Sparse transition matrix among transient states (shape n_transient x n_transient).
        R: Sparse transition matrix from transient to absorbing exit states (shape n_transient x n_absorbing).
        start_state_idx: Transient state index (0 to n_transient - 1) where agent begins.
        M: Number of independent simulation trials.
        rng: Optional np.random.Generator or integer seed.

    Returns:
        dict containing:
            'trial_times': np.ndarray of shape (M,) with steps to absorption per trial.
            'mean': Sample mean evacuation time across M trials.
            'std_error': Standard error of the sample mean (std / sqrt(M)).
    """
    n_transient = Q.shape[0]
    if not (0 <= start_state_idx < n_transient):
        raise ValueError(f"start_state_idx {start_state_idx} is out of bounds for n_transient={n_transient}.")
    if M <= 0:
        raise ValueError("M must be a positive integer.")

    if rng is None or isinstance(rng, (int, np.integer)):
        gen = np.random.default_rng(rng)
    elif isinstance(rng, np.random.Generator):
        gen = rng
    else:
        gen = np.random.default_rng()

    P = sp.hstack([Q, R], format="csr")

    # Precompute row CDFs and targets for fast categorical sampling
    row_targets = []
    row_cdfs = []
    for s in range(n_transient):
        start_ptr = P.indptr[s]
        end_ptr = P.indptr[s + 1]
        idx = P.indices[start_ptr:end_ptr]
        probs = P.data[start_ptr:end_ptr].astype(np.float64)
        if len(probs) == 0:
            raise ValueError(f"Transient state {s} has no outgoing transitions.")
        p_sum = probs.sum()
        if p_sum <= 0:
            raise ValueError(f"Transient state {s} has non-positive outgoing probability sum.")
        probs = probs / p_sum
        cum = np.cumsum(probs)
        cum[-1] = 1.0  # Guarantee upper bound for searchsorted
        row_targets.append(idx)
        row_cdfs.append(cum)

    trial_times = np.zeros(M, dtype=int)
    for m in range(M):
        curr = start_state_idx
        steps = 0
        while curr < n_transient:
            u = gen.random()
            targets = row_targets[curr]
            cdfs = row_cdfs[curr]
            next_s = targets[np.searchsorted(cdfs, u)]
            steps += 1
            curr = next_s
        trial_times[m] = steps

    mean_val = float(np.mean(trial_times))
    std_val = float(np.std(trial_times, ddof=1)) if M > 1 else 0.0
    std_error = std_val / np.sqrt(M) if M > 0 else 0.0

    return {
        "trial_times": trial_times,
        "mean": mean_val,
        "std_error": float(std_error),
    }
