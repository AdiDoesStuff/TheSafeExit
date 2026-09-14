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

    return results
