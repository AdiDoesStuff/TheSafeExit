import numpy as np
import scipy.sparse as sp
from typing import Tuple, Dict
from src.floor_plan import CELL_WALL, CELL_WALKABLE, CELL_EXIT, validate_reachability


def build_state_map(grid: np.ndarray) -> Tuple[Dict[Tuple[int, int], int], Dict[int, Tuple[int, int]], int, int]:
    """
    Constructs a 1D state index mapping from a 2D floor plan grid.
    
    Validates reachability first.
    Indices 0 to n_transient - 1 map to transient cells (CELL_WALKABLE = 1).
    Indices n_transient to n_transient + n_absorbing - 1 map to exit cells (CELL_EXIT = 2).
    
    Returns:
        state_map: Dict mapping (row, col) tuple to 1D state index
        index_to_state: Dict mapping 1D state index to (row, col) tuple
        n_transient: Number of transient states
        n_absorbing: Number of absorbing exit states
    """
    validate_reachability(grid)

    walkable_coords = list(zip(*np.where(grid == CELL_WALKABLE)))
    exit_coords = list(zip(*np.where(grid == CELL_EXIT)))

    n_transient = len(walkable_coords)
    n_absorbing = len(exit_coords)

    state_map = {}
    index_to_state = {}

    # Assign transient state indices [0, n_transient - 1]
    for idx, coord in enumerate(walkable_coords):
        state_map[coord] = idx
        index_to_state[idx] = coord

    # Assign absorbing exit state indices [n_transient, n_transient + n_absorbing - 1]
    for idx, coord in enumerate(exit_coords):
        state_idx = n_transient + idx
        state_map[coord] = state_idx
        index_to_state[state_idx] = coord

    return state_map, index_to_state, n_transient, n_absorbing


def build_QR(grid: np.ndarray, state_map: Dict[Tuple[int, int], int], n_transient: int, n_absorbing: int) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs sparse transition matrices Q (transient->transient) and R (transient->absorbing)
    for an unbiased grid random walk.
    
    For each cell, 4 directional moves (Up, Down, Left, Right) each have 0.25 probability.
    - If neighbor is a transient cell: probability 0.25 goes to Q[i, j].
    - If neighbor is an exit cell: probability 0.25 goes to R[i, k].
    - If neighbor is a wall (0) or grid boundary: agent stays in current cell (self-loop with prob 0.25).
    """
    rows, cols = grid.shape
    Q = sp.lil_matrix((n_transient, n_transient), dtype=np.float64)
    R = sp.lil_matrix((n_transient, n_absorbing), dtype=np.float64)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for (r, c), i in state_map.items():
        if i >= n_transient:
            continue  # Skip absorbing exit cells

        self_loop_prob = 0.0
        p_step = 0.25  # Unbiased 4-way direction probability

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cell_val = grid[nr, nc]
                if cell_val == CELL_WALL:
                    self_loop_prob += p_step
                elif cell_val == CELL_WALKABLE:
                    j = state_map[(nr, nc)]
                    Q[i, j] += p_step
                elif cell_val == CELL_EXIT:
                    exit_idx = state_map[(nr, nc)] - n_transient
                    R[i, exit_idx] += p_step
            else:
                # Bounded by grid edge (wall)
                self_loop_prob += p_step

        if self_loop_prob > 0.0:
            Q[i, i] += self_loop_prob

    return Q.tocsr(), R.tocsr()


# Alias for clarity: existing build_QR generates the uniform random walk
build_uniform_QR = build_QR


def build_rational_QR(
    grid: np.ndarray,
    distance_field: np.ndarray,
    state_map: Dict[Tuple[int, int], int],
    n_transient: int,
    n_absorbing: int,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Constructs steepest-descent transition matrices Q (transient->transient)
    and R (transient->absorbing) based on the geodesic exit distance field.

    From each transient cell i at (r, c), agents move uniformly only among
    passable neighbor cells with distance exactly equal to (distance(r, c) - 1).
    Ties are split equally.

    Returns:
        Q: scipy.sparse.csr_matrix of shape (n_transient, n_transient)
        R: scipy.sparse.csr_matrix of shape (n_transient, n_absorbing)
    """
    rows, cols = grid.shape
    Q = sp.lil_matrix((n_transient, n_transient), dtype=np.float64)
    R = sp.lil_matrix((n_transient, n_absorbing), dtype=np.float64)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for (r, c), i in state_map.items():
        if i >= n_transient:
            continue  # Skip absorbing exit cells

        d_curr = distance_field[r, c]
        descent_neighbors = []

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != CELL_WALL and distance_field[nr, nc] == d_curr - 1:
                    descent_neighbors.append((nr, nc))

        if not descent_neighbors:
            raise ValueError(
                f"No descent neighbor found for cell {(r, c)} with distance {d_curr}. "
                "Ensure floor plan reachability validation passed."
            )

        p_step = 1.0 / len(descent_neighbors)

        for nr, nc in descent_neighbors:
            cell_val = grid[nr, nc]
            if cell_val == CELL_WALKABLE:
                j = state_map[(nr, nc)]
                Q[i, j] += p_step
            elif cell_val == CELL_EXIT:
                exit_idx = state_map[(nr, nc)] - n_transient
                R[i, exit_idx] += p_step

    return Q.tocsr(), R.tocsr()


def build_mixed_QR(
    Q_rational: sp.csr_matrix,
    R_rational: sp.csr_matrix,
    Q_uniform: sp.csr_matrix,
    R_uniform: sp.csr_matrix,
    lam: np.ndarray,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Performs row-wise convex combination of rational and uniform transition matrices:

        Q_t[i, :] = (1 - lam[i]) * Q_rational[i, :] + lam[i] * Q_uniform[i, :]
        R_t[i, :] = (1 - lam[i]) * R_rational[i, :] + lam[i] * R_uniform[i, :]

    Parameters:
        Q_rational, R_rational: Steepest-descent transition matrices.
        Q_uniform, R_uniform: Unbiased random-walk transition matrices.
        lam: Array of shape (n_transient,) or scalar, with values in [0.0, 1.0].

    Returns:
        Q_t, R_t: Mutated sparse transition matrices satisfying row stochasticity.
    """
    n_transient = Q_rational.shape[0]
    lam_arr = np.asarray(lam, dtype=np.float64)

    if lam_arr.ndim == 0:
        lam_arr = np.full(n_transient, float(lam_arr), dtype=np.float64)
    elif lam_arr.shape != (n_transient,):
        raise ValueError(
            f"lam array shape {lam_arr.shape} does not match n_transient={n_transient}"
        )

    lam_arr = np.clip(lam_arr, 0.0, 1.0)

    one_minus_lam = sp.diags(1.0 - lam_arr, offsets=0, shape=(n_transient, n_transient), format="csr")
    lam_diag = sp.diags(lam_arr, offsets=0, shape=(n_transient, n_transient), format="csr")

    Q_t = (one_minus_lam @ Q_rational + lam_diag @ Q_uniform).tocsr()
    R_t = (one_minus_lam @ R_rational + lam_diag @ R_uniform).tocsr()

    return Q_t, R_t


def get_markov_stats(Q: sp.csr_matrix, R: sp.csr_matrix, n_transient: int, n_absorbing: int) -> dict:
    """
    Returns summary statistics for the constructed Markov chain.
    """
    nnz_Q = Q.nnz
    nnz_R = R.nnz
    density_Q = nnz_Q / (n_transient * n_transient) if n_transient > 0 else 0.0

    return {
        "n_transient": n_transient,
        "n_absorbing": n_absorbing,
        "shape_Q": Q.shape,
        "shape_R": R.shape,
        "nnz_Q": nnz_Q,
        "nnz_R": nnz_R,
        "density_Q": density_Q,
    }

