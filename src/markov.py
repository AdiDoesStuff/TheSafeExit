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
