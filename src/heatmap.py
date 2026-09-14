import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Dict, Tuple, Optional
from src.floor_plan import CELL_WALL, CELL_WALKABLE, CELL_EXIT


def fundamental_to_heatmap(N: np.ndarray, 
                           grid: np.ndarray, 
                           state_map: Dict[Tuple[int, int], int], 
                           n_transient: int) -> np.ndarray:
    """
    Maps the fundamental matrix N = (I - Q)^(-1) row sums back to a 2D floor plan grid.
    
    Each row sum sum_j N_ij gives the expected total time steps spent in the building 
    starting from transient cell i before reaching an exit.
    
    Returns:
        heatmap_grid: 2D float array matching grid shape. Walkable cells contain expected
                      evacuation times, wall cells are NaN, exit cells are 0.0.
    """
    row_sums = np.sum(N, axis=1)
    heatmap = np.full(grid.shape, np.nan, dtype=np.float64)

    for (r, c), idx in state_map.items():
        if idx < n_transient:
            heatmap[r, c] = row_sums[idx]
        elif grid[r, c] == CELL_EXIT:
            heatmap[r, c] = 0.0

    return heatmap


def plot_heatmap(heatmap: np.ndarray, 
                 grid: np.ndarray, 
                 title: str = "Static Fundamental Matrix Bottleneck Heatmap", 
                 save_path: Optional[str] = None, 
                 ax=None):
    """
    Renders and saves the bottleneck heatmap overlay on the floor plan grid.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    # Create masked array to draw walls distinctly
    masked_heatmap = np.ma.masked_invalid(heatmap)
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='#1e1e24')  # Walls shown in dark charcoal

    im = ax.imshow(masked_heatmap, cmap=cmap, origin='upper')

    # Draw exit cells cleanly in bright green
    exit_mask = (grid == CELL_EXIT)
    if np.any(exit_mask):
        exit_r, exit_c = np.where(exit_mask)
        ax.scatter(exit_c, exit_r, color='#2ec4b6', marker='s', s=40, label='Exit (Absorbing State)', zorder=4)

    # Highlight top 3 bottleneck locations (cells with highest expected time to absorption)
    valid_coords = np.argwhere(~np.isnan(heatmap) & (grid == CELL_WALKABLE))
    if len(valid_coords) > 0:
        values = heatmap[valid_coords[:, 0], valid_coords[:, 1]]
        top_k_indices = np.argsort(values)[::-1][:3]
        for rank, k_idx in enumerate(top_k_indices, 1):
            br, bc = valid_coords[k_idx]
            b_val = values[k_idx]
            ax.scatter(bc, br, color='#ff0054', marker='o', s=70, edgecolors='white', linewidth=1.2, zorder=5)
            if rank == 1:
                ax.annotate(f"Peak Bottleneck\n({b_val:.1f} steps)", xy=(bc, br), xytext=(bc + 1.5, br + 1.5),
                            arrowprops=dict(arrowstyle='->', color='#ff0054', lw=1.5),
                            fontsize=9, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e24', alpha=0.85, edgecolor='#ff0054'))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ticks_x = np.arange(0, grid.shape[1], 5)
    ticks_y = np.arange(0, grid.shape[0], 5)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(ticks_x)
    ax.set_yticklabels(ticks_y)
    ax.grid(True, color='#ffffff', linestyle=':', linewidth=0.5, alpha=0.3)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Expected Evacuation Time Steps", fontsize=10, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax
