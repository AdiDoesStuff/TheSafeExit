import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Grid cell encodings
CELL_WALL = 0
CELL_WALKABLE = 1
CELL_EXIT = 2


def get_simple_room(rows: int = 20, cols: int = 20, exit_width: int = 2) -> np.ndarray:
    """
    Creates a simple rectangular room layout.
    Outer boundary consists of walls, except for an exit opening on the right boundary.
    Inside is fully walkable.
    """
    if rows < 5 or cols < 5:
        raise ValueError("Room dimensions must be at least 5x5.")

    grid = np.full((rows, cols), CELL_WALKABLE, dtype=int)
    
    # Outer walls
    grid[0, :] = CELL_WALL
    grid[-1, :] = CELL_WALL
    grid[:, 0] = CELL_WALL
    grid[:, -1] = CELL_WALL
    
    # Exit placement (middle of the right wall)
    mid = rows // 2
    half_w = max(1, exit_width // 2)
    start_r = max(1, mid - half_w)
    end_r = min(rows - 1, start_r + exit_width)
    grid[start_r:end_r, -1] = CELL_EXIT
    
    return grid


def get_hallway(rows: int = 25, cols: int = 25, corridor_width: int = 3, exit_width: int = 1) -> np.ndarray:
    """
    Creates an L-shaped hallway corridor with a chokepoint bottleneck before a single exit.
    """
    if rows < 10 or cols < 10:
        raise ValueError("Hallway dimensions must be at least 10x10.")

    grid = np.full((rows, cols), CELL_WALL, dtype=int)
    
    # Corridor 1: Horizontal section along the top
    grid[1:1 + corridor_width, 1:cols - 1] = CELL_WALKABLE
    
    # Corridor 2: Vertical section down the right
    grid[1:rows - 1, cols - 1 - corridor_width:cols - 1] = CELL_WALKABLE

    # Create a constriction / wall obstruction to simulate a bottleneck chokepoint near the end
    choke_r = rows - 4
    grid[choke_r, cols - 1 - corridor_width : cols - 2] = CELL_WALL
    
    # Exit at the bottom right end of the corridor
    exit_start = rows - 2
    grid[exit_start : exit_start + exit_width, cols - 2] = CELL_EXIT
    
    return grid


def validate_reachability(grid: np.ndarray) -> bool:
    """
    Validates that every walkable cell (1) has a path to at least one exit cell (2) via BFS.
    Raises ValueError if isolated walkable cells exist.
    """
    rows, cols = grid.shape
    exits = list(zip(*np.where(grid == CELL_EXIT)))
    if not exits:
        raise ValueError("Floor plan has no exit cells (CELL_EXIT = 2).")

    walkable_coords = set(zip(*np.where(grid == CELL_WALKABLE)))
    if not walkable_coords:
        raise ValueError("Floor plan has no walkable cells (CELL_WALKABLE = 1).")

    # Multi-source BFS starting from all exit cells
    queue = deque(exits)
    visited = set(exits)

    # 4-connected grid directions (Up, Down, Left, Right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if (nr, nc) not in visited and grid[nr, nc] == CELL_WALKABLE:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    unreachable = walkable_coords - visited
    if unreachable:
        raise ValueError(
            f"Floor plan reachability validation failed! Found {len(unreachable)} unreachable walkable cells "
            f"out of {len(walkable_coords)} total. Unreachable sample: {list(unreachable)[:5]}"
        )
    return True


def plot_floor_plan(grid: np.ndarray, title: str = "Floor Plan Layout", ax=None, save_path: str = None):
    """
    Plots the floor plan using color coding (Wall=Black, Walkable=White, Exit=Green).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Colormap: 0=Wall (dark charcoal), 1=Walkable (light grey/white), 2=Exit (emerald green)
    cmap = mcolors.ListedColormap(['#1e1e24', '#f4f4f6', '#2ec4b6'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(grid, cmap=cmap, norm=norm, origin='upper')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ticks_x = np.arange(0, grid.shape[1], 5)
    ticks_y = np.arange(0, grid.shape[0], 5)
    ax.set_xticks(ticks_x)
    ax.set_yticks(ticks_y)
    ax.set_xticklabels(ticks_x)
    ax.set_yticklabels(ticks_y)
    ax.grid(True, color='#888888', linestyle=':', linewidth=0.5, alpha=0.5)

    # Colorbar legend
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.7)
    cbar.ax.set_yticklabels(['Wall (0)', 'Walkable (1)', 'Exit (2)'])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax


def compute_exit_distance_field(grid: np.ndarray) -> np.ndarray:
    """
    Computes geodesic (grid-step) shortest distance from every walkable cell
    to the nearest exit cell using multi-source BFS.

    Returns:
        dist: 2D np.ndarray of shape grid.shape with integer distances.
              Exit cells have distance 0.
              Walkable cells have shortest path distance >= 1 to an exit.
              Wall cells have distance -1.
    """
    rows, cols = grid.shape
    dist = np.full((rows, cols), -1, dtype=int)

    exits = list(zip(*np.where(grid == CELL_EXIT)))
    if not exits:
        raise ValueError("Floor plan has no exit cells (CELL_EXIT = 2).")

    queue = deque()
    for r, c in exits:
        dist[r, c] = 0
        queue.append((r, c))

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != CELL_WALL and dist[nr, nc] == -1:
                    dist[nr, nc] = dist[r, c] + 1
                    queue.append((nr, nc))

    return dist

