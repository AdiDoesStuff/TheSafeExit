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


def get_two_exit_room(rows: int = 20, cols: int = 20, exit_width: int = 2) -> np.ndarray:
    """
    Creates a rectangular room layout with two exits on opposite walls (left and right).
    """
    if rows < 5 or cols < 5:
        raise ValueError("Room dimensions must be at least 5x5.")

    grid = np.full((rows, cols), CELL_WALKABLE, dtype=int)

    # Outer boundary walls
    grid[0, :] = CELL_WALL
    grid[-1, :] = CELL_WALL
    grid[:, 0] = CELL_WALL
    grid[:, -1] = CELL_WALL

    # Exit placement (middle of left wall and middle of right wall)
    mid = rows // 2
    half_w = max(1, exit_width // 2)
    start_r = max(1, mid - half_w)
    end_r = min(rows - 1, start_r + exit_width)

    grid[start_r:end_r, 0] = CELL_EXIT   # Left exit
    grid[start_r:end_r, -1] = CELL_EXIT  # Right exit

    return grid


def get_room_with_pillar(
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    pillar_size: int = 4,
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Creates a rectangular room with an exit on the right wall and a solid square pillar (wall obstruction).
    The pillar can be placed in the center (default) or at a custom (row, col) center position.
    """
    if rows < 7 or cols < 7:
        raise ValueError("Room dimensions must be at least 7x7 for a central pillar.")
    if pillar_size >= min(rows - 2, cols - 2):
        raise ValueError("Pillar size is too large for the room dimensions.")

    grid = get_simple_room(rows, cols, exit_width=exit_width)

    # Pillar placement
    if center is None:
        mid_r = rows // 2
        mid_c = cols // 2
    else:
        mid_r, mid_c = center

    half_p = pillar_size // 2
    r_start = mid_r - half_p
    r_end = r_start + pillar_size
    c_start = mid_c - half_p
    c_end = c_start + pillar_size

    if r_start < 1 or r_end > rows - 1 or c_start < 1 or c_end > cols - 1:
        raise ValueError(f"Pillar bounds [{r_start}:{r_end}, {c_start}:{c_end}] exceed room interior.")

    grid[r_start:r_end, c_start:c_end] = CELL_WALL

    return grid


def get_room_with_barrier(
    rows: int = 20,
    cols: int = 20,
    exit_width: int = 2,
    barrier_col: Optional[int] = None,
    barrier_row_end: Optional[int] = None,
) -> np.ndarray:
    """
    Creates a rectangular room with a vertical dividing barrier wall protruding downwards from
    the top boundary, leaving a single passageway/chokepoint near the bottom.

    Parameters:
        rows, cols: Room dimensions (minimum 7x7).
        exit_width: Width of the exit opening on the boundary.
        barrier_col: Column index for the vertical barrier (defaults to mid column).
        barrier_row_end: Ending row index for barrier (defaults to ~2/3 of room height, leaving bottom passage).

    Returns:
        grid: 2D numpy array representing the floor plan.
    """
    if rows < 7 or cols < 7:
        raise ValueError("Room dimensions must be at least 7x7 for a barrier room.")

    grid = get_simple_room(rows, cols, exit_width=exit_width)

    if barrier_col is None:
        barrier_col = cols // 2
    if barrier_row_end is None:
        barrier_row_end = max(2, int(rows * 0.65))

    if not (1 < barrier_col < cols - 1):
        raise ValueError(f"barrier_col {barrier_col} must be strictly inside the interior (1..{cols-2}).")
    if not (1 <= barrier_row_end < rows - 2):
        raise ValueError(f"barrier_row_end {barrier_row_end} must leave at least 2 walkable rows at bottom.")

    grid[1:barrier_row_end + 1, barrier_col] = CELL_WALL
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

