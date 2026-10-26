import sys
sys.path.insert(0, ".")
import numpy as np
from src.floor_plan import get_room_with_barrier, compute_exit_distance_field, CELL_WALKABLE, CELL_EXIT

# Build the barrier room with exit on bottom wall at pos 8 (row 19, cols 9..10)
grid_bottom = get_room_with_barrier(20, 20, exit_width=2, barrier_col=10, barrier_row_end=13)
# Move exit to bottom wall at pos 8
grid_bottom[1:19, -1] = 0  # right wall is solid
grid_bottom[-1, 9:11] = CELL_EXIT # bottom exit
dist_bottom = compute_exit_distance_field(grid_bottom)

# Farthest points
walkable_mask = (grid_bottom == CELL_WALKABLE)
max_d = np.max(dist_bottom[walkable_mask])
farthest_cells = list(zip(*np.where(dist_bottom == max_d)))

print(f"Bottom-wall exit at cols 9..10 on row 19:")
print(f"Max distance J = {max_d}")
print(f"Farthest cells: {farthest_cells}")

for r, c in farthest_cells:
    print(f"Cell ({r}, {c}): dist = {dist_bottom[r, c]}")
    # Trace shortest path to exit
    curr = (r, c)
    path = [curr]
    while dist_bottom[curr[0], curr[1]] > 0:
        cr, cc = curr
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < 20 and 0 <= nc < 20 and grid_bottom[nr, nc] != 0:
                if dist_bottom[nr, nc] == dist_bottom[cr, cc] - 1:
                    curr = (nr, nc)
                    path.append(curr)
                    break
    print(f"  Path from ({r}, {c}) to exit: length={len(path)-1}")
    print(f"  Path coords: {path[:4]} ... {path[-3:]}")
    # Check if path crosses barrier col 10 at rows 1..13
    touches_barrier = any(pr <= 13 and pc == 10 for pr, pc in path)
    print(f"  Does path ever touch or route around barrier? {touches_barrier}")
