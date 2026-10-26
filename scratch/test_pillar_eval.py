import sys
sys.path.insert(0, ".")
import numpy as np
from src.floor_plan import get_room_with_pillar, compute_exit_distance_field
from src.optimization import optimize_exit_placement

# 1. Centered Pillar Room
grid_pillar_center = get_room_with_pillar(20, 20, exit_width=2, pillar_size=4, center=(10, 10))
pillar_center_obs = list(zip(*np.where(grid_pillar_center[1:19, 1:19] == 0)))
pillar_center_obs = [(r + 1, c + 1) for r, c in pillar_center_obs]

res_pc = optimize_exit_placement(20, 20, 2, obstacles=pillar_center_obs)
print("=== Centered 4x4 Pillar Room ===")
print(f"Best Wall: {res_pc['best_wall']} | Best Pos: {res_pc['best_position']} | J = {res_pc['best_J']:.2f}")
for w in ["top", "bottom", "left", "right"]:
    print(f"  {w:<6}: pos={res_pc['walls'][w]['rounded_position']} J={res_pc['walls'][w]['actual_J']:.2f}")

# 2. Off-Center 4x4 Pillar Room (shifted to top-right quadrant: center=(5, 14))
grid_pillar_off = get_room_with_pillar(20, 20, exit_width=2, pillar_size=4, center=(5, 14))
pillar_off_obs = list(zip(*np.where(grid_pillar_off[1:19, 1:19] == 0)))
pillar_off_obs = [(r + 1, c + 1) for r, c in pillar_off_obs]

res_po = optimize_exit_placement(20, 20, 2, obstacles=pillar_off_obs)
print("\n=== Off-Center 4x4 Pillar Room (center=(5, 14)) ===")
print(f"Best Wall: {res_po['best_wall']} | Best Pos: {res_po['best_position']} | J = {res_po['best_J']:.2f}")
for w in ["top", "bottom", "left", "right"]:
    print(f"  {w:<6}: pos={res_po['walls'][w]['rounded_position']} J={res_po['walls'][w]['actual_J']:.2f}")
