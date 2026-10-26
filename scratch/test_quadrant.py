import sys
sys.path.insert(0, ".")
import numpy as np
from scratch.test_asymmetric import optimize_wall_test

# Room 20x20 with large off-center pillar occupying top-left block: rows 1..10, cols 1..10
# (leaving an L-shaped walkable space)
pillar_quadrant = [(r, c) for r in range(1, 11) for c in range(1, 11)]

print("=== Off-Center Quadrant Pillar Room (20x20) ===")
for wall in ["top", "bottom", "left", "right"]:
    max_p = 16
    res0 = optimize_wall_test(wall, 20, 20, 2, x0_val=0.0, obstacles=pillar_quadrant)
    res_mid = optimize_wall_test(wall, 20, 20, 2, x0_val=8.0, obstacles=pillar_quadrant)
    res_max = optimize_wall_test(wall, 20, 20, 2, x0_val=16.0, obstacles=pillar_quadrant)
    print(f"Wall {wall:<6}: seed 0 -> pos={res0['rounded_pos']} (J={res0['actual_j']:.1f}) | seed 8 -> pos={res_mid['rounded_pos']} (J={res_mid['actual_j']:.1f}) | seed 16 -> pos={res_max['rounded_pos']} (J={res_max['actual_j']:.1f})")
