import sys
sys.path.insert(0, ".")
import numpy as np
from scratch.test_asymmetric import optimize_wall_test

# Off-center pillar in top-right quadrant: rows 3..7, cols 12..16 (size 4x4)
pillar_offcenter = [(r, c) for r in range(3, 7) for c in range(12, 16)]

print("=== Off-Center 4x4 Pillar Room (20x20) ===")
for wall in ["top", "bottom", "left", "right"]:
    res0 = optimize_wall_test(wall, 20, 20, 2, x0_val=0.0, obstacles=pillar_offcenter)
    res_mid = optimize_wall_test(wall, 20, 20, 2, x0_val=8.0, obstacles=pillar_offcenter)
    res_max = optimize_wall_test(wall, 20, 20, 2, x0_val=16.0, obstacles=pillar_offcenter)
    print(f"Wall {wall:<6}: seed 0 -> pos={res0['rounded_pos']} (J={res0['actual_j']:.1f}) | seed 8 -> pos={res_mid['rounded_pos']} (J={res_mid['actual_j']:.1f}) | seed 16 -> pos={res_max['rounded_pos']} (J={res_max['actual_j']:.1f})")

# Let's also test a larger 6x8 off-center pillar (rows 3..9, cols 10..18 - leaving hallway on right or cols 10..16)
pillar_large = [(r, c) for r in range(3, 10) for c in range(10, 16)]
print("\n=== Large Off-Center 7x6 Pillar Room (20x20) ===")
for wall in ["top", "bottom", "left", "right"]:
    res0 = optimize_wall_test(wall, 20, 20, 2, x0_val=0.0, obstacles=pillar_large)
    res_mid = optimize_wall_test(wall, 20, 20, 2, x0_val=8.0, obstacles=pillar_large)
    res_max = optimize_wall_test(wall, 20, 20, 2, x0_val=16.0, obstacles=pillar_large)
    print(f"Wall {wall:<6}: seed 0 -> pos={res0['rounded_pos']} (J={res0['actual_j']:.1f}) | seed 8 -> pos={res_mid['rounded_pos']} (J={res_mid['actual_j']:.1f}) | seed 16 -> pos={res_max['rounded_pos']} (J={res_max['actual_j']:.1f})")
