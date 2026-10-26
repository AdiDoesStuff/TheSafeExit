import sys
sys.path.insert(0, ".")
import numpy as np
from src.optimization import build_grid_with_exit_on_wall, worst_case_evacuation_time
from scratch.test_asymmetric import optimize_wall_test

# Test asymmetric partition wall: 20x20 room with a vertical barrier at col 10 from row 1 to 13
# Leaving an opening only at rows 14..18
barrier = [(r, 10) for r in range(1, 14)]

print("=== Asymmetric Partition Wall Room (20x20) ===")
for wall in ["top", "bottom", "left", "right"]:
    interior_len = 18
    max_p = 16
    res0 = optimize_wall_test(wall, 20, 20, 2, x0_val=0.0, obstacles=barrier)
    res_mid = optimize_wall_test(wall, 20, 20, 2, x0_val=8.0, obstacles=barrier)
    res_max = optimize_wall_test(wall, 20, 20, 2, x0_val=16.0, obstacles=barrier)
    print(f"Wall {wall:<6}: seed 0 -> pos={res0['rounded_pos']} (J={res0['actual_j']:.1f}) | seed 8 -> pos={res_mid['rounded_pos']} (J={res_mid['actual_j']:.1f}) | seed 16 -> pos={res_max['rounded_pos']} (J={res_max['actual_j']:.1f})")
