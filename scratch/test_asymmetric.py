import sys
sys.path.insert(0, ".")
import numpy as np
from scipy.optimize import minimize, approx_fprime
from src.optimization import build_grid_with_exit_on_wall, worst_case_evacuation_time

def optimize_wall_test(wall, rows, cols, exit_width=2, x0_val=0.0, obstacles=None):
    interior_len = (cols - 2) if wall in {"top", "bottom"} else (rows - 2)
    max_pos = interior_len - exit_width
    x0_clamped = float(np.clip(x0_val, 0.0, float(max_pos)))

    def f_discrete(i: int) -> float:
        i_clamped = int(np.clip(i, 0, max_pos))
        test_grid = build_grid_with_exit_on_wall(
            wall=wall, position=i_clamped, rows=rows, cols=cols, exit_width=exit_width
        )
        if obstacles:
            for r, c in obstacles:
                test_grid[r, c] = 0  # Wall
        return worst_case_evacuation_time(test_grid)

    def continuous_objective(p: np.ndarray | float) -> float:
        p_val = float(np.asarray(p).item())
        p_clamped = float(np.clip(p_val, 0.0, float(max_pos)))
        i0 = int(np.floor(p_clamped))
        i1 = min(i0 + 1, max_pos)
        alpha = p_clamped - i0
        if i0 == i1 or np.isclose(alpha, 0.0):
            return f_discrete(i0)
        return (1.0 - alpha) * f_discrete(i0) + alpha * f_discrete(i1)

    def gradient_approx(p: np.ndarray) -> np.ndarray:
        p_val = float(np.asarray(p).item())
        if p_val >= max_pos - 0.5:
            grad = (continuous_objective(p_val) - continuous_objective(p_val - 1.0)) / 1.0
        elif p_val <= 0.5:
            grad = (continuous_objective(p_val + 1.0) - continuous_objective(p_val)) / 1.0
        else:
            grad = (continuous_objective(p_val + 0.5) - continuous_objective(p_val - 0.5)) / 1.0
        return np.array([grad])

    opt_res = minimize(
        fun=continuous_objective,
        x0=[x0_clamped],
        method="BFGS",
        jac=gradient_approx,
        options={"disp": False, "maxiter": 25},
    )

    raw_pos = float(opt_res.x[0])
    rounded_pos = int(np.clip(round(raw_pos), 0, max_pos))
    actual_j = f_discrete(rounded_pos)

    return {"raw_pos": raw_pos, "rounded_pos": rounded_pos, "actual_j": actual_j, "nit": opt_res.nit}

print("=== 1. Square Room (20x20) ===")
for seed in [0.0, 8.0, 16.0]:
    res = optimize_wall_test("right", 20, 20, 2, x0_val=seed)
    print(f"Seed {seed:4.1f} -> raw={res['raw_pos']:.3f}, rounded={res['rounded_pos']}, J={res['actual_j']:.1f}, nit={res['nit']}")

print("\n=== 2. Rectangular Room (15x30) ===")
print("Top wall (length 30):")
for seed in [0.0, 13.0, 26.0]:
    res = optimize_wall_test("top", 15, 30, 2, x0_val=seed)
    print(f"Seed {seed:4.1f} -> raw={res['raw_pos']:.3f}, rounded={res['rounded_pos']}, J={res['actual_j']:.1f}, nit={res['nit']}")

print("Left wall (length 15):")
for seed in [0.0, 5.5, 11.0]:
    res = optimize_wall_test("left", 15, 30, 2, x0_val=seed)
    print(f"Seed {seed:4.1f} -> raw={res['raw_pos']:.3f}, rounded={res['rounded_pos']}, J={res['actual_j']:.1f}, nit={res['nit']}")

print("\n=== 3. Off-Center Pillar (20x20) ===")
# Pillar shifted towards the right (rows 6..12, cols 12..16)
pillar_obs = [(r, c) for r in range(6, 13) for c in range(12, 17)]
for wall in ["top", "bottom", "left", "right"]:
    interior_len = (20 - 2)
    max_p = interior_len - 2
    res_start0 = optimize_wall_test(wall, 20, 20, 2, x0_val=0.0, obstacles=pillar_obs)
    res_mid = optimize_wall_test(wall, 20, 20, 2, x0_val=8.0, obstacles=pillar_obs)
    res_end = optimize_wall_test(wall, 20, 20, 2, x0_val=float(max_p), obstacles=pillar_obs)
    print(f"Wall {wall:<6}: seed 0 -> {res_start0['rounded_pos']} (J={res_start0['actual_j']:.1f}) | seed 8 -> {res_mid['rounded_pos']} (J={res_mid['actual_j']:.1f}) | seed 16 -> {res_end['rounded_pos']} (J={res_end['actual_j']:.1f})")
