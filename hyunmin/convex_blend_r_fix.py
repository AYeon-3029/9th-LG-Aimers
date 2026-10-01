"""R의 볼록결합 가중치가 3구간 전부 정확히 0.500/0.500(초기값과 동일)으로 나온 게
SLSQP가 실제로는 수렴하지 못하고 x0를 그대로 반환한 것인지 확인하고, 1차원
스칼라 최적화(minimize_scalar, bounds=(0,1))로 재검증한다 - 2변수+등식제약 SLSQP
정식화를 완전히 피해서 대조한다. 이미 캐싱된 예측값을 재사용(재학습 불필요)."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar
import sys
sys.path.insert(0, "../LGAimers")
from common import brier_skill_score

cache = np.load("convex_blend_cache.npz")


def objective_2d(w, p1, p2, y):
    p = w[0] * p1 + w[1] * p2
    return np.mean((p - y) ** 2)


def objective_1d(w1, p1, p2, y):
    p = w1 * p1 + (1 - w1) * p2
    return np.mean((p - y) ** 2)


for split_i in range(3):
    y = cache[f"R_{split_i}_y"]
    p_tm = cache[f"R_{split_i}_p_tm"]
    p_ours = cache[f"R_{split_i}_p_ours"]

    # 원래 SLSQP 재현 - 진단 정보(success/message) 같이 출력
    res_slsqp = minimize(
        objective_2d, x0=[0.5, 0.5], args=(p_tm, p_ours, y),
        bounds=[(0, 1), (0, 1)], method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    # 1차원 스칼라 최적화로 대조 검증
    res_scalar = minimize_scalar(objective_1d, args=(p_tm, p_ours, y), bounds=(0, 1), method="bounded")
    w1_scalar = res_scalar.x

    _, bss_ours = brier_skill_score(y, p_ours)
    _, bss_scalar_opt = brier_skill_score(y, w1_scalar * p_tm + (1 - w1_scalar) * p_ours)

    print(f"[split {split_i}]")
    print(f"  SLSQP: w={res_slsqp.x}  success={res_slsqp.success}  message={res_slsqp.message}  nit={res_slsqp.nit}")
    print(f"  scalar(bounded): w_teammate={w1_scalar:.4f}  w_ours={1-w1_scalar:.4f}  "
          f"bss(ours solo)={bss_ours:,.0f}  bss(scalar-optimal blend)={bss_scalar_opt:,.0f}  delta={bss_scalar_opt-bss_ours:+.0f}")

    # grid search로 한 번 더 확인(완전히 다른 방법)
    grid = np.linspace(0, 1, 101)
    losses = [objective_1d(w1, p_tm, p_ours, y) for w1 in grid]
    best_grid_w1 = grid[np.argmin(losses)]
    _, bss_grid_opt = brier_skill_score(y, best_grid_w1 * p_tm + (1 - best_grid_w1) * p_ours)
    print(f"  grid search: w_teammate={best_grid_w1:.4f}  bss={bss_grid_opt:,.0f}")
    print()
