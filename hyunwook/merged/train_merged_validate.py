"""병합 버전 검증 — 우리 파이프라인 + 팀원(ayeon) 파이프라인의 교차검증된
결론을 결합해서 다중 fold로 확인한다.

두 파이프라인을 비교해서 정리한 결합 설계:

[교차검증된 공통 결론 — 그대로 유지/최적화]
  - pitcher_id/batter_id 범주형 금지 (둘 다 실측/실험으로 확인된 실패)
  - Li 기반 위기상황 조건부 집계 피처 금지 (둘 다 실패)
  - 명시적 추세/레짐 보정(recency weighting, Platt/calibration, 시즌 센터링) 금지
    (둘 다 실패) — shrinkage(외삽이 아니라 표본 부족분을 당기는 것)만 유지

[서로 다른 강점 — 결합]
  - 아키텍처: 팀원 쪽의 R(1군)/F(퓨처스) 완전 분리 + game_type 라우팅을 채택
    (팀원 실측 LB 772.38로 검증됨). game_type을 피처로 넣지 않음(팀원이
    발견한 비정상성 붕괴 위험 회피).
  - F는 신체제(season>=2023) 데이터만 학습(팀원 발견: 구레짐 F를 섞으면
    분리해도 여전히 무너짐).
  - 베이스 모델: 우리 쪽 CatBoost(네이티브 범주형 처리)를 채택 — 우리 실험에서
    LightGBM/XGBoost/3-way 블렌딩보다 전부 앞섰음(train_v14_blend.py).
  - 피처: 우리 쪽 shrinkage(+23.84)/combo(+23.63)/season_trend(+47.20) 엔지니어링을
    유지하되, season_trend는 리그별로 따로 fit(우리 6.14/13단계 실험은 단일
    공유모델 안에서는 효과가 노이즈 수준이었지만, 이제 모델 자체가 분리되니
    자연스럽게 리그별 추세가 됨).
  - F는 표본이 작으므로(팀원 발견) 더 얕은 CatBoost(depth를 낮춤)를 사용.

검증: R은 val_season in {2022, 2024} 둘 다 테스트(데이터 충분).
F는 신체제 데이터가 2023~뿐이라 val=2024만 테스트 가능(train=2023만, ~2.5만행) —
팀원이 지적한 것과 동일한 구조적 한계.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    CatBoostWrapper,
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "../open/data"
ID = "row_id"
TARGET = "control_success"
K = 50

# game_type 제외(리그별로 분리하니 상수라 무의미 + 비정상성 회피),
# pitcher_id/batter_id 제외(교차검증된 실패)
BASE_CAT_COLS = ["top_bottom", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

R_PARAMS = dict(iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
                random_seed=42, loss_function="Logloss", eval_metric="Logloss",
                verbose=False, thread_count=-1)
# F는 표본이 작음(팀원 발견: 더 얕은 트리가 이김) -> depth/lr 보수적으로
F_PARAMS = dict(iterations=3000, learning_rate=0.02, depth=3, l2_leaf_reg=10.0,
                random_seed=42, loss_function="Logloss", eval_metric="Logloss",
                verbose=False, thread_count=-1)


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def fit_league_model(train, is_train_mask, is_val_mask, FEATURES, cat_cols, params):
    trend = fit_season_trend(train.loc[is_train_mask])
    sub = add_season_trend_feature(train, trend)

    X_train, y_train = sub.loc[is_train_mask, FEATURES], sub.loc[is_train_mask, TARGET]
    X_val, y_val = sub.loc[is_val_mask, FEATURES], sub.loc[is_val_mask, TARGET]
    num_cols = [c for c in FEATURES if c not in cat_cols]

    wrapper = CatBoostWrapper(cat_cols, num_cols, **params)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    pred = wrapper.predict_proba(X_val)[:, 1]
    return pred, y_val.to_numpy(), wrapper.best_iteration_


def run_fold(raw_train, prior_table, val_season, test_F=True):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET, "game_type")]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season
    is_R = train["game_type"] == "R"
    is_F = train["game_type"] == "F"
    is_F_newregime = train["season"] >= 2023

    t = time.time()
    pred_R, y_R, it_R = fit_league_model(train, is_train & is_R, is_val & is_R, FEATURES, CAT_COLS, R_PARAMS)
    print(f"    R모델 | best_iter={it_R:4d} | n_val={len(y_R)} | Score={score(pred_R, y_R):.2f} | {time.time()-t:.1f}s")

    if not test_F:
        return score(pred_R, y_R), None

    t = time.time()
    pred_F, y_F, it_F = fit_league_model(train, is_train & is_F & is_F_newregime, is_val & is_F,
                                          FEATURES, CAT_COLS, F_PARAMS)
    print(f"    F모델 | best_iter={it_F:4d} | n_val={len(y_F)} | Score={score(pred_F, y_F):.2f} | {time.time()-t:.1f}s")

    pred_all = np.concatenate([pred_R, pred_F])
    y_all = np.concatenate([y_R, y_F])
    s_combined = score(pred_all, y_all)
    print(f"    통합 Score={s_combined:.2f}")
    return s_combined, score(pred_F, y_F)


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table("../model/prior_table.json")

    print("=== val_season=2022 (F는 신체제 데이터 없어 R만 테스트) ===")
    s_2022, _ = run_fold(raw_train, prior_table, 2022, test_F=False)

    print("\n=== val_season=2024 (R+F 통합) ===")
    s_2024, s_2024_F = run_fold(raw_train, prior_table, 2024, test_F=True)

    print("\n\n=== 요약 ===")
    print(f"R단독 2022 = {s_2022:.2f}  (참고: 기존 A 공유모델 2022 = 2261.64)")
    print(f"통합 2024  = {s_2024:.2f}  (참고: 기존 A 공유모델 2024 = 685.86, CatBoost 단독 공유모델 2024 = 783.01)")
    print(f"F단독 2024 = {s_2024_F:.2f}")


if __name__ == "__main__":
    main()
