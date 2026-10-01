"""variant A — R/F 완전 분리를 버리고 공유 모델을 유지.

merged/ 실험에서 확인: CatBoost의 강점(game_type을 다른 범주형과 교차 조합 +
R+F 합친 큰 학습 데이터)은 정확히 R/F 분리가 없애는 것과 같아서, 분리하면
오히려 손해였다(2024 통합 680.20 < 공유 CatBoost 783.01). 그래서 이 버전은
**공유 모델 구조를 그대로 유지**하고(`game_type`도 계속 범주형 피처로 사용),
팀원 파이프라인에서 구조 충돌 없이 가져올 수 있는 것만 추가한다 —
"직전 시즌 리그별 shrinkage prior"(features.add_league_shrunk_feature)를
기존 shrinkage/combo/season_trend 피처 옆에 추가 피처로만 얹는다.

로컬 참고 기준선(이미 측정됨):
  - 공유 LightGBM(A, 정규화/추가피처 없음): 2022=2261.64, 2024=685.86
  - 공유 CatBoost(이 피처 추가 전, train_v14_blend.py): 2022=2347.67, 2024=783.01
    (단일 2024 분할 기준으로는 train_v15_catboost.py에서 783.01 재확인,
    실제 제출은 아직 안 됨 — 내일 제출 예정)
이 스크립트의 목표는 위 CatBoost 기준선(783.01/2347.67)을
league_season_shrunk_pitcher_rate 피처 추가로 더 개선할 수 있는지 확인하는 것.
"""

import time
from pathlib import Path

import pandas as pd

from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    LEAGUE_SHRUNK_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    CatBoostWrapper,
    add_combo_features,
    add_league_shrunk_feature,
    add_season_trend_feature,
    add_shrinkage_features,
    compute_league_season_means,
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "../open/data"
ID = "row_id"
TARGET = "control_success"
K = 50

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    season_means = compute_league_season_means(train.loc[is_train])
    train = add_league_shrunk_feature(train, season_means)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = (BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES
                + TREND_FEATURE_NAMES + LEAGUE_SHRUNK_FEATURE_NAMES)
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

    wrapper = CatBoostWrapper(CAT_COLS, NUM_COLS, **CATBOOST_PARAMS)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    pred = wrapper.predict_proba(X_val)[:, 1]
    return score(pred, y_val.to_numpy()), wrapper.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table("../model/prior_table.json")

    results = {}
    for val_season in [2022, 2024]:
        print(f"\n=== val_season={val_season} ===")
        t = time.time()
        s, it = run_fold(raw_train, prior_table, val_season)
        print(f"  best_iter={it} | Score={s:.2f} | {time.time()-t:.1f}s")
        results[val_season] = s

    print("\n\n=== 요약 ===")
    print(results)
    print(f"참고: 공유 CatBoost(이 피처 없이) 2022=2347.67, 2024=783.01")
    print(f"참고: 공유 LightGBM(A) 2022=2261.64, 2024=685.86")


if __name__ == "__main__":
    main()
