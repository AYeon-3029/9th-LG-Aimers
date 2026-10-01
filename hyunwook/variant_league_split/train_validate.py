"""variant B — 공유 모델을 버리고 R/F 완전 분리를 유지 (팀원 아키텍처 채택).

팀원 실측 기준: 이 아키텍처(R/F 분리 + game_type 라우팅, 4-way 앙상블)로
**실제 리더보드 약 800점대(정확히는 HANDOFF.md 기준 772.38)**를 받았다고
보고됨 — 로컬 테스트 시 이 숫자를 목표/참고 기준으로 삼는다. 단, 우리는
4-way(LGB+XGB+CAT+DCNv2) 대신 우리가 이미 검증한 **CatBoost 단독**을 각
리그의 베이스 모델로 쓴다(우리 실험에서 CatBoost 단독이 3-way 블렌딩보다도
좋았음 — train_v14_blend.py).

merged/ 폴더와의 차이: 여기에 팀원의 "직전 시즌 리그별 shrinkage prior"
(features.add_league_shrunk_feature)를 추가해서, R/F 분리 자체의 순수 효과에
팀원이 검증한 피처까지 더했을 때 merged/의 680.20(2024 통합)보다 나아지는지
확인한다.

F는 팀원 발견대로 신체제(season>=2023) 데이터만 학습 — 구레짐을 섞으면
분리해도 여전히 무너진다는 게 팀원 실험으로 확인됨.
"""

import time

import numpy as np
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

# game_type 제외(리그별 분리라 상수) + pitcher_id/batter_id 제외(교차검증된 실패)
BASE_CAT_COLS = ["top_bottom", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

R_PARAMS = dict(iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
                random_seed=42, loss_function="Logloss", eval_metric="Logloss",
                verbose=False, thread_count=-1)
# F는 표본이 작음(팀원 발견: 더 얕은 트리가 이김)
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

    season_means = compute_league_season_means(train.loc[is_train_mask])
    sub = add_league_shrunk_feature(sub, season_means)

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
    FEATURES = (BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES
                + TREND_FEATURE_NAMES + LEAGUE_SHRUNK_FEATURE_NAMES)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season
    is_R = train["game_type"] == "R"
    is_F = train["game_type"] == "F"
    is_F_newregime = train["season"] >= 2023

    t = time.time()
    pred_R, y_R, it_R = fit_league_model(train, is_train & is_R, is_val & is_R, FEATURES, CAT_COLS, R_PARAMS)
    print(f"    R모델 | best_iter={it_R:4d} | n_val={len(y_R)} | R단독Score={score(pred_R, y_R):.2f} | {time.time()-t:.1f}s")

    if not test_F:
        return None

    t = time.time()
    pred_F, y_F, it_F = fit_league_model(train, is_train & is_F & is_F_newregime, is_val & is_F,
                                          FEATURES, CAT_COLS, F_PARAMS)
    print(f"    F모델 | best_iter={it_F:4d} | n_val={len(y_F)} | F단독Score={score(pred_F, y_F):.2f} | {time.time()-t:.1f}s")

    pred_all = np.concatenate([pred_R, pred_F])
    y_all = np.concatenate([y_R, y_F])
    s_combined = score(pred_all, y_all)
    print(f"    통합 Score={s_combined:.2f}")
    return s_combined


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table("../model/prior_table.json")

    print("=== val_season=2022 (F는 신체제 데이터 없어 R만 테스트) ===")
    run_fold(raw_train, prior_table, 2022, test_F=False)

    print("\n=== val_season=2024 (R+F 통합) ===")
    s_2024 = run_fold(raw_train, prior_table, 2024, test_F=True)

    print("\n\n=== 요약 ===")
    print(f"2024 통합 Score = {s_2024:.2f}")
    print(f"참고: merged/(이 피처 추가 전) 2024 통합 = 680.20")
    print(f"참고: 공유 CatBoost(분리 없음) 2024 = 783.01")
    print(f"참고: 팀원 실측 LB(4-way 앙상블 기준) = 772.38 (~800으로 보고됨)")


if __name__ == "__main__":
    main()
