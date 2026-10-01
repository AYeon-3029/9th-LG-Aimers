"""15단계 — CatBoost(네이티브 범주형 처리) 제출용 모델 학습.

train_v14_blend.py에서 확인: 동일 82피처(팀ID만 범주형, k=50)로 CatBoost가
LightGBM보다 다중 fold 평균 +91.6(1473.75 -> 1565.34) 앞섬 — 지금까지 실패한
튜닝들(+5~9 수준)과 자릿수가 다른 개선. features.CatBoostWrapper로 학습/추론
인터페이스(model.predict_proba(X), X는 FEATURES 컬럼의 원본 DataFrame)를
LightGBM 파이프라인과 동일하게 유지한다.

train_v4_lgbm.py와 동일한 절차: 2024 홀드아웃으로 early stopping → best
iteration 확인 → 전체 데이터(2019~2024)로 고정 iteration 재학습 → 저장.
"""

import os
import time

import joblib
import pandas as pd

from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    CatBoostWrapper,
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
    load_prior_table,
)

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"

# train_v4_lgbm.py와 동일 — 팀ID만 범주형, k=50 (현재 유일하게 실제 검증된 구성)
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state",
                  "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

CATBOOST_PARAMS = dict(
    iterations=3000,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    loss_function="Logloss",
    eval_metric="Logloss",
    verbose=False,
    thread_count=-1,
)


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=BASE_FEATURES + [TARGET])

    prior_table = load_prior_table()
    train = add_shrinkage_features(train, prior_table)
    train = add_combo_features(train)
    train = add_season_trend_feature(train, prior_table["season_trend"])

    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES

    print("train:", train.shape, "| 피처:", len(FEATURES))

    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    t = time.time()
    wrapper = CatBoostWrapper(CAT_COLS, [c for c in FEATURES if c not in CAT_COLS], **CATBOOST_PARAMS)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    print(f"학습 완료 :: {time.time() - t:.1f}s (best_iteration={wrapper.best_iteration_})")

    val_pred = wrapper.predict_proba(X_val)[:, 1]
    r = y_val.mean()
    brier = ((val_pred - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))

    print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"  현재 채택(LightGBM, 685.86) 대비 {score - 685.86:+.2f}")

    # ---- 전체 데이터(2019~2024)로 재학습 & 저장 ----
    best_iter = wrapper.best_iteration_
    final_params = dict(CATBOOST_PARAMS)
    final_params["iterations"] = best_iter
    final_wrapper = CatBoostWrapper(CAT_COLS, [c for c in FEATURES if c not in CAT_COLS], **final_params)

    t = time.time()
    final_wrapper.fit(train[FEATURES], train[TARGET])
    print(f"\n전체 데이터 재학습 완료 (iterations={best_iter}) :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump({"model": final_wrapper, "features": FEATURES, "prior_table": prior_table},
                "./model/catboost_v1.pkl", compress=3)
    print("저장 완료: ./model/catboost_v1.pkl")


if __name__ == "__main__":
    main()
