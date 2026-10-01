"""902.04 구성(train_v15_catboost.py)에 shrinkage K만 파라미터화한 최종 제출용 학습.
K_ADOPTED는 test_k_sweep.py(2022/2024 다중 fold, ±10 규칙)의 결론에 따라 결정한다.
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

K_ADOPTED = 50  # <-- test_k_sweep.py 결과 보고 50 또는 100으로 확정

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state",
                  "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=BASE_FEATURES + [TARGET])

    prior_table = load_prior_table()
    train = add_shrinkage_features(train, prior_table, k=K_ADOPTED)
    train = add_combo_features(train)
    train = add_season_trend_feature(train, prior_table["season_trend"])

    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    print(f"K_ADOPTED={K_ADOPTED} | train:", train.shape, "| 피처:", len(FEATURES))

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
    print(f"Validation Score(2024 holdout): {score:.2f} | 902점 구성(k=50) 대비 {score - 783.01:+.2f}")

    best_iter = wrapper.best_iteration_
    final_params = dict(CATBOOST_PARAMS)
    final_params["iterations"] = best_iter
    final_wrapper = CatBoostWrapper(CAT_COLS, [c for c in FEATURES if c not in CAT_COLS], **final_params)

    t = time.time()
    final_wrapper.fit(train[FEATURES], train[TARGET])
    print(f"전체 데이터 재학습 완료 (iterations={best_iter}) :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump({"model": final_wrapper, "features": FEATURES, "prior_table": prior_table,
                 "k_adopted": K_ADOPTED},
                "./model/catboost_final.pkl", compress=3)
    print("저장 완료: ./model/catboost_final.pkl")


if __name__ == "__main__":
    main()
