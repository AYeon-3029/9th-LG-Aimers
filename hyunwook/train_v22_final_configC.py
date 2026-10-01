"""22단계 — config C(depth=8, lr=0.015, l2=6.0) 최종 제출용 학습.

20/21단계에서 검증: 902점 구성(depth=6, lr=0.02, l2=3.0) 대비 2022/2024 두 fold
모두, 시드 3개씩 반복해도 완전히 분리되는 신호(gap/pooled_std=3.90배, A 최댓값 <
C 최솟값)로 이겼다. `train_final_catboost.py`(902.04를 만든 실제 프로덕션 스크립트)
와 완전히 동일한 절차 — CATBOOST_PARAMS만 config C로 교체, K_ADOPTED=50 등 나머지는
전부 그대로 유지(한 번에 한 변수만 바꾸는 원칙, [[lgaimers-validation-lessons]] 규칙
6). 902.04 모델(`model/catboost_final.pkl`)은 덮어쓰지 않고 별도 파일로 저장한다 —
비교/롤백을 위해 두 모델 다 보존.
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

K_ADOPTED = 50  # 902점 구성과 동일 — 이번 실험은 depth/lr/l2만 바꾼다

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state",
                  "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

# 20/21단계에서 검증된 config C
CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.015, depth=8, l2_leaf_reg=6.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

OUT_PATH = "./model/catboost_v22_configC.pkl"


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
                OUT_PATH, compress=3)
    print(f"저장 완료: {OUT_PATH}  (902.04 모델 model/catboost_final.pkl은 그대로 보존)")


if __name__ == "__main__":
    main()
