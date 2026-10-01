"""hyunwook 902.04 구성(단일 CatBoost)에서 shrinkage k만 50->100으로 바꿔
train_v14_blend.py와 동일한 다중 fold(2022/2024) 검증으로 diff를 확인한다.
CatBoost만 학습(LGBM/XGB는 시간 절약을 위해 생략, 이번 실험엔 불필요).
"""

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "./open/data"
ID = "row_id"
TARGET = "control_success"

VAL_SEASONS = [2022, 2024]
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def fit_catboost(X_train, y_train, X_val, y_val, cat_cols):
    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)
    clf = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
        random_seed=42, loss_function="Logloss", eval_metric="Logloss",
        early_stopping_rounds=100, verbose=False, thread_count=-1,
    )
    clf.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return clf.predict_proba(val_pool)[:, 1], clf.get_best_iteration()


def run_fold(raw_train, prior_table, val_season, k):
    train = add_shrinkage_features(raw_train, prior_table, k=k)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

    num_imputer_median = X_train[NUM_COLS].median()
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_train_cb[NUM_COLS] = X_train[NUM_COLS].fillna(num_imputer_median)
    X_val_cb[NUM_COLS] = X_val[NUM_COLS].fillna(num_imputer_median)
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    t = time.time()
    pred_cb, it_cb = fit_catboost(X_train_cb[FEATURES], y_train_arr, X_val_cb[FEATURES], y_val_arr, CAT_COLS)
    s = score(pred_cb, y_val_arr)
    print(f"    k={k:3d} val={val_season} | best_iter={it_cb:4d} | Score={s:8.2f} | {time.time()-t:.1f}s", flush=True)
    return s


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    results = {}
    for k in [50, 100]:
        print(f"=== k={k} ===", flush=True)
        results[k] = {}
        for val_season in VAL_SEASONS:
            results[k][val_season] = run_fold(raw_train, prior_table, val_season, k)
        results[k]["avg"] = np.mean(list(results[k].values()))

    print("\n=== 요약 ===")
    df = pd.DataFrame(results).T
    print(df)
    print(f"\nk=50->100 diff: 2022={results[100][2022]-results[50][2022]:+.2f} "
          f"2024={results[100][2024]-results[50][2024]:+.2f} "
          f"avg={results[100]['avg']-results[50]['avg']:+.2f}")


if __name__ == "__main__":
    main()
