"""CatBoost 원시 범주형 추가 실험 (k=50 고정, 2022/2024 다중 fold).

주의: base_state/top_bottom은 이미 BASE_CAT_COLS에 원본 범주형으로 들어가 있음
(train_v15_catboost.py 확인) - 새로 추가해볼 후보는 pitcher_hand/batter_hand뿐.
한 번에 하나씩만 추가해서 diff를 본다.
"""

import sys
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
K = 50

VAL_SEASONS = [2022, 2024]
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]


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


def run_fold(raw_train, prior_table, val_season, cat_cols):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in cat_cols]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

    med = X_train[NUM_COLS].median()
    X_train_cb, X_val_cb = X_train.copy(), X_val.copy()
    X_train_cb[NUM_COLS] = X_train[NUM_COLS].fillna(med)
    X_val_cb[NUM_COLS] = X_val[NUM_COLS].fillna(med)
    for c in cat_cols:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    y_train_arr, y_val_arr = y_train.to_numpy(), y_val.to_numpy()

    t = time.time()
    pred_cb, it_cb = fit_catboost(X_train_cb[FEATURES], y_train_arr, X_val_cb[FEATURES], y_val_arr, cat_cols)
    s = score(pred_cb, y_val_arr)
    print(f"    val={val_season} | best_iter={it_cb:4d} | Score={s:8.2f} | {time.time()-t:.1f}s", flush=True)
    return s


def main():
    extra_col = sys.argv[1] if len(sys.argv) > 1 else "pitcher_hand"
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    configs = {
        "baseline": BASE_CAT_COLS + CATEGORICAL_COMBO_COLS,
        f"+{extra_col}": BASE_CAT_COLS + CATEGORICAL_COMBO_COLS + [extra_col],
    }

    results = {}
    for name, cat_cols in configs.items():
        print(f"=== {name} (cat_cols={cat_cols}) ===", flush=True)
        results[name] = {}
        for val_season in VAL_SEASONS:
            results[name][val_season] = run_fold(raw_train, prior_table, val_season, cat_cols)
        results[name]["avg"] = np.mean(list(results[name].values()))

    print("\n=== 요약 ===")
    df = pd.DataFrame(results).T
    print(df)
    base, new = "baseline", f"+{extra_col}"
    print(f"\n{base}->{new} diff: 2022={results[new][2022]-results[base][2022]:+.2f} "
          f"2024={results[new][2024]-results[base][2024]:+.2f} "
          f"avg={results[new]['avg']-results[base]['avg']:+.2f}")


if __name__ == "__main__":
    main()
