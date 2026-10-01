"""14단계 — LightGBM + XGBoost + CatBoost 균등 가중 블렌딩.

개별로는 XGBoost(657.00)/CatBoost(639.22)가 LightGBM(685.86)보다 약하지만,
오차 패턴이 충분히 다르면 블렌딩 시 단일 최고 모델보다 나아질 수 있다는
아이디어(공유받은 다른 팀 사례, LGBM+XGB+CatBoost 블렌딩으로 개선)를 그대로
검증한다. 세 모델 모두 동일한 82피처(팀ID만 범주형, k=50, A 구성)로 학습.
"""

import time

import lightgbm
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

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
K = 50
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def fit_lgbm(Xtr_t, y_train, Xval_t, y_val):
    clf = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=50,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr_t, y_train, eval_X=Xval_t, eval_y=y_val, eval_metric="binary_logloss",
            callbacks=[lightgbm.early_stopping(100, verbose=False)])
    return clf.predict_proba(Xval_t)[:, 1], clf.best_iteration_


def fit_xgb(Xtr_t, y_train, Xval_t, y_val):
    # train_v7_xgboost.py에서 가장 성능이 좋았던 설정(657.00)
    clf = XGBClassifier(
        n_estimators=3000, learning_rate=0.02, max_leaves=63, grow_policy="lossguide",
        min_child_weight=50, subsample=1.0, colsample_bytree=1.0, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=100, random_state=42, n_jobs=-1,
    )
    clf.fit(Xtr_t, y_train, eval_set=[(Xval_t, y_val)], verbose=False)
    return clf.predict_proba(Xval_t)[:, 1], clf.best_iteration


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


def run_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
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

    # LightGBM/XGBoost용: 범주형 OrdinalEncoder + 수치형 median impute
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])
    pre.fit(X_train)
    Xtr_t = pre.transform(X_train)
    Xval_t = pre.transform(X_val)

    # CatBoost용: 범주형은 문자열 그대로(네이티브 처리), 수치형만 동일하게 median impute
    num_imputer = SimpleImputer(strategy="median")
    num_imputer.fit(X_train[NUM_COLS])
    X_train_cb = X_train.copy()
    X_val_cb = X_val.copy()
    X_train_cb[NUM_COLS] = num_imputer.transform(X_train[NUM_COLS])
    X_val_cb[NUM_COLS] = num_imputer.transform(X_val[NUM_COLS])
    for c in CAT_COLS:
        X_train_cb[c] = X_train_cb[c].astype(str)
        X_val_cb[c] = X_val_cb[c].astype(str)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    t = time.time()
    pred_lgbm, it_lgbm = fit_lgbm(Xtr_t, y_train_arr, Xval_t, y_val_arr)
    print(f"    LightGBM | best_iter={it_lgbm:4d} | Score={score(pred_lgbm, y_val_arr):8.2f} | {time.time()-t:.1f}s")

    t = time.time()
    pred_xgb, it_xgb = fit_xgb(Xtr_t, y_train_arr, Xval_t, y_val_arr)
    print(f"    XGBoost  | best_iter={it_xgb:4d} | Score={score(pred_xgb, y_val_arr):8.2f} | {time.time()-t:.1f}s")

    t = time.time()
    pred_cb, it_cb = fit_catboost(X_train_cb[FEATURES], y_train_arr, X_val_cb[FEATURES], y_val_arr, CAT_COLS)
    print(f"    CatBoost | best_iter={it_cb:4d} | Score={score(pred_cb, y_val_arr):8.2f} | {time.time()-t:.1f}s")

    blend_all3 = (pred_lgbm + pred_xgb + pred_cb) / 3
    blend_lgbm_xgb = (pred_lgbm + pred_xgb) / 2
    blend_lgbm_cb = (pred_lgbm + pred_cb) / 2
    blend_weighted = 0.6 * pred_lgbm + 0.2 * pred_xgb + 0.2 * pred_cb

    scores = {
        "lgbm_only": score(pred_lgbm, y_val_arr),
        "blend_equal_3": score(blend_all3, y_val_arr),
        "blend_lgbm_xgb": score(blend_lgbm_xgb, y_val_arr),
        "blend_lgbm_cb": score(blend_lgbm_cb, y_val_arr),
        "blend_weighted_6_2_2": score(blend_weighted, y_val_arr),
    }
    for name, s in scores.items():
        print(f"      {name:22s} = {s:.2f}")
    return scores


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    all_results = {}
    for val_season in VAL_SEASONS:
        print(f"\n=== val_season={val_season} ===")
        all_results[val_season] = run_fold(raw_train, prior_table, val_season)

    print("\n\n=== 요약 (2022/2024 평균) ===")
    df = pd.DataFrame(all_results).T
    df.loc["avg"] = df.mean()
    print(df)


if __name__ == "__main__":
    main()
