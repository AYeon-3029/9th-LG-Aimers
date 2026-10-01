"""11-b단계 — 부트스트랩 없이 순수 random_state만 다른 시드 앙상블.

11단계(train_v11_bagging.py)의 부트스트랩 복원추출은 매번 원본의 ~63%만
쓰게 되어 개별 모델이 약해지는 부작용이 앙상블 이득을 덮어버렸다(2024
674.89 < 원본 685.86). 이번엔 데이터 손실 없이 전체 학습 데이터를 그대로
쓰고 LightGBM `random_state`만 바꿔 N개 학습 → 예측 평균. subsample=1.0/
colsample_bytree=1.0이라 행/열 서브샘플링에 의한 다양성은 없지만, 트리
분기 시 동점 처리·초기화 등에서 약간의 다양성은 있을 수 있어 실제로
얼마나 도움이 되는지(혹은 안 되는지) 확인한다.
"""

import time

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

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

N_SEEDS_GRID = [1, 3, 5]


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def fit_one(Xtr_t, y_train, Xval_t, y_val, seed):
    clf = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=50,
        subsample=1.0, colsample_bytree=1.0, random_state=seed, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr_t, y_train, eval_X=Xval_t, eval_y=y_val, eval_metric="binary_logloss",
            callbacks=[lightgbm.early_stopping(100, verbose=False)])
    return clf


def run_fold(raw_train, prior_table, val_season, max_seeds):
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

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])
    pre.fit(X_train)
    Xtr_t = pre.transform(X_train)
    Xval_t = pre.transform(X_val)

    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    pred_sum = np.zeros(len(y_val_arr))
    results = {}
    for i in range(max_seeds):
        seed = 42 + i
        t = time.time()
        clf = fit_one(Xtr_t, y_train_arr, Xval_t, y_val_arr, seed)
        pred_sum += clf.predict_proba(Xval_t)[:, 1]
        n_so_far = i + 1
        if n_so_far in N_SEEDS_GRID:
            s = score(pred_sum / n_so_far, y_val_arr)
            results[n_so_far] = s
            print(f"    val={val_season} | n_seeds={n_so_far:2d} | Score={s:8.2f} | "
                  f"seed={seed}: best_iter={clf.best_iteration_} {time.time()-t:.1f}s")
    return results


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    all_results = {}
    for val_season in VAL_SEASONS:
        print(f"\n=== val_season={val_season} ===")
        all_results[val_season] = run_fold(raw_train, prior_table, val_season, max(N_SEEDS_GRID))

    print("\n\n=== 전체 요약 (n_seeds별 2022/2024 평균) ===")
    summary = {}
    for n_seeds in N_SEEDS_GRID:
        s2022 = all_results[2022][n_seeds]
        s2024 = all_results[2024][n_seeds]
        summary[n_seeds] = {"2022": s2022, "2024": s2024, "avg": (s2022 + s2024) / 2}
    print(pd.DataFrame(summary).T)
    print("\n(참고) 원본 A(단일 시드, train_v6 기준): 2022=2261.64, 2024=685.86, avg=1473.75")


if __name__ == "__main__":
    main()
