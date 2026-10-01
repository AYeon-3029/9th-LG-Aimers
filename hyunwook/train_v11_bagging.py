"""11단계 — 부트스트랩 시드 앙상블(bagging)로 분산만 줄여본다.

지금까지 실패한 5가지(피처 추가, 범주형 인코딩, k튜닝, 정규화, recency
weighting)는 전부 "A 구성이 무엇을 배울지"를 바꾸는 시도였고, 로컬에서
좋아 보인 게 반복적으로 실제 성능은 깎았다(README 6.10~6.17). 이번엔
성격이 다르다 — A의 하이퍼파라미터/피처/학습 목표는 전혀 바꾸지 않고,
학습 데이터를 부트스트랩 재표본추출해 여러 개(random_state만 다름) 학습한
뒤 예측을 평균만 낸다. "무엇을 배우는지"는 그대로이고 개별 모델의
노이즈(분산)만 줄이는 방향이라, 지금까지의 실패 패턴과 리스크 성격이
다르다고 판단해 시도한다.

주의: subsample=1.0/colsample_bytree=1.0(A의 기존 설정)을 유지한 채
LightGBM 내부 random_state만 바꾸면 행 샘플링 자체가 없어 트리가 거의
동일하게 나올 수 있음 — 그래서 여기서는 학습 데이터를 직접 부트스트랩
재표본추출(복원추출, 같은 크기)해서 bag마다 실제로 다른 데이터를 보게
만든다. 검증(val)/평가는 항상 원본 val 세트 그대로 사용(부트스트랩 대상
아님 — 누출 없음).
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

N_BAGS_GRID = [1, 3, 5, 10]


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


def run_fold(raw_train, prior_table, val_season, max_bags):
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

    n = len(y_train)
    y_train_arr = y_train.to_numpy()
    y_val_arr = y_val.to_numpy()

    pred_sum = np.zeros(len(y_val_arr))
    results = {}
    for i in range(max_bags):
        seed = 42 + i
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, n, size=n)  # 복원추출(부트스트랩), 같은 크기
        t = time.time()
        clf = fit_one(Xtr_t[idx], y_train_arr[idx], Xval_t, y_val_arr, seed)
        pred_sum += clf.predict_proba(Xval_t)[:, 1]
        n_bags_so_far = i + 1
        if n_bags_so_far in N_BAGS_GRID:
            s = score(pred_sum / n_bags_so_far, y_val_arr)
            results[n_bags_so_far] = s
            print(f"    val={val_season} | n_bags={n_bags_so_far:2d} | Score={s:8.2f} | "
                  f"bag {i}: best_iter={clf.best_iteration_} {time.time()-t:.1f}s")
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
        all_results[val_season] = run_fold(raw_train, prior_table, val_season, max(N_BAGS_GRID))

    print("\n\n=== 전체 요약 (n_bags별 2022/2024 평균) ===")
    summary = {}
    for n_bags in N_BAGS_GRID:
        s2022 = all_results[2022][n_bags]
        s2024 = all_results[2024][n_bags]
        summary[n_bags] = {"2022": s2022, "2024": s2024, "avg": (s2022 + s2024) / 2}
    print(pd.DataFrame(summary).T)


if __name__ == "__main__":
    main()
