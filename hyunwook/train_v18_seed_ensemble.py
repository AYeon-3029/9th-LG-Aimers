"""18단계 — CatBoost 시드 앙상블 (902점 구성 대비 단일 변수 변경).

배경: 17단계(`train_v17_ctr_complexity.py`)에서 발견한 것 — 완전히 동일한 코드·설정·
random_seed=42로 두 번 돌렸는데 2024 fold 점수가 783.01(6.20절 기록) vs 776.56(이번
실행)으로 6.45점 차이가 났다. `thread_count=-1` 병렬 처리로 인한 실행 간 비결정성으로
추정된다. 이 자체가 "시드/실행 노이즈를 평균으로 줄이면 도움이 될 수 있다"는 가설을
뒷받침한다 — capacity 추가가 아니라 순수 분산 감소이므로 리스크가 가장 낮은 실험이다.

11-b단계(`train_v11b_seed_ensemble.py`)에서 LightGBM으로 같은 아이디어를 시도했지만
CatBoost 전환 전이라 결과가 README에 기록되지 않았다 — 902점 구성으로는 처음 검증.

방법: `random_seed`만 42, 43, 44, 45, 46으로 바꿔 독립적으로 N개 학습 후
predict_proba를 평균. n=1(순수 baseline 재현, 노이즈 크기 확인용), n=3, n=5를 비교.

검증 프로토콜은 6.14/6.16/6.20/17단계와 동일: val_season ∈ {2022, 2024}, 두 fold 모두
같은 방향이어야 신호로 인정. 채택 기준(6.18): ±10 수준 등락은 노이즈로 기각.
"""

import time

import numpy as np
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
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "./open/data"
ID = "row_id"
TARGET = "control_success"
K = 50

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

BASE_SEED = 42
N_SEEDS_GRID = [1, 3, 5]

CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def prep_fold(raw_train, prior_table, val_season):
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
    return X_train, y_train, X_val, y_val, NUM_COLS


def fit_one_seed(X_train, y_train, X_val, y_val, num_cols, seed):
    params = dict(CATBOOST_PARAMS, random_seed=seed)
    wrapper = CatBoostWrapper(CAT_COLS, num_cols, **params)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    return wrapper.predict_proba(X_val)[:, 1], wrapper.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    max_n = max(N_SEEDS_GRID)
    results = {n: {} for n in N_SEEDS_GRID}

    for val_season in [2022, 2024]:
        X_train, y_train, X_val, y_val, num_cols = prep_fold(raw_train, prior_table, val_season)
        y_val_np = y_val.to_numpy()

        seed_preds = []
        for i in range(max_n):
            seed = BASE_SEED + i
            t = time.time()
            pred, best_iter = fit_one_seed(X_train, y_train, X_val, y_val, num_cols, seed)
            seed_preds.append(pred)
            print(f"  val={val_season} | seed={seed} | best_iter={best_iter:4d} | "
                  f"solo Score={score(pred, y_val_np):8.2f} | {time.time()-t:.0f}s")

        for n in N_SEEDS_GRID:
            avg_pred = np.mean(seed_preds[:n], axis=0)
            results[n][val_season] = score(avg_pred, y_val_np)

    print("\n\n=== 요약 (baseline = n_seeds=1, 즉 현재 902점 프로덕션과 동일) ===")
    b = results[1]
    for n in N_SEEDS_GRID:
        s = results[n]
        avg = (s[2022] + s[2024]) / 2
        d2022 = s[2022] - b[2022]
        d2024 = s[2024] - b[2024]
        print(f"  n_seeds={n} | 2022={s[2022]:8.2f} ({d2022:+7.2f}) | "
              f"2024={s[2024]:8.2f} ({d2024:+7.2f}) | 평균={avg:8.2f}")


if __name__ == "__main__":
    main()
