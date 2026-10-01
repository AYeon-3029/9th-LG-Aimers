"""21단계 — config C(depth=8, lr=0.015, l2=6.0)의 신호가 노이즈보다 큰지 확인.

배경: 20단계에서 config C가 baseline(A, 902점 구성) 대비 두 fold 모두에서
이겼다(2022 +18.45, 2024 +5.14, 평균 +11.8) — 오늘 나온 것 중 가장 크고 일관된
신호. 그런데 17단계에서 완전히 동일한 코드·설정·seed=42를 두 번 돌렸을 때만도
2024 fold 점수가 6.45점 차이 났다(`thread_count=-1` 병렬 비결정성 추정). 즉 "한 번
비교"로는 신호(+11.8)가 노이즈(~6.45)보다 큰지 확신할 수 없다.

방법: A와 C 각각을 random_seed ∈ {42, 43, 44}로 3번씩 독립 학습(순수 반복 측정,
앙상블 평균 아님 — solo 점수의 분포/산포를 보는 것) → 각 config의 평균과
표준편차를 계산해서, "C의 평균 - A의 평균"이 두 config 각각의 seed-to-seed 변동폭
대비 충분히 큰지 판단한다. 두 fold(2022, 2024) 모두에서 같은 결론이 나와야
신뢰한다.
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

COMMON_PARAMS = dict(
    iterations=3000, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

CONFIGS = {
    "A(902점 baseline)": dict(depth=6, learning_rate=0.02, l2_leaf_reg=3.0),
    "C(depth8/lr.015/l2=6)": dict(depth=8, learning_rate=0.015, l2_leaf_reg=6.0),
}

SEEDS = [42, 43, 44]


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


def fit_one(X_train, y_train, X_val, y_val, num_cols, extra_params, seed):
    params = dict(COMMON_PARAMS, random_seed=seed, **extra_params)
    wrapper = CatBoostWrapper(CAT_COLS, num_cols, **params)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    return wrapper.predict_proba(X_val)[:, 1], wrapper.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    all_scores = {label: {vs: [] for vs in [2022, 2024]} for label in CONFIGS}

    for val_season in [2022, 2024]:
        X_train, y_train, X_val, y_val, num_cols = prep_fold(raw_train, prior_table, val_season)
        y_val_np = y_val.to_numpy()

        for label, extra_params in CONFIGS.items():
            for seed in SEEDS:
                t = time.time()
                pred, best_iter = fit_one(X_train, y_train, X_val, y_val, num_cols, extra_params, seed)
                s = score(pred, y_val_np)
                all_scores[label][val_season].append(s)
                print(f"  val={val_season} | {label:25s} | seed={seed} | best_iter={best_iter:4d} | "
                      f"Score={s:8.2f} | {time.time()-t:.0f}s")

    print("\n\n=== 시드 간 분포 (n=3) ===")
    for label in CONFIGS:
        for vs in [2022, 2024]:
            vals = all_scores[label][vs]
            print(f"  {label:25s} | val={vs} | scores={[round(v,2) for v in vals]} | "
                  f"평균={np.mean(vals):8.2f} | std={np.std(vals):6.2f} | range={max(vals)-min(vals):6.2f}")

    print("\n=== A vs C 평균 비교 (신호 vs 노이즈) ===")
    for vs in [2022, 2024]:
        a_vals = all_scores["A(902점 baseline)"][vs]
        c_vals = all_scores["C(depth8/lr.015/l2=6)"][vs]
        a_mean, c_mean = np.mean(a_vals), np.mean(c_vals)
        a_std, c_std = np.std(a_vals), np.std(c_vals)
        gap = c_mean - a_mean
        pooled_std = np.sqrt((a_std**2 + c_std**2) / 2)
        print(f"  val={vs} | A 평균={a_mean:8.2f}(std={a_std:5.2f}) | C 평균={c_mean:8.2f}(std={c_std:5.2f}) | "
              f"C-A={gap:+7.2f} | gap/pooled_std={gap/pooled_std if pooled_std > 0 else float('nan'):.2f}배")


if __name__ == "__main__":
    main()
