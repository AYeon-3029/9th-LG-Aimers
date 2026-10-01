"""10단계 — 최근 시즌 가중치(recency weighting)로 오래된(pre-ABS) 시즌의
영향력을 줄여본다.

지금까지 시도(피처 추가/범주형 인코딩/k튜닝/정규화, 4회 실제 제출)는 전부
"2019~2024 안에서 더 잘 맞추는" 방향이었는데 실제 2025 리더보드에는 반복적으로
반대로 작용했다(README 6.10~6.13 + 정규화 제출 결과). 이번엔 방향을 바꿔서
"무엇을 더 배울지" 자체를 조정한다 — sample_weight로 타깃 시즌에 가까운
시즌일수록 더 큰 가중치를 줘서, 2019~2020(ABS 도입 전) 등 오래된 분포가
모델을 덜 지배하게 만든다.

weight(season) = decay ** (target_season - season)   (decay=1.0 == 가중치 없음)

backtest에서는 target_season = val_season으로 두고(그 시점 기준 "최근"이 뭔지는
walk-forward라 val_season 이전 데이터만 쓰는 것과 일관됨), decay를 격자 탐색해서
어떤 decay가 2022/2024 fold 평균을 개선하는지 본다. 실제 프로덕션 모델은
target_season=2025로 재학습한다.
"""

import time

import lightgbm
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

DECAY_GRID = [1.0, 0.9, 0.7, 0.5, 0.3]


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, decay):
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
    w_train = decay ** (val_season - train.loc[is_train, "season"])

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])
    pre.fit(X_train)
    Xtr_t = pre.transform(X_train)
    Xval_t = pre.transform(X_val)

    clf = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=50,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr_t, y_train, sample_weight=w_train, eval_X=Xval_t, eval_y=y_val,
            eval_metric="binary_logloss", callbacks=[lightgbm.early_stopping(100, verbose=False)])

    pred = clf.predict_proba(Xval_t)[:, 1]
    return score(pred, y_val), clf.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    print("season 분포:")
    print(raw_train["season"].value_counts().sort_index())

    results = {}
    for decay in DECAY_GRID:
        print(f"\n=== decay={decay} ===")
        scores = {}
        for val_season in VAL_SEASONS:
            t = time.time()
            s, it = run_fold(raw_train, prior_table, val_season, decay)
            scores[val_season] = s
            print(f"  val={val_season} | best_iter={it:4d} | Score={s:8.2f} | {time.time()-t:.1f}s")
        avg = sum(scores.values()) / len(scores)
        print(f"  → 평균: {avg:.2f}")
        results[f"decay={decay}"] = {**scores, "avg": avg}

    print("\n\n=== 전체 요약 ===")
    print(pd.DataFrame(results).T)


if __name__ == "__main__":
    main()
