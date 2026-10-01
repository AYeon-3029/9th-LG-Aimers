"""13단계 — season_trend를 game_type(R/F)별로 따로 fit해서 결합.

12단계(완전 분리 모델)는 표본 손실 부작용이 개선분보다 커서 평균 -5.26로
실패했다. 이번엔 표본을 쪼개지 않고 공유 모델은 그대로 둔 채, 진짜 결함
(R+F를 하나의 선형 추세로 뭉개는 season_trend_success_prior)만 정확히
고친다 — 리그별로 따로 추세선을 fit하고 각 행은 자기 game_type의 추세값을
쓰도록 season_trend_success_prior 계산 방식만 바꾼다. 다른 모든 피처/학습
데이터/하이퍼파라미터는 A와 동일.
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


def fit_season_trend_by_league(df, league_col="game_type", season_col="season", target_col="control_success"):
    """리그(game_type)별로 독립된 season_trend 계수를 fit."""
    return {league: fit_season_trend(g, season_col, target_col)
            for league, g in df.groupby(league_col)}


def add_season_trend_feature_by_league(df, trends, league_col="game_type", season_col="season"):
    df = df.copy()
    slope = df[league_col].map({lg: t["slope"] for lg, t in trends.items()})
    intercept = df[league_col].map({lg: t["intercept"] for lg, t in trends.items()})
    pred = slope * df[season_col] + intercept
    df["season_trend_success_prior"] = pred.clip(0, 1)
    return df


def run_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trends = fit_season_trend_by_league(train.loc[is_train])
    print(f"    리그별 추세: {trends}")
    train = add_season_trend_feature_by_league(train, trends)

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

    clf = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=50,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    clf.fit(Xtr_t, y_train, eval_X=Xval_t, eval_y=y_val, eval_metric="binary_logloss",
            callbacks=[lightgbm.early_stopping(100, verbose=False)])

    pred = clf.predict_proba(Xval_t)[:, 1]
    return score(pred, y_val), clf.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    results = {}
    for val_season in VAL_SEASONS:
        print(f"\n=== val_season={val_season} (리그별 season_trend) ===")
        t = time.time()
        s, it = run_fold(raw_train, prior_table, val_season)
        print(f"    best_iter={it} | Score={s:.2f} | {time.time()-t:.1f}s")
        results[val_season] = s

    avg = sum(results.values()) / len(results)
    print(f"\n\n=== 요약 ===")
    print(results)
    print(f"리그별 season_trend 평균: {avg:.2f}")
    print(f"(참고) 기존 A(단일 추세) 평균: 1473.75  (2022=2261.64, 2024=685.86)")


if __name__ == "__main__":
    main()
