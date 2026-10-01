"""7단계 — LightGBM을 walk-forward 다중 fold로 재검증.

기각했던 두 변경(shrinkage k=85, pitcher_id/batter_id 범주형 전환)이 단일 2024
분할에서만 좋아 보인 노이즈였는지, 아니면 다중 fold에서도 일관되게 도움이 되는
진짜 신호인지 확인한다. val_season마다 그 시점 이전 데이터로만 학습(walk-forward)
하고, season_trend도 매 fold마다 새로 fit해서 누출을 방지한다.

2023 fold는 이미 알려진 대로(README 6.5절, ABS 구조적 단절) 어떤 설정이든
0점에 가깝게 나오는 병적인 케이스라 — 참고용으로만 보고, 2022/2024 평균을
주된 판단 기준으로 삼는다.
"""

import time

import lightgbm
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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

VAL_SEASONS = [2022, 2023, 2024]

CONFIGS = {
    "A. 팀ID만 범주형, k=50 (현재 채택)": dict(extra_cat=[], k=50),
    "B. +선수ID 범주형, k=50": dict(extra_cat=["pitcher_id", "batter_id"], k=50),
    "C. 팀ID만 범주형, k=85": dict(extra_cat=[], k=85),
    "D. +선수ID 범주형, k=85": dict(extra_cat=["pitcher_id", "batter_id"], k=85),
}


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, extra_cat, k):
    train = add_shrinkage_features(raw_train, prior_table, k=k)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"] \
        + extra_cat + CATEGORICAL_COMBO_COLS
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
    for label, cfg in CONFIGS.items():
        print(f"\n=== {label} ===")
        scores = {}
        for val_season in VAL_SEASONS:
            t = time.time()
            s, best_iter = run_fold(raw_train, prior_table, val_season, cfg["extra_cat"], cfg["k"])
            scores[val_season] = s
            print(f"  val={val_season} | best_iter={best_iter:4d} | Score={s:8.2f} | {time.time()-t:.1f}s")
        avg_2022_2024 = (scores[2022] + scores[2024]) / 2
        print(f"  → 2022/2024 평균(2023 제외): {avg_2022_2024:.2f}")
        results[label] = {**scores, "avg_2022_2024": avg_2022_2024}

    print("\n\n=== 전체 요약 ===")
    summary = pd.DataFrame(results).T
    print(summary)


if __name__ == "__main__":
    main()
