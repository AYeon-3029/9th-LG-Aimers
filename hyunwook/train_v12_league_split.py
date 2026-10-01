"""12단계 — R(1군)/F(퓨처스) 완전 분리 모델.

검증된 근거: F는 2022(70.9%)->2023(47.3%) 사이에 R과 전혀 다른 급격한
반전이 있는데(R은 완만한 하락뿐), 현재 파이프라인은 season_trend를
R+F를 합쳐 단일 선형 추세 하나로 fit하고 있어(features.fit_season_trend)
이 반전을 제대로 못 담는다. game_type별로 완전히 독립된 모델(피처는 동일,
학습 데이터와 season_trend만 그 리그로 한정)을 학습해서 개선 여부를 확인한다.

지금까지 실패한 5가지(피처 추가/범주형 인코딩/k튜닝/정규화/recency
weighting)와 달리 "기존 단일 모델을 더 잘 맞추기"가 아니라 아예 다른
아키텍처(리그별 라우팅)라는 점에서 실패 패턴이 다를 것으로 기대.
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
# game_type은 리그별로 분리 학습하므로 각 서브모델 안에서는 상수 -> CAT_COLS에서 제외
BASE_CAT_COLS = ["top_bottom", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def fit_league_model(train, is_train_mask, is_val_mask, FEATURES, NUM_COLS):
    trend = fit_season_trend(train.loc[is_train_mask])
    sub = add_season_trend_feature(train, trend)

    X_train, y_train = sub.loc[is_train_mask, FEATURES], sub.loc[is_train_mask, TARGET]
    X_val, y_val = sub.loc[is_val_mask, FEATURES], sub.loc[is_val_mask, TARGET]

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
    return pred, y_val.to_numpy(), clf.best_iteration_


def run_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET, "game_type")]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    is_R = train["game_type"] == "R"
    is_F = train["game_type"] == "F"

    t = time.time()
    pred_R, y_R, it_R = fit_league_model(train, is_train & is_R, is_val & is_R, FEATURES, NUM_COLS)
    print(f"    R모델 | best_iter={it_R:4d} | n_val={len(y_R)} | {time.time()-t:.1f}s")

    t = time.time()
    pred_F, y_F, it_F = fit_league_model(train, is_train & is_F, is_val & is_F, FEATURES, NUM_COLS)
    print(f"    F모델 | best_iter={it_F:4d} | n_val={len(y_F)} | {time.time()-t:.1f}s")

    pred_all = np.concatenate([pred_R, pred_F])
    y_all = np.concatenate([y_R, y_F])

    s_R = score(pred_R, y_R)
    s_F = score(pred_F, y_F)
    s_combined = score(pred_all, y_all)
    print(f"    R단독 Score={s_R:.2f} | F단독 Score={s_F:.2f} | 통합 Score={s_combined:.2f}")
    return s_combined


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    results = {}
    for val_season in VAL_SEASONS:
        print(f"\n=== val_season={val_season} (R/F 분리 모델) ===")
        results[val_season] = run_fold(raw_train, prior_table, val_season)

    avg = sum(results.values()) / len(results)
    print(f"\n\n=== 요약 ===")
    print(results)
    print(f"R/F 분리 평균: {avg:.2f}")
    print(f"(참고) 기존 공유 모델(A) 평균: 1473.75  (2022=2261.64, 2024=685.86)")


if __name__ == "__main__":
    main()
