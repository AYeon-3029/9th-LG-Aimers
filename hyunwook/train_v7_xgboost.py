"""7단계 — LightGBM 대신 XGBoost로 같은 82피처(팀ID만 범주형, k=50)를 학습해 비교.

전처리(ColumnTransformer: OrdinalEncoder + SimpleImputer)와 피처 구성은
train_v4_lgbm.py(현재 채택 구성, 실제 리더보드에서 유일하게 검증된 조합)와
완전히 동일하게 유지하고, 분류기만 LGBMClassifier -> XGBClassifier로 교체한다.
"모델 자체의 차이"만 순수하게 비교하기 위함.
"""

import os
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    load_prior_table,
)

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"

# train_v4_lgbm.py와 동일 — 팀ID만 범주형, 선수ID는 수치형 유지 (현재 채택 구성)
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state",
                  "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=BASE_FEATURES + [TARGET])

    prior_table = load_prior_table()
    train = add_shrinkage_features(train, prior_table)
    train = add_combo_features(train)
    train = add_season_trend_feature(train, prior_table["season_trend"])

    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    print("train:", train.shape, "| 피처:", len(FEATURES))

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    preprocessor.fit(X_train)
    X_train_t = preprocessor.transform(X_train)
    X_val_t = preprocessor.transform(X_val)

    # LightGBM 설정(learning_rate=0.02, n_estimators=3000, early stopping 100)과
    # 최대한 대응되는 값으로 시작 (max_leaves로 num_leaves=63과 맞춤, hist 트리 방식으로
    # leaf-wise에 가깝게 growth policy 설정)
    clf = XGBClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        max_leaves=63,
        grow_policy="lossguide",
        min_child_weight=50,
        subsample=1.0,
        colsample_bytree=1.0,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
    )

    t = time.time()
    clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)], verbose=False)
    print(f"학습 완료 :: {time.time() - t:.1f}s (best_iteration={clf.best_iteration})")

    model = Pipeline([("pre", preprocessor), ("clf", clf)])

    val_pred = model.predict_proba(X_val)[:, 1]
    r = y_val.mean()
    brier = ((val_pred - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))

    print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"  현재 채택(LightGBM, 685.86) 대비 {score - 685.86:+.2f}")


if __name__ == "__main__":
    main()
