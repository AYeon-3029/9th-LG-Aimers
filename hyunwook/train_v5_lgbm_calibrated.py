"""6단계(Action Item 4) 최종 — LightGBM(튜닝) + CalibratedClassifierCV(cv=5, sigmoid).

이전 실험(train_v4_lgbm.py)에서 확인한 것:
  - RandomForest(max_features=None, 82피처): 606.98
  - LightGBM(튜닝 전, early stopping 없음): 546.38 (RF보다 낮음)
  - LightGBM(n_estimators=3000, lr=0.02, num_leaves=63, min_child_samples=50,
    early stopping): 685.61 (RF보다 +78.63)
  - 위 LightGBM에 sigmoid 확률 보정 추가(2024 검증셋을 calib/eval 절반으로 나눠
    깨끗하게 테스트): eval half에서 +67.99 추가 개선

이 스크립트는 "2019~2023 학습 / 2024 검증"이라는 지금까지의 비교 기준을 그대로
유지하면서, CalibratedClassifierCV(cv=5)로 보정까지 포함한 최종 수치를 뽑는다.
cv=5는 내부적으로 5-fold cross-validation으로 보정하므로 별도 held-out 세트가
필요 없다 (조기종료로 찾은 best_iteration=211을 고정 n_estimators로 사용).
"""

import os
import time

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
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
    load_prior_table,
)

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

# train_v4_lgbm.py의 early stopping 실험(2019~2023 학습/2024 검증)에서 찾은 값
BEST_N_ESTIMATORS = 211


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    baseline_brier = r * (1 - r)
    return max(0, 100000 * (1 - brier / baseline_brier)), brier


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

    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    base_clf = LGBMClassifier(
        n_estimators=BEST_N_ESTIMATORS,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    for label, clf in [
        ("보정 없음", base_clf),
        ("sigmoid 보정 (cv=5)", CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)),
        ("isotonic 보정 (cv=5)", CalibratedClassifierCV(base_clf, method="isotonic", cv=5)),
    ]:
        model = Pipeline([("pre", preprocessor), ("clf", clf)])
        t = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t

        val_pred = model.predict_proba(X_val)[:, 1]
        s, b = score(val_pred, y_val)
        print(f"[{label:20s}] 학습={elapsed:5.1f}s | Brier={b:.6f} | Score={s:.2f}")


if __name__ == "__main__":
    main()
