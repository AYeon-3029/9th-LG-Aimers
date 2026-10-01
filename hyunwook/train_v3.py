"""4단계 — shrinkage 피처(3단계) + Sample.md 조합 피처(4단계)를 함께 적용한 학습 스크립트.

train_v2.py에 features.add_combo_features()로 만든 조합 피처들을 추가한다.
CAT_COLS에 새 범주형 조합(count_state, hand_matchup, outs_base_state)을 포함시켜야
OrdinalEncoder가 문자열을 정수로 바꿔준다.
"""

import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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

    print("train:", train.shape, "| 피처:", len(FEATURES),
          f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)}, "
          f"shrinkage {len(SHRINKAGE_FEATURE_NAMES)}, combo {len(COMBO_FEATURE_NAMES)}, "
          f"trend {len(TREND_FEATURE_NAMES)})")

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=200,
            max_features=None,  # 기본값 'sqrt'는 분기당 후보 피처가 너무 적어(약 9개)
                                 # 새로 추가한 피처들끼리 서로 기회를 빼앗는 병목이었음.
                                 # None(전체 피처 후보)으로 바꾸자 502.24 → 606.58로 급등.
            n_jobs=-1,
            random_state=42,
        )),
    ])

    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    t = time.time()
    model.fit(X_train, y_train)
    print(f"학습 완료 :: {time.time() - t:.1f}s")

    val_pred = model.predict_proba(X_val)[:, 1]
    r = y_val.mean()
    brier = ((val_pred - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))

    print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"  베이스라인(1단계) 416.18 대비 {score - 416.18:+.2f}")
    print(f"  shrinkage만(3단계) 440.02 대비 {score - 440.02:+.2f}")
    print(f"  shrinkage+combo(4단계) 463.65 대비 {score - 463.65:+.2f}")

    importances = model.named_steps["clf"].feature_importances_
    imp_df = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
    print("\n=== 피처 중요도 Top 15 ===")
    print(imp_df.head(15))
    print("\n=== combo 피처 중요도 (내림차순) ===")
    print(imp_df.loc[COMBO_FEATURE_NAMES].sort_values(ascending=False))
    print("\n=== trend 피처 중요도 ===")
    print(imp_df.loc[TREND_FEATURE_NAMES])

    t = time.time()
    model.fit(train[FEATURES], train[TARGET])
    print(f"\n재학습 완료 :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "prior_table": prior_table},
                "./model/rf_v3.pkl", compress=3)
    print("저장 완료: ./model/rf_v3.pkl")


if __name__ == "__main__":
    main()
