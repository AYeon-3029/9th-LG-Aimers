"""3단계 — shrinkage 피처를 추가한 학습 스크립트.

train_baseline.py와 구조는 동일하고, features.add_shrinkage_features()로 만든
4개 신규 컬럼만 FEATURES에 추가한다. 나머지(모델, 검증 분할, 지표)는 베이스라인과
동일하게 유지해서 "이 피처들이 실제로 점수를 올리는지"만 순수하게 비교한다.
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

from features import SHRINKAGE_FEATURE_NAMES, add_shrinkage_features, load_prior_table

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=BASE_FEATURES + [TARGET])

    prior_table = load_prior_table()
    train = add_shrinkage_features(train, prior_table)

    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    print("train:", train.shape, "| 피처:", len(FEATURES),
          f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)}, 신규 shrinkage {len(SHRINKAGE_FEATURE_NAMES)})")

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
    print(f"Validation Score: {score:.2f}  (베이스라인 416.18 대비 {score - 416.18:+.2f})")

    # 피처 중요도로 shrinkage 피처가 실제로 쓰이는지 확인
    importances = model.named_steps["clf"].feature_importances_
    imp_df = pd.Series(importances, index=FEATURES).sort_values(ascending=False)
    print("\n=== 피처 중요도 Top 10 ===")
    print(imp_df.head(10))
    print("\n=== shrinkage 피처 중요도 ===")
    print(imp_df.loc[SHRINKAGE_FEATURE_NAMES])

    t = time.time()
    model.fit(train[FEATURES], train[TARGET])
    print(f"\n재학습 완료 :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "prior_table": prior_table},
                "./model/rf_v2.pkl", compress=3)
    print("저장 완료: ./model/rf_v2.pkl")


if __name__ == "__main__":
    main()
