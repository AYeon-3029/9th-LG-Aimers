"""9단계 — Optuna로 찾은 정규화 파라미터(reg_alpha/reg_lambda/min_split_gain)를
적용한 제출용 모델 학습.

train_v4_lgbm.py(현재 채택 구성: 팀ID만 범주형, k=50)와 피처/전처리는 완전히
동일하고, LGBMClassifier에 train_v8_optuna.py에서 찾은 최적 정규화 파라미터
(trial 28, 다중 fold 2022+2024 평균 1473.75 -> 1481.65, +7.89)만 추가한다.

⚠️ 이 개선폭은 이전에 다중 fold에서 비슷한 크기로 좋아 보였다가 실제
리더보드에서는 나빠진 선례(README 6.10~6.12, config B)와 같은 급의 크기라
100% 신뢰할 수 없다 — 그래서 기존 lgbm_v1.pkl(유일하게 실제 검증된 구성)은
그대로 두고, 이 모델은 lgbm_v2_reg.pkl로 별도 저장해 실제 제출로 직접 검증한다.
"""

import os
import time

import joblib
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
    load_prior_table,
)

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state",
                  "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

# train_v8_optuna.py trial 28 (다중 fold 2022+2024 평균 기준 최적)
REG_PARAMS = dict(
    reg_alpha=1.047061314594877,
    reg_lambda=0.029327315094994536,
    min_split_gain=0.21925222920320947,
)


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
    print("정규화 파라미터:", REG_PARAMS)

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

    clf = LGBMClassifier(
        n_estimators=3000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **REG_PARAMS,
    )

    t = time.time()
    clf.fit(X_train_t, y_train, eval_set=[(X_val_t, y_val)],
            eval_metric="binary_logloss",
            callbacks=[__import__("lightgbm").early_stopping(100, verbose=False)])
    print(f"학습 완료 :: {time.time() - t:.1f}s (best_iteration={clf.best_iteration_})")

    model = Pipeline([("pre", preprocessor), ("clf", clf)])

    val_pred = model.predict_proba(X_val)[:, 1]
    r = y_val.mean()
    brier = ((val_pred - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))

    print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
    print(f"Validation Score: {score:.2f}")
    print(f"  현재 채택(정규화 없음, 685.86) 대비 {score - 685.86:+.2f}")

    # ---- 전체 데이터(2019~2024)로 재학습 & 저장 ----
    final_clf = LGBMClassifier(
        n_estimators=clf.best_iteration_,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=50,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **REG_PARAMS,
    )
    final_model = Pipeline([("pre", preprocessor), ("clf", final_clf)])

    t = time.time()
    final_model.fit(train[FEATURES], train[TARGET])
    print(f"\n전체 데이터 재학습 완료 (n_estimators={clf.best_iteration_}) :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURES, "prior_table": prior_table},
                "./model/lgbm_v2_reg.pkl", compress=3)
    print("저장 완료: ./model/lgbm_v2_reg.pkl")


if __name__ == "__main__":
    main()
