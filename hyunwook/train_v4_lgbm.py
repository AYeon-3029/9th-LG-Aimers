"""6단계(Action Item 4) — RandomForest 대신 LightGBM으로 같은 82피처를 학습해 비교.

전처리(ColumnTransformer: OrdinalEncoder + SimpleImputer)는 train_v3.py와 동일하게
유지하고, 분류기만 RandomForestClassifier -> LGBMClassifier로 교체한다.
이렇게 해야 "모델 자체의 차이"만 순수하게 비교할 수 있다.
"""

import os
import time

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
# pitcher_team_id/batter_team_id는 원래 수치형으로 취급되고 있었는데(Action Item 6
# 점검 중 발견), 이러면 트리가 "team_id <= 15.5" 같은 무의미한 크기 기준으로만
# 분기할 수 있어 batter_team_id 중요도가 0으로 완전히 죽어 있었음. 범주형으로
# 바꾸자 685.61 -> 685.86 (+0.25)로 개선되고 batter_team_id 중요도가 77위(0)에서
# 7위(396)로 급상승 — 팀 단위 효과(구장 특성 등)가 실제로 존재했다는 증거.
#
# ⚠️ pitcher_id/batter_id도 같은 원리로 범주형 전환해봤고 로컬은 690.52로
# 올랐지만(+4.66), 실제 리더보드 제출 결과는 758.37 -> 755.91로 오히려
# 하락했다. 세 번의 제출(685.86->758.37, 710.00->753.41, 690.52->755.91)이
# 로컬-실제 순위가 정확히 뒤바뀌는 패턴을 보여서, 로컬 검증 최고 기준점이었던
# 이 상태(팀ID만 범주형)로 되돌림 — 지금까지 실제 리더보드에서 가장 높았던
# 유일한 검증된 구성.
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

    # 1차 시도(n_estimators=500, lr=0.05, min_child_samples=200, subsample 0.8)는
    # 546.38로 RandomForest(606.98)보다 낮았음. early stopping + 더 많은 트리 +
    # 더 낮은 learning rate + 더 작은 min_child_samples(leaf-wise 트리 특성 고려)로 재시도.
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
    print(f"  이전 최고(685.86, 팀ID만 범주형 k=50) 대비 {score - 685.86:+.2f}")

    # ---- 전체 데이터(2019~2024)로 재학습 & 저장 ----
    # early stopping으로 찾은 best_iteration을 고정 n_estimators로 써서 재학습
    # (전체 데이터로 학습할 땐 더 이상 떼어둘 검증셋이 없으므로 early stopping 불가)
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
    )
    final_model = Pipeline([("pre", preprocessor), ("clf", final_clf)])

    t = time.time()
    final_model.fit(train[FEATURES], train[TARGET])
    print(f"\n전체 데이터 재학습 완료 (n_estimators={clf.best_iteration_}) :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    import joblib
    joblib.dump({"model": final_model, "features": FEATURES, "prior_table": prior_table},
                "./model/lgbm_v1.pkl", compress=3)
    print("저장 완료: ./model/lgbm_v1.pkl")


if __name__ == "__main__":
    main()
