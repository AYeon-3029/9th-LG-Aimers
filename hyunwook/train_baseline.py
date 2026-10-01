"""[Baseline_Train]_RandomForest를 활용한 모델 학습 및 피쳐엔지니어링 (학습).ipynb를 그대로 재현한 스크립트.

목적: 개선 작업을 시작하기 전, 현재 기준 Brier Skill Score를 확보하기 위한 기준점(baseline) 확인용.
노트북 내용과 100% 동일한 로직이며, 코드 흐름만 .py 스크립트로 옮겼다.
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

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def main():
    # =======================
    # 1. 데이터 불러오기
    # =======================
    # 사용할 피처 목록은 test.csv가 정한다. train.csv에만 있는 컬럼을 학습에 넣으면
    # 평가 시점에 그 컬럼이 없어 추론이 실패하기 때문.
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    FEATURES = [c for c in test_cols if c != ID]
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=FEATURES + [TARGET])

    print("train:", train.shape, "| 피처:", len(FEATURES),
          f"(범주형 {len(CAT_COLS)}, 수치형 {len(NUM_COLS)})")
    print("시즌:", train["season"].min(), "~", train["season"].max())
    print(f"제구 성공률: {train[TARGET].mean():.4f}")

    # =======================
    # 2. 전처리 정의
    # =======================
    # 범주형 3개는 정수로 바꾸고, 수치형 결측값은 중앙값으로 채운다.
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    # =======================
    # 3. 모델 정의와 학습
    # =======================
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

    # 2024 시즌을 검증용으로 떼어 두고 2019~2023으로 학습한다.
    is_val = train["season"] == 2024
    X_train, y_train = train.loc[~is_val, FEATURES], train.loc[~is_val, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    print("train:", len(X_train), "| val:", len(X_val))

    t = time.time()
    model.fit(X_train, y_train)
    print(f"학습 완료 :: {time.time() - t:.1f}s")

    # =======================
    # 4. 검증 — Brier Skill Score
    # =======================
    val_pred = model.predict_proba(X_val)[:, 1]

    r = y_val.mean()
    brier = ((val_pred - y_val) ** 2).mean()
    baseline_brier = r * (1 - r)
    score = max(0, 100000 * (1 - brier / baseline_brier))

    print(f"Brier: {brier:.6f} | 기준선 r(1-r): {baseline_brier:.6f}")
    print(f"Validation Score: {score:.2f}")

    # =======================
    # 5. 전체 데이터로 재학습 & 모델 저장
    # =======================
    t = time.time()
    model.fit(train[FEATURES], train[TARGET])
    print(f"재학습 완료 :: {time.time() - t:.1f}s")

    os.makedirs("./model", exist_ok=True)
    joblib.dump(model, "./model/rf.pkl", compress=3)
    print("저장 완료: ./model/rf.pkl")


if __name__ == "__main__":
    main()
