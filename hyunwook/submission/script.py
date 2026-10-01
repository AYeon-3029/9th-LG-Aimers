# script.py
"""추론(inference) 스크립트 — 평가 서버가 이 파일을 그대로 실행한다.

베이스라인(open/baseline_submit/script.py)과의 차이점:
  - 모델 파일이 raw Pipeline이 아니라 {"model", "features", "prior_table"} 딕셔너리로
    저장되어 있음 (train_v3.py 참고) — prior_table을 모델 파일 안에 함께 담아뒀기
    때문에 별도 prior_table.json을 제출 zip에 추가로 넣을 필요가 없다.
  - build_features()에서 features.py의 shrinkage/combo/season-trend 피처 엔지니어링을
    학습 때와 동일하게 재현한다. features.py는 이 스크립트와 같은 디렉터리에 있어야
    import가 된다.
"""
import os

import joblib
import pandas as pd

from features import (
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
)

ID_COL = "row_id"
TARGET_COL = "control_success"


# =======================
# 데이터 로드 유틸
# =======================

def load_test(path):
    """평가 데이터(csv) 로드. 한 행이 투구 하나."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 row_id 순서/컬럼 기준."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
            f"{list(df.columns)}")
    return df


# =======================
# 학습 때 사용한 전처리 (features.py 재사용으로 학습/추론 일치 보장)
# =======================

def build_features(df, prior_table, feature_names, k):
    """모델 입력 추출.

    1) features.py의 shrinkage/combo/season-trend 피처를 학습 때와 동일하게 생성
       (전부 이 행 안의 값 + prior_table 조회만 사용 — test 내부 통계 집계 없음)
       shrinkage k는 반드시 학습 때 쓴 값과 같아야 한다(model.pkl의 k_adopted를
       그대로 전달 — 하드코딩하면 학습/추론 desync 위험).
    2) 학습 때 저장해 둔 feature_names 순서/구성 그대로 선택
       (범주형 인코딩과 결측 대치는 모델 파일 안의 파이프라인이 함께 수행하므로
       여기서는 컬럼 생성/선택만 한다)
    """
    df = add_shrinkage_features(df, prior_table, k=k)
    df = add_combo_features(df)
    df = add_season_trend_feature(df, prior_table["season_trend"])
    return df[feature_names]


# =======================
# 제출 파일 생성 유틸
# =======================

def merge_predictions(sub, ids, preds):
    """sample_submission의 row_id 순서에 맞춰 예측 확률 병합.

    예측에 없는 row_id는 sample_submission의 기존 값(placeholder)을 유지한다.
    """
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.csv, sample_submission.csv 위치
    MODEL_DIR = "./model"          # rf_v3.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "catboost_final.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["features"]
    prior_table = bundle["prior_table"]
    k_adopted = bundle.get("k_adopted", 50)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}"
          f" (feature_names={len(feature_names)})")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    # ---- 전처리 (학습과 동일) ----
    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, prior_table, feature_names, k_adopted)
    print(f" features={X.shape[1]}")

    # ---- 예측 (제구 성공 확률) ----
    print("Inference model...")
    preds = model.predict_proba(X)[:, 1] if len(X) else []
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 ----
    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
