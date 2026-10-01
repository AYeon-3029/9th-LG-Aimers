"""20단계 — 이질적 하이퍼파라미터 CatBoost 앙상블 (902점 구성 대비).

배경: 18단계(시드 앙상블)는 같은 설정에서 초기화(random_seed)만 다르게 해 평균낸
것이었고, 결과는 +3~5 수준(애매 — 3번의 과거 LB 반전 사례와 같은 자릿수라 보류
중). [[lgaimers-validation-lessons]] 규칙 5("개별 정확도보다 에러 다양성이 앙상블에
더 중요")를 더 직접적으로 노려, 이번엔 depth/l2_leaf_reg/learning_rate 자체가
서로 다른 4개 모델을 평균한다 — 단순 초기화 노이즈가 아니라 실제로 다른 편향
(더 얕은/깊은 트리, 다른 정규화 강도)을 가진 모델들의 에러 패턴이 다양해지는지
확인한다.

⚠️ 개별 config 각각의 solo 점수는 902점 구성보다 낮을 수 있다(그게 자연스럽다 —
목적은 개별 최고점이 아니라 앙상블 후 다양성 이득). 19단계(grow_policy)에서 확인한
"이 데이터는 CatBoost의 SymmetricTree 구조에 유독 잘 맞는다"는 결론은 유지하고,
여기선 트리 구조(grow_policy)는 그대로 두고 depth/lr/l2만 바꾼다.

4개 config:
  A. depth=6, lr=0.02,  l2=3.0  (현재 902점 baseline)
  B. depth=4, lr=0.03,  l2=5.0  (얕고, 더 정규화)
  C. depth=8, lr=0.015, l2=6.0  (깊고, 느리게, 더 정규화로 상쇄)
  D. depth=6, lr=0.05,  l2=1.0  (같은 깊이, 빠르게 학습, 약한 정규화)

검증: A 단독(baseline) vs {A,B,C,D} 4-way 평균 vs {B,C,D} 3-way 평균(baseline 자체를
빼고 순수하게 다양한 것들끼리만 평균 — baseline이 상대적으로 강해서 평균을
자기 쪽으로 끌어당기는 효과를 배제하기 위함). val_season ∈ {2022, 2024}, 두 fold
모두 같은 방향이어야 신호로 인정, ±10 수준 등락은 노이즈로 기각.
"""

import time

import numpy as np
import pandas as pd

from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    CatBoostWrapper,
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "./open/data"
ID = "row_id"
TARGET = "control_success"
K = 50

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

COMMON_PARAMS = dict(
    iterations=3000, random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

CONFIGS = {
    "A(902점 baseline)": dict(depth=6, learning_rate=0.02, l2_leaf_reg=3.0),
    "B(얕음+정규화강화)": dict(depth=4, learning_rate=0.03, l2_leaf_reg=5.0),
    "C(깊음+정규화강화)": dict(depth=8, learning_rate=0.015, l2_leaf_reg=6.0),
    "D(빠른lr+약한정규화)": dict(depth=6, learning_rate=0.05, l2_leaf_reg=1.0),
}


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def prep_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]
    return X_train, y_train, X_val, y_val, NUM_COLS


def fit_config(X_train, y_train, X_val, y_val, num_cols, extra_params):
    params = dict(COMMON_PARAMS, **extra_params)
    wrapper = CatBoostWrapper(CAT_COLS, num_cols, **params)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    return wrapper.predict_proba(X_val)[:, 1], wrapper.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    labels = list(CONFIGS.keys())
    solo_scores = {label: {} for label in labels}
    blend4_scores = {}
    blend3_scores = {}  # B,C,D only (excludes A)

    for val_season in [2022, 2024]:
        X_train, y_train, X_val, y_val, num_cols = prep_fold(raw_train, prior_table, val_season)
        y_val_np = y_val.to_numpy()

        preds = {}
        for label, extra_params in CONFIGS.items():
            t = time.time()
            pred, best_iter = fit_config(X_train, y_train, X_val, y_val, num_cols, extra_params)
            preds[label] = pred
            s = score(pred, y_val_np)
            solo_scores[label][val_season] = s
            print(f"  val={val_season} | {label:20s} | best_iter={best_iter:4d} | "
                  f"solo Score={s:8.2f} | {time.time()-t:.0f}s")

        blend4 = np.mean([preds[l] for l in labels], axis=0)
        blend3 = np.mean([preds[l] for l in labels if l != "A(902점 baseline)"], axis=0)
        blend4_scores[val_season] = score(blend4, y_val_np)
        blend3_scores[val_season] = score(blend3, y_val_np)

    print("\n\n=== solo 점수 (참고용) ===")
    for label in labels:
        s = solo_scores[label]
        print(f"  {label:20s} | 2022={s[2022]:8.2f} | 2024={s[2024]:8.2f} | 평균={(s[2022]+s[2024])/2:8.2f}")

    print("\n=== 앙상블 요약 (baseline = A 단독, 902점 프로덕션과 동일) ===")
    b = solo_scores["A(902점 baseline)"]
    for label, scores in [("A 단독 (baseline)", b),
                           ("4-way 평균 (A+B+C+D)", blend4_scores),
                           ("3-way 평균 (B+C+D, A 제외)", blend3_scores)]:
        avg = (scores[2022] + scores[2024]) / 2
        d2022 = scores[2022] - b[2022]
        d2024 = scores[2024] - b[2024]
        print(f"  {label:28s} | 2022={scores[2022]:8.2f} ({d2022:+7.2f}) | "
              f"2024={scores[2024]:8.2f} ({d2024:+7.2f}) | 평균={avg:8.2f}")


if __name__ == "__main__":
    main()
