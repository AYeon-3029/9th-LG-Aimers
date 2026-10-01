"""5단계 — 시간 기반 다중 fold 검증 (walk-forward / 확장 윈도우 방식).

⚠️ 1차 시도는 `season == val_season`만으로 나머지를 전부 학습에 썼는데, 이러면
예를 들어 val=2022일 때 학습 데이터에 2023~2024(미래 시즌)까지 섞여 들어가는
문제가 있었다. 실제 제출 시나리오(2019~2024로 학습 → 미공개 2025로 평가)와
인과 구조가 달라서, 그 결과(2022=1810.72, 2023=0.00, 2024=463.65)는 신뢰할 수
없다고 판단해 폐기했다.

수정한 방식: 검증 시즌보다 "앞선 시즌만" 학습에 사용하는 walk-forward
(확장 윈도우) 방식으로 바꿨다. val=2021은 2019~2020으로, val=2024는
2019~2023으로 학습하는 식 — 매 fold가 실제 제출 시나리오와 동일한 시간 방향을
가진다.

남은 한계: prior_table.json은 여전히 전체 2019~2024로 미리 계산되어 있어,
이른 시즌(예: val=2021)을 검증할 때는 prior 안에 미래 정보가 약간 섞여 있다.
완벽하게 하려면 fold마다 prior도 다시 계산해야 하지만, 이번 점검의 목적은
"모델/피처 구조가 특정 시즌에만 과적합되지 않았는가"를 보는 안정성 확인이라
이 정도 근사는 허용하고 진행한다.
"""

import os
import time

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
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "./open/data"

ID = "row_id"
TARGET = "control_success"
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

VAL_SEASONS = [2021, 2022, 2023, 2024]


def brier_skill_score(y_true, y_pred):
    r = y_true.mean()
    brier = ((y_pred - y_true) ** 2).mean()
    baseline_brier = r * (1 - r)
    return max(0, 100000 * (1 - brier / baseline_brier)), brier, baseline_brier


def main():
    test_cols = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),
                             encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]

    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=BASE_FEATURES + [TARGET])

    prior_table = load_prior_table()
    train = add_shrinkage_features(train, prior_table)
    train = add_combo_features(train)

    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    scores = []
    for val_season in VAL_SEASONS:
        # walk-forward: 검증 시즌보다 "앞선" 시즌만 학습에 사용 (미래 정보 누출 방지)
        is_train = train["season"] < val_season
        is_val = train["season"] == val_season

        # 시즌 추세도 fold마다 "그 시점까지의 학습 데이터"만으로 다시 fit
        # (prior_table.json의 추세는 전체 2019~2024 기준이라 여기선 안 씀 — 누출 방지)
        fold_trend = fit_season_trend(train.loc[is_train])
        fold_df = add_season_trend_feature(train, fold_trend)

        X_train, y_train = fold_df.loc[is_train, FEATURES], train.loc[is_train, TARGET]
        X_val, y_val = fold_df.loc[is_val, FEATURES], train.loc[is_val, TARGET]

        preprocessor = ColumnTransformer([
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1), CAT_COLS),
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
        ])
        model = Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=200,
                n_jobs=-1, random_state=42,
            )),
        ])

        t = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t

        val_pred = model.predict_proba(X_val)[:, 1]
        score, brier, base_brier = brier_skill_score(y_val, val_pred)
        scores.append(score)
        print(f"[val={val_season}] train={len(X_train)} val={len(X_val)} "
              f"학습={elapsed:.1f}s | Brier={brier:.6f} | Score={score:.2f}")

    s = pd.Series(scores, index=VAL_SEASONS)
    print("\n=== fold별 Validation Score 요약 ===")
    print(s)
    print(f"평균: {s.mean():.2f} | 표준편차: {s.std():.2f}")
    print(f"(참고: 2024 단일 분할 결과는 463.65)")


if __name__ == "__main__":
    main()
