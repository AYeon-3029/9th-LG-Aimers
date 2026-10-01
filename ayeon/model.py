"""LightGBM, 무작위 20% holdout 버전 - 비교/대조용.

주의: 이 무작위 holdout은 실제 리더보드(2025 시즌, 완전히 못 본 시즌) 성능을
크게 과대평가한다 - 검증셋에도 학습셋과 같은 시즌 데이터가 섞이기 때문이다.
실제 제출 모델을 고를 때는 이 스크립트가 아니라 train.py의 시즌 단위
walk-forward holdout(train<=N -> val=N+1) 결과를 기준으로 판단할 것.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    ASOF_FEATURES,
    FEATURES,
    FINAL_N_ESTIMATORS,
    LGB_BASE_PARAMS,
    SITUATIONAL_FEATURES,
    TARGET,
    add_matchup_features,
    brier_skill_score,
)

DATA_DIR = Path(__file__).parent / "data"


def main():
    cols = (
        ASOF_FEATURES
        + SITUATIONAL_FEATURES
        + ["season", "pitcher_hand", "batter_hand", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    df = add_matchup_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(**LGB_BASE_PARAMS, n_estimators=FINAL_N_ESTIMATORS, verbose=-1)
    model.fit(X_train, y_train)

    p_holdout = model.predict_proba(X_holdout)[:, 1]
    brier, bss = brier_skill_score(y_holdout, p_holdout)

    print(f"train n={len(X_train):,} / holdout n={len(X_holdout):,}")
    print(f"holdout Brier = {brier:.5f}")
    print(f"holdout BSS   = {bss:,.0f}")


if __name__ == "__main__":
    main()
