"""16단계 — kang 팀원 코드의 미검증 아이디어 3개를 CatBoost 파이프라인에 결합 검증.

kang/script.py의 파생 피처 8개 중:
  - is_scoring_pos, pitcher_batter_diff, recent_trend → ayeon이 R/F 분리 구조
    위에서 이미 재검증하고 기각(노이즈 밴드 또는 F에서 -18~-27)
  - is_same_hand, is_full_count → 우리 파이프라인에 이미 사실상 존재
    (hand_matchup 범주형, full_count 플래그)
  - crisis_pressure, stubbornness_index, control_struggle_index → **아직 아무도
    검증 안 한 3개**. 전부 li(상황 중요도)를 다른 값과 곱하는 row-wise 피처.

⚠️ 사전 우려: 우리 프로젝트에서 "이미 강한 두 피처를 곱해 만든 interaction"은
반복적으로 실패했고(inning×asof_pitcher -5.15, inning×win_expectancy -7.16,
asof 곱/차이 -48.11), ayeon도 li 조건부 피처는 전부 실패시켰다(li가 target과
거의 무상관이라는 게 원인이라고 결론). 다만 (1) 그 실패들은 전부 LightGBM/RF
시절 결론이고 지금은 CatBoost로 바뀌었으며(CatBoost는 예전 639 -> 지금 783으로
평가가 완전히 뒤집힌 전례가 있음), (2) 이 3개는 집계/lookup이 아니라 그 행
자신의 값만 쓰는 leak-safe 피처라 시도 가치는 있다.

채택 기준: 실제 리더보드 결과(로컬 783.01 -> 실제 902)로 확인된 바,
자릿수가 다른 큰 개선만 실제로 전이된다. ±10 수준 등락은 노이즈로 보고 기각.
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

# 현재 프로덕션(실제 902점)과 완전히 동일한 설정
CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

KANG_FEATURE_NAMES = ["crisis_pressure", "stubbornness_index", "control_struggle_index"]


def add_kang_features(df):
    """kang/script.py와 동일 로직. 그 행 자신의 값만 사용하므로 leak-safe."""
    df = df.copy()
    df["crisis_pressure"] = df["li"] * (df["num_runners_on"] + 1)
    max_rate = df[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
                    "asof_pitcher_offspeed_rate"]].max(axis=1)
    df["stubbornness_index"] = max_rate.fillna(0) * df["li"]
    df["control_struggle_index"] = df["asof_pitcher_ball_rate"].fillna(0) * df["li"]
    return df


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, use_kang):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)
    if use_kang:
        train = add_kang_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    if use_kang:
        FEATURES = FEATURES + KANG_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

    wrapper = CatBoostWrapper(CAT_COLS, NUM_COLS, **CATBOOST_PARAMS)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    pred = wrapper.predict_proba(X_val)[:, 1]

    # 신규 피처가 실제로 쓰이는지(중요도 0이 아닌지) 확인
    imp = None
    if use_kang:
        fi = wrapper.model.get_feature_importance()
        imp = {c: round(float(fi[FEATURES.index(c)]), 3) for c in KANG_FEATURE_NAMES}
    return score(pred, y_val.to_numpy()), wrapper.best_iteration_, imp


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    results = {}
    for label, use_kang in [("baseline (현재 902점 구성)", False), ("+kang 3개 피처", True)]:
        print(f"\n=== {label} ===")
        scores = {}
        for val_season in [2022, 2024]:
            t = time.time()
            s, it, imp = run_fold(raw_train, prior_table, val_season, use_kang)
            scores[val_season] = s
            extra = f" | 신규피처 중요도={imp}" if imp else ""
            print(f"  val={val_season} | best_iter={it:4d} | Score={s:8.2f} | {time.time()-t:.0f}s{extra}")
        results[label] = scores

    print("\n\n=== 요약 ===")
    b, k = results["baseline (현재 902점 구성)"], results["+kang 3개 피처"]
    for vs in [2022, 2024]:
        print(f"  val={vs}: {b[vs]:8.2f} -> {k[vs]:8.2f}  ({k[vs]-b[vs]:+.2f})")
    avg_b = (b[2022] + b[2024]) / 2
    avg_k = (k[2022] + k[2024]) / 2
    print(f"  평균  : {avg_b:8.2f} -> {avg_k:8.2f}  ({avg_k-avg_b:+.2f})")


if __name__ == "__main__":
    main()
