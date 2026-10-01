"""17단계 — CatBoost `max_ctr_complexity` 스윕 (902점 구성 대비 단일 변수 변경).

배경: 6.18절에서 LightGBM->CatBoost 전환(네이티브 범주형 target statistics)이
로컬 +97/실제 LB +143.7이라는 유일한 자릿수급 개선을 만들었다. `max_ctr_complexity`는
CatBoost가 CAT_COLS(현재 8개: top_bottom/game_type/base_state/team_id 2개 +
count_state/hand_matchup/outs_base_state) 중 몇 개까지 동시에 조합해 target
statistics를 만들지 정하는 파라미터 — 지금은 CatBoost 기본값(CPU 기준 4)을 그대로
쓰고 있다. hand_matchup/count_state 같은 조합은 이미 수작업으로 만들어 넣었지만,
`game_type × hand_matchup × base_state` 같은 더 높은 차수의 조합은 아직 아무도
명시적으로 열어준 적이 없다 — 이건 902점을 만든 것과 "같은 메커니즘"의 연장이라
±10 튜닝이 아니라 capacity/structure 범주로 분류된다(HANDOFF 방법론 규칙).

⚠️ 리스크: F(퓨처스)처럼 표본이 작은 범주 조합(game_type=F × 희귀 hand_matchup)에서
고차 조합의 CTR 버킷이 콜랩스할 수 있다 — 여기서는 game_type을 분리하지 않으므로
전체 표본(122만 행) 기준이라 위험이 크진 않지만, feature importance와 두 fold
모두에서 일관된 방향인지 반드시 확인한다.

검증 프로토콜은 6.14/6.16/6.20과 동일: val_season ∈ {2022, 2024} (2023은 ABS 단절로
병적인 fold라 제외), 두 fold 모두 같은 방향이어야 신호로 인정. 채택 기준(6.18):
±10 수준 등락은 노이즈로 기각 — 자릿수 다른 개선만 제출 후보로 고려.
"""

import time

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

# 현재 902점 프로덕션과 완전히 동일 + max_ctr_complexity만 변경
BASE_CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

# None = 파라미터 자체를 안 넘김 (CatBoost 기본값, 현재 902점 구성과 완전히 동일)
CTR_COMPLEXITY_GRID = [None, 4, 5, 6, 8]


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, ctr_complexity):
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

    params = dict(BASE_CATBOOST_PARAMS)
    if ctr_complexity is not None:
        params["max_ctr_complexity"] = ctr_complexity

    t = time.time()
    wrapper = CatBoostWrapper(CAT_COLS, NUM_COLS, **params)
    wrapper.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=100)
    pred = wrapper.predict_proba(X_val)[:, 1]
    elapsed = time.time() - t

    return score(pred, y_val.to_numpy()), wrapper.best_iteration_, elapsed


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    results = {}
    for ctr_complexity in CTR_COMPLEXITY_GRID:
        label = f"max_ctr_complexity={ctr_complexity if ctr_complexity is not None else '기본값(902점 baseline)'}"
        print(f"\n=== {label} ===")
        scores = {}
        for val_season in [2022, 2024]:
            s, it, elapsed = run_fold(raw_train, prior_table, val_season, ctr_complexity)
            scores[val_season] = s
            print(f"  val={val_season} | best_iter={it:4d} | Score={s:8.2f} | {elapsed:.0f}s")
        results[label] = scores

    print("\n\n=== 요약 (baseline = max_ctr_complexity 기본값) ===")
    baseline_label = "max_ctr_complexity=기본값(902점 baseline)"
    b = results[baseline_label]
    for label, scores in results.items():
        avg = (scores[2022] + scores[2024]) / 2
        d2022 = scores[2022] - b[2022]
        d2024 = scores[2024] - b[2024]
        print(f"  {label:45s} | 2022={scores[2022]:8.2f} ({d2022:+7.2f}) | "
              f"2024={scores[2024]:8.2f} ({d2024:+7.2f}) | 평균={avg:8.2f}")


if __name__ == "__main__":
    main()
