"""19단계 — CatBoost `grow_policy` 전환 (902점 구성 대비 단일 변수 변경).

배경: 17/18단계에서 확인된 것 — `max_ctr_complexity` 확장은 무신호, 시드 앙상블은
±10 노이즈 밴드 안의 애매한 신호였다. 둘 다 "기존 트리 성장 방식 안에서" 변화를
준 것이었다. 이번엔 트리 성장 알고리즘 자체를 바꾼다.

지금 902점 구성은 CatBoost 기본값인 `SymmetricTree`(대칭/oblivious tree — 모든
리프가 같은 깊이, 같은 분기 규칙을 공유하는 CatBoost 고유 구조)를 그대로 쓰고
있다. 이건 한 번도 명시적으로 검토된 적이 없다 — CatBoost 채택 자체가 "네이티브
범주형 처리" 때문이었지 트리 구조 때문이 아니었다.

- `Depthwise`: 깊이 우선으로 모든 노드를 분기(LightGBM의 depth-wise와 유사),
  리프별로 다른 분기 규칙을 가질 수 있어 SymmetricTree보다 유연하지만 정규화는
  약함.
- `Lossguide`: 리프 단위로 손실 감소가 가장 큰 곳부터 분기(LightGBM 기본 방식과
  동일한 leaf-wise). `max_leaves`로 용량을 제어(`depth`가 아님).

이건 단순 정규화 강도 조절(이미 4번 실패)이 아니라 모델이 트리를 만드는 방식
자체를 바꾸는 구조 변경이라 capacity/structure 범주로 분류한다. 단, 6.15절에서
XGBoost(depth-wise 계열)가 LightGBM(leaf-wise)에 진 전례가 있어 — 이 데이터가
leaf-wise 계열과 안 맞을 가능성도 이미 한 번 시사된 적 있다는 점은 감안한다.

검증 프로토콜은 기존과 동일: val_season ∈ {2022, 2024}, 두 fold 모두 같은 방향이어야
신호로 인정, ±10 수준 등락은 노이즈로 기각.
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

BASE_CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

# (label, extra kwargs) — SymmetricTree(depth=6)가 현재 902점 구성(기본값, 명시 안 해도 동일)
GROW_POLICY_GRID = [
    ("SymmetricTree(depth=6, 902점 baseline)", dict(grow_policy="SymmetricTree", depth=6)),
    ("Depthwise(depth=6)", dict(grow_policy="Depthwise", depth=6)),
    ("Lossguide(max_leaves=64)", dict(grow_policy="Lossguide", max_leaves=64)),
]


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, extra_params):
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

    params = dict(BASE_CATBOOST_PARAMS, **extra_params)

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
    for label, extra_params in GROW_POLICY_GRID:
        print(f"\n=== {label} ===")
        scores = {}
        for val_season in [2022, 2024]:
            s, it, elapsed = run_fold(raw_train, prior_table, val_season, extra_params)
            scores[val_season] = s
            print(f"  val={val_season} | best_iter={it:4d} | Score={s:8.2f} | {elapsed:.0f}s")
        results[label] = scores

    print("\n\n=== 요약 (baseline = SymmetricTree, 902점 구성) ===")
    baseline_label = GROW_POLICY_GRID[0][0]
    b = results[baseline_label]
    for label, _ in GROW_POLICY_GRID:
        s = results[label]
        avg = (s[2022] + s[2024]) / 2
        d2022 = s[2022] - b[2022]
        d2024 = s[2024] - b[2024]
        print(f"  {label:40s} | 2022={s[2022]:8.2f} ({d2022:+7.2f}) | "
              f"2024={s[2024]:8.2f} ({d2024:+7.2f}) | 평균={avg:8.2f}")


if __name__ == "__main__":
    main()
