"""26단계 — 비선형 GBDT 스태킹 메타러너 (CatBoost 2변형 + 맥락 피처).

배경: 지금까지 시도한 블렌딩은 전부 선형/볼록결합이었고([[lgaimers-model-state]]
항목 1의 OOF stacking 조사에서 균등가중치로 수렴한다는 게 이미 확인됨), 이건
"상황에 따라 어느 모델을 더 믿을지"를 표현할 수 없다. 이번엔 두 CatBoost 변형의
OOF 예측값 + 맥락 피처(game_type, asof_pitcher_n 등)를 작은 CatBoost 메타러너에
넣어 비선형 조합을 학습한다.

베이스 모델 2개:
  - 모델1: hyunwook님 902.04 그대로(공유, R/F 안 나눔)
  - 모델2: R/F 완전 분리(game_type별로 별도 CatBoost) — ayeon 아키텍처의 핵심
    아이디어를 hyunwook님 피처셋 위에 그대로 적용. hyunwook님 자신의 6.19절
    실험(R/F분리+CatBoost, merged/ 폴더)에서 이미 -100점 이상 손해였다는 게
    확인됐지만, 그건 "이 자체로 제출할 단일 모델"로 평가했을 때 얘기다 - 여기서는
    "다른 관점을 제공하는 메타러너의 입력 중 하나"로만 쓴다. 개별 정확도가
    낮아도 에러 패턴이 다르면 메타러너에 유용할 수 있다는 게 이 실험의 가설.

OOF 생성: train<2024를 3-fold(랜덤)로 나눠 각 fold를 검증으로 삼아 두 모델을
학습 -> 나머지 fold에 대한 예측을 모아 전체 train<2024의 OOF 예측을 만든다(각
서브 fit은 그 fold의 5%를 다시 떼어 early stopping에 씀, 원본 CatBoost
하이퍼파라미터는 902.04와 동일 depth/lr/l2 유지 - 오늘 확인한 대로 이 방향 튜닝은
건드리지 않는다).

메타러너 학습: [oof_pred_shared, oof_pred_split, game_type, asof_pitcher_n,
asof_batter_n, season, inning, balls_before, strikes_before, base_state] ->
control_success, 작은 CatBoost(depth=3)로.

평가: train<2024 전체로 두 모델을 다시 학습(OOF 아닌 진짜 fit) -> val=2024 예측 생성
-> 메타러너에 통과 -> val=2024 BSS. 902.04 baseline, 그리고 "메타러너가 실제로
비선형 정보를 추가하는지" 확인용으로 단순 평균 블렌딩과도 비교한다.
"""

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
N_FOLDS = 3
EVAL_HOLDOUT_FRAC = 0.05

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)

META_CONTEXT_NUM = ["asof_pitcher_n", "asof_batter_n", "season", "inning",
                     "balls_before", "strikes_before"]
META_CONTEXT_CAT = ["game_type", "base_state"]
META_CAT_PARAMS = dict(
    iterations=500, learning_rate=0.05, depth=3, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)


def bss(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def prep_base(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    train_df = train.loc[is_train].reset_index(drop=True)
    val_df = train.loc[is_val].reset_index(drop=True)
    return train_df, val_df, FEATURES, NUM_COLS


def fit_with_holdout(X, y, features, cat_cols, rng):
    """early stopping용으로 5%를 떼고 나머지로 학습, best_iteration까지만."""
    n = len(X)
    idx = rng.permutation(n)
    n_hold = max(1, int(n * EVAL_HOLDOUT_FRAC))
    hold_idx, fit_idx = idx[:n_hold], idx[n_hold:]
    num_cols = [c for c in features if c not in cat_cols]
    wrapper = CatBoostWrapper(cat_cols, num_cols, **CATBOOST_PARAMS)
    wrapper.fit(X.iloc[fit_idx][features], y.iloc[fit_idx],
                eval_set=(X.iloc[hold_idx][features], y.iloc[hold_idx]),
                early_stopping_rounds=50)
    return wrapper


def generate_oof(train_df, features, cat_cols, model2_split, n_folds=N_FOLDS, seed=0):
    """model2_split=False -> 공유 모델(모델1). True -> R/F 분리(모델2)."""
    n = len(train_df)
    oof = np.zeros(n, dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)

    for fold_i, (tr_idx, va_idx) in enumerate(kf.split(train_df)):
        t = time.time()
        tr_part = train_df.iloc[tr_idx].reset_index(drop=True)
        va_part = train_df.iloc[va_idx]

        if not model2_split:
            wrapper = fit_with_holdout(tr_part, tr_part[TARGET], features, cat_cols, rng)
            pred = wrapper.predict_proba(va_part[features])[:, 1]
            oof[va_idx] = pred
        else:
            pred = np.zeros(len(va_idx), dtype=np.float64)
            for league in ["R", "F"]:
                tr_league = tr_part[tr_part["game_type"] == league].reset_index(drop=True)
                va_mask = (va_part["game_type"] == league).to_numpy()
                if va_mask.sum() == 0 or len(tr_league) < 100:
                    continue
                wrapper = fit_with_holdout(tr_league, tr_league[TARGET], features, cat_cols, rng)
                pred[va_mask] = wrapper.predict_proba(va_part[features][va_mask])[:, 1]
            oof[va_idx] = pred

        print(f"    fold={fold_i} (split={model2_split}) done | {time.time()-t:.1f}s")

    return oof


def fit_full_and_predict(train_df, val_df, features, cat_cols, model2_split):
    if not model2_split:
        wrapper = CatBoostWrapper(cat_cols, [c for c in features if c not in cat_cols], **CATBOOST_PARAMS)
        wrapper.fit(train_df[features], train_df[TARGET],
                    eval_set=(val_df[features], val_df[TARGET]), early_stopping_rounds=100)
        return wrapper.predict_proba(val_df[features])[:, 1]
    else:
        pred = np.zeros(len(val_df), dtype=np.float64)
        for league in ["R", "F"]:
            tr_league = train_df[train_df["game_type"] == league]
            va_mask = (val_df["game_type"] == league).to_numpy()
            if va_mask.sum() == 0:
                continue
            wrapper = CatBoostWrapper(cat_cols, [c for c in features if c not in cat_cols], **CATBOOST_PARAMS)
            va_league = val_df[va_mask]
            wrapper.fit(tr_league[features], tr_league[TARGET],
                        eval_set=(va_league[features], va_league[TARGET]), early_stopping_rounds=100)
            pred[va_mask] = wrapper.predict_proba(va_league[features])[:, 1]
        return pred


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    val_season = 2024
    train_df, val_df, FEATURES, NUM_COLS = prep_base(raw_train, prior_table, val_season)
    print(f"train={len(train_df)} | val={len(val_df)}")

    print("\n=== OOF 생성: 모델1(공유) ===")
    oof1 = generate_oof(train_df, FEATURES, CAT_COLS, model2_split=False)
    print("\n=== OOF 생성: 모델2(R/F 분리) ===")
    oof2 = generate_oof(train_df, FEATURES, CAT_COLS, model2_split=True)

    print("\n=== 전체 train<2024로 재학습 -> val=2024 예측 ===")
    t = time.time()
    val_pred1 = fit_full_and_predict(train_df, val_df, FEATURES, CAT_COLS, model2_split=False)
    print(f"  모델1(공유) val 학습 완료 | {time.time()-t:.1f}s")
    t = time.time()
    val_pred2 = fit_full_and_predict(train_df, val_df, FEATURES, CAT_COLS, model2_split=True)
    print(f"  모델2(R/F분리) val 학습 완료 | {time.time()-t:.1f}s")

    print("\n=== 메타러너 학습 ===")
    meta_train = pd.DataFrame({
        "oof_shared": oof1, "oof_split": oof2,
    })
    for c in META_CONTEXT_NUM:
        meta_train[c] = pd.to_numeric(train_df[c], errors="coerce")
    for c in META_CONTEXT_CAT:
        meta_train[c] = train_df[c].astype(str)
    meta_features = ["oof_shared", "oof_split"] + META_CONTEXT_NUM + META_CONTEXT_CAT

    meta_val = pd.DataFrame({"oof_shared": val_pred1, "oof_split": val_pred2})
    for c in META_CONTEXT_NUM:
        meta_val[c] = pd.to_numeric(val_df[c], errors="coerce")
    for c in META_CONTEXT_CAT:
        meta_val[c] = val_df[c].astype(str)

    meta_wrapper = CatBoostWrapper(META_CONTEXT_CAT, ["oof_shared", "oof_split"] + META_CONTEXT_NUM,
                                    **META_CAT_PARAMS)
    meta_wrapper.fit(meta_train[meta_features], train_df[TARGET],
                      eval_set=(meta_val[meta_features], val_df[TARGET]), early_stopping_rounds=50)
    meta_pred = meta_wrapper.predict_proba(meta_val[meta_features])[:, 1]

    y_val = val_df[TARGET].to_numpy()
    score_shared_solo = bss(val_pred1, y_val)
    score_split_solo = bss(val_pred2, y_val)
    score_avg_blend = bss((val_pred1 + val_pred2) / 2, y_val)
    score_meta = bss(meta_pred, y_val)

    print(f"\n=== 결과 (val={val_season}) ===")
    print(f"  모델1(공유, 902점 구성) 단독      : {score_shared_solo:8.2f}")
    print(f"  모델2(R/F분리) 단독                : {score_split_solo:8.2f}")
    print(f"  단순 평균 블렌딩(선형)             : {score_avg_blend:8.2f}")
    print(f"  비선형 메타러너                    : {score_meta:8.2f}")
    print(f"  메타러너 - 공유 단독: {score_meta - score_shared_solo:+.2f}")
    print(f"  메타러너 - 평균 블렌딩: {score_meta - score_avg_blend:+.2f}")


if __name__ == "__main__":
    main()
