"""신규 피처 시도: asof_pitcher_late_inning_rate - "이 투수가 8회 이후에
등판하는 빈도" (성공률과 엮지 않은 순수 사용 패턴/보직 신호).

배포 가능성 제약(data_description.md 5절, 대회 규칙): test.csv 행 순서 기반
rolling/expanding 피처는 금지된다. 따라서 이 피처는 shrunk_pitcher_rate와
똑같은 패턴으로 설계한다:
  - train_df 내부에서는 진짜 leak-safe 누적값(그 행 이전까지만, causal)을 쓴다
    - 이건 train.csv 자기 자신 안에서의 피처 엔지니어링이라 규칙 위반이 아니다.
  - val_df(walk-forward에서 실제 배포 시나리오의 대역)에는 train_df 끝
    시점까지의 "고정된" 투수별 스냅샷을 조인해서 쓴다 - val_df 자신의 행 순서를
    전혀 쓰지 않으므로 실제 배포 제약을 충실히 시뮬레이션한다.

shrinkage는 기존 SHRINK_K/season_means 패턴을 그대로 재사용(직전 시즌
(game_type) 리그 평균 쪽으로 축소) - 새 하이퍼파라미터를 늘리지 않는다.

1단계: R 3구간 walk-forward, 트리 3종 앙상블만으로 신호 확인 (싸고 빠름).
신호 있으면 2단계(4모델 블렌드)로 진행한다.
"""
import train as T
import pandas as pd
import numpy as np
from common import (
    FEATURES, TARGET, ASOF_FEATURES, SITUATIONAL_FEATURES, SHRINK_K,
    LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    add_matchup_features, brier_skill_score,
)

DATA_DIR = T.DATA_DIR

NEW_FEAT = "asof_pitcher_late_inning_rate"


def load_data_with_pitcher():
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand", "pitcher_id", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def add_train_causal_feature(df):
    """train_df 전용: 그 행 자신 이전까지의 진짜 누적치(causal, in-df 순서 신뢰)."""
    df = df.copy()
    df["_is_late"] = (df["inning"] >= 8).astype(np.float64)
    cumsum_incl = df.groupby("pitcher_id")["_is_late"].cumsum()
    late_before = cumsum_incl - df["_is_late"]
    n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    raw = late_before.to_numpy() / n
    raw[n == 0] = np.nan
    df["_late_raw"] = raw
    return df


def compute_league_late_means(train_df_with_causal):
    """(game_type, season) -> 리그 전체 late-inning 등판 비율. train에서만."""
    return train_df_with_causal.groupby(["game_type", "season"])["_is_late"].mean()


def shrink(raw, n, game_type, season, league_means, k=SHRINK_K):
    prior_idx = pd.MultiIndex.from_arrays([game_type, season - 1])
    prior = league_means.reindex(prior_idx).to_numpy()
    shrunk = (n * raw + k * prior) / (n + k)
    shrunk[np.isnan(raw) | np.isnan(prior)] = np.nan
    return shrunk


def build_pitcher_snapshot(train_df_with_causal):
    """train_df 끝 시점까지의 투수별 고정 스냅샷: 총 late 누적, 총 n."""
    last = train_df_with_causal.groupby("pitcher_id").agg(
        late_total=("_is_late", "sum"), n_total=("asof_pitcher_n", "max")
    )
    # n_total은 asof_pitcher_n(그 행 이전 누적)의 최댓값 + 그 행 자체 포함해야
    # "train 전체 학습 후" 시점의 총 누적이 된다.
    late_total_incl = train_df_with_causal.groupby("pitcher_id")["_is_late"].sum()
    n_incl = train_df_with_causal.groupby("pitcher_id").size()
    return pd.DataFrame({"late_total": late_total_incl, "n_total": n_incl})


def add_val_frozen_feature(val_df, snapshot, league_means, k=SHRINK_K):
    df = val_df.copy()
    joined = df["pitcher_id"].map(snapshot["late_total"])
    n_joined = df["pitcher_id"].map(snapshot["n_total"]).fillna(0).to_numpy(dtype=np.float64)
    raw = (joined / n_joined).to_numpy()
    raw[n_joined == 0] = np.nan
    df[NEW_FEAT] = shrink(raw, n_joined, df["game_type"], df["season"], league_means, k)
    return df


def add_train_feature_final(train_df_with_causal, league_means, k=SHRINK_K):
    df = train_df_with_causal
    n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    df[NEW_FEAT] = shrink(df["_late_raw"].to_numpy(), n, df["game_type"], df["season"], league_means, k)
    return df


def main():
    df = load_data_with_pitcher()
    feat_with_new = FEATURES + [NEW_FEAT]

    print("=== Stage 1: R walk-forward 3-split, tree ensemble only ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R = train_df[train_df["game_type"] == "R"].copy()
        val_R = val_df[val_df["game_type"] == "R"].copy()

        train_R_causal = add_train_causal_feature(train_R)
        league_means = compute_league_late_means(train_R_causal)
        train_R_final = add_train_feature_final(train_R_causal, league_means)
        snapshot = build_pitcher_snapshot(train_R_causal)
        val_R_final = add_val_frozen_feature(val_R, snapshot, league_means)

        y_val = val_R_final[TARGET].to_numpy()
        coverage = val_R_final[NEW_FEAT].notna().mean()

        p_lgb_base = T.fit_lgb(train_R_final, val_R_final, LGB_BASE_PARAMS, FEATURES)
        p_xgb_base = T.fit_xgb(train_R_final, val_R_final, XGB_BASE_PARAMS, FEATURES)
        p_cat_base = T.fit_cat(train_R_final, val_R_final, CAT_BASE_PARAMS, FEATURES)
        p_base = (p_lgb_base + p_xgb_base + p_cat_base) / 3
        _, bss_base = brier_skill_score(y_val, p_base)

        p_lgb_new = T.fit_lgb(train_R_final, val_R_final, LGB_BASE_PARAMS, feat_with_new)
        p_xgb_new = T.fit_xgb(train_R_final, val_R_final, XGB_BASE_PARAMS, feat_with_new)
        p_cat_new = T.fit_cat(train_R_final, val_R_final, CAT_BASE_PARAMS, feat_with_new)
        p_new = (p_lgb_new + p_xgb_new + p_cat_new) / 3
        _, bss_new = brier_skill_score(y_val, p_new)

        print(f"[train<={cutoff} val={val_season}] val coverage(non-NaN)={coverage:.1%}  "
              f"baseline bss={bss_base:,.0f}  +feature bss={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
