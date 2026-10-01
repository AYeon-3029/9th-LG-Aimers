"""도메인 지식 4종(주자 상황/불리한 카운트/경기 후반/최근 컨디션) 중 3개는 신규
"조건부 as-of 성공률" 피처로 leak-safe하게 구현한다(4번째 asof_pitcher_prev1_
game_success_rate는 이미 FEATURES에 있어 재사용). 오늘 이미 시도했던
asof_pitcher_late_inning_rate(등판 빈도 버전, 기각됨)와는 이름은 같지만 다른
정의(조건부 성공률)라 변수명을 late_inning_cond_rate로 구분한다.

설계는 shrunk_pitcher_rate와 동일한 패턴: train_df 내부는 진짜 causal 누적
(그 행 이전까지만), val_df에는 train_df 끝 시점의 투수별 고정 스냅샷을 조인해서
쓴다(test.csv rolling 금지 규칙 준수). shrinkage는 직전 시즌 (game_type) 리그
평균(해당 조건 하에서)으로 - 기존 SHRINK_K=100 그대로 재사용.

3개를 한 블록으로 묶어서 R 3구간 walk-forward, 트리 3종으로 먼저 스크리닝한다
(사용자 지시 1단계)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import pandas as pd
import train as T
from common import (
    FEATURES, TARGET, ASOF_FEATURES, SITUATIONAL_FEATURES, SHRINK_K,
    LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    add_matchup_features, brier_skill_score, split_regime,
)

DATA_DIR = T.DATA_DIR

CONDITIONS = {
    "asof_pitcher_scoring_pos_cond_rate": lambda df: (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1),
    "asof_pitcher_behind_count_cond_rate": lambda df: df["balls_before"] > df["strikes_before"],
    "asof_pitcher_late_inning_cond_rate": lambda df: df["inning"] >= 7,
}
NEW_FEATS = list(CONDITIONS.keys())


def load_data_with_extra_cols():
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "pitcher_id", "runner_on_2b", "runner_on_3b", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def shrink(raw, n, game_type, season, league_means, k=SHRINK_K):
    prior_idx = pd.MultiIndex.from_arrays([game_type, season - 1])
    prior = league_means.reindex(prior_idx).to_numpy()
    shrunk = (n * raw + k * prior) / (n + k)
    shrunk[np.isnan(raw) | np.isnan(prior)] = np.nan
    return shrunk


def main():
    df = load_data_with_extra_cols()
    feat_with_new = FEATURES + NEW_FEATS

    print("=== R walk-forward 3-split, tree ensemble only (3 conditional-rate features) ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R = train_df[train_df["game_type"] == "R"].copy()
        val_R = val_df[val_df["game_type"] == "R"].copy()

        coverages = []
        for feat_name, cond_fn in CONDITIONS.items():
            # --- train: causal cumulative ---
            is_cond_train = cond_fn(train_R).astype(np.float64)
            is_success_train = train_R[TARGET].to_numpy(dtype=np.float64)
            cond_success_train = is_cond_train.to_numpy() * is_success_train
            tmp = train_R.copy()
            tmp["_cond"] = is_cond_train.to_numpy()
            tmp["_cond_success"] = cond_success_train
            cum_cond = tmp.groupby("pitcher_id")["_cond"].cumsum() - tmp["_cond"]
            cum_cond_success = tmp.groupby("pitcher_id")["_cond_success"].cumsum() - tmp["_cond_success"]
            raw_train = (cum_cond_success / cum_cond).to_numpy()
            n_train = cum_cond.to_numpy()
            raw_train[n_train == 0] = np.nan

            tmp["_raw"] = raw_train
            league_means = tmp.groupby(["game_type", "season"])["_cond_success"].sum() / tmp.groupby(["game_type", "season"])["_cond"].sum()
            train_R[feat_name] = shrink(raw_train, n_train, train_R["game_type"], train_R["season"], league_means)

            # --- val: train 끝 시점 투수별 고정 스냅샷 ---
            total_cond = tmp.groupby("pitcher_id")["_cond"].sum()
            total_cond_success = tmp.groupby("pitcher_id")["_cond_success"].sum()
            n_val = val_R["pitcher_id"].map(total_cond).fillna(0).to_numpy(dtype=np.float64)
            raw_val = (val_R["pitcher_id"].map(total_cond_success) / val_R["pitcher_id"].map(total_cond)).to_numpy()
            raw_val[n_val == 0] = np.nan
            val_R[feat_name] = shrink(raw_val, n_val, val_R["game_type"], val_R["season"], league_means)
            coverages.append(val_R[feat_name].notna().mean())

        y_val = val_R[TARGET].to_numpy()
        p_lgb_b = T.fit_lgb(train_R, val_R, LGB_BASE_PARAMS, FEATURES)
        p_xgb_b = T.fit_xgb(train_R, val_R, XGB_BASE_PARAMS, FEATURES)
        p_cat_b = T.fit_cat(train_R, val_R, CAT_BASE_PARAMS, FEATURES)
        p_base = (p_lgb_b + p_xgb_b + p_cat_b) / 3
        _, bss_base = brier_skill_score(y_val, p_base)

        p_lgb_n = T.fit_lgb(train_R, val_R, LGB_BASE_PARAMS, feat_with_new)
        p_xgb_n = T.fit_xgb(train_R, val_R, XGB_BASE_PARAMS, feat_with_new)
        p_cat_n = T.fit_cat(train_R, val_R, CAT_BASE_PARAMS, feat_with_new)
        p_new = (p_lgb_n + p_xgb_n + p_cat_n) / 3
        _, bss_new = brier_skill_score(y_val, p_new)

        print(f"[train<={cutoff} val={val_season}] coverage={[f'{c:.1%}' for c in coverages]}  "
              f"baseline={bss_base:,.0f}  +3feats={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
