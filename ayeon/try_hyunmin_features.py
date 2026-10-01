"""Hyunmin(팀원)이 제안한 10개 피처 블록을 프로덕션에 반영하기 전에 R 3구간
walk-forward, 트리 앙상블로 먼저 검증한다. 이 중 다수(is_scoring_pos,
pitcher_batter_diff, recent_trend, crisis_pressure/stubbornness_index/
control_struggle_index의 li 계열, is_full_count)는 오늘 세션에서 이미 개별적으로
검증되어 기각된 피처들과 동일/거의 동일하다 - 그래도 "10개를 한 블록으로 같이
넣으면" 상호작용으로 다른 결과가 나올 가능성을 배제하지 않고 직접 재검증한다."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import train as T
from common import FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, brier_skill_score, split_regime

NEW_FEATS = [
    "crisis_pressure", "is_full_count", "pitcher_ahead", "batter_ahead",
    "max_pitch_rate", "stubbornness_index", "control_struggle_index",
    "is_scoring_pos", "pitcher_batter_diff", "recent_trend",
]


def add_hyunmin_features(df):
    df = df.copy()
    df['crisis_pressure'] = df['li'] * (df['num_runners_on'] + 1)
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(np.float64)
    df['pitcher_ahead'] = (df['strikes_before'] > df['balls_before']).astype(np.float64)
    df['batter_ahead'] = (df['balls_before'] > df['strikes_before']).astype(np.float64)

    max_rate = df[['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate']].max(axis=1)
    df['max_pitch_rate'] = max_rate.fillna(0)
    df['stubbornness_index'] = df['max_pitch_rate'] * df['li']
    df['control_struggle_index'] = df['asof_pitcher_ball_rate'].fillna(0) * df['li']

    df['is_scoring_pos'] = ((df['runner_on_2b'] == 1) | (df['runner_on_3b'] == 1)).astype(np.float64)

    df['pitcher_batter_diff'] = df['asof_pitcher_success_rate'].fillna(0.5) - df['asof_batter_success_rate'].fillna(0.5)
    df['recent_trend'] = (df['asof_pitcher_prev1_game_success_rate'].fillna(df['asof_pitcher_success_rate']) - df['asof_pitcher_success_rate'].fillna(0.5)).fillna(0)
    return df


def load_data_with_runners():
    import pandas as pd
    from common import ASOF_FEATURES, SITUATIONAL_FEATURES, add_matchup_features
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "runner_on_2b", "runner_on_3b", TARGET]
    )
    df = pd.read_csv(T.DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def main():
    df = load_data_with_runners()
    df = add_hyunmin_features(df)
    feat_with_new = FEATURES + NEW_FEATS

    print("=== R walk-forward 3-split, tree ensemble only (hyunmin 10-feature block) ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R, _ = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
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

        print(f"[train<={cutoff} val={val_season}] baseline={bss_base:,.0f}  +10feats={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
