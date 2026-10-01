"""try_population_cond_features.py의 3개 조건(주자상황/카운트/이닝)을 각각
단독으로 R 3구간 walk-forward에서 테스트한다 - 셋 중 하나가 진짜 신호이고
나머지 둘이 희석시키고 있었는지 확인(사용자 지시 ablation)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import pandas as pd
import train as T
from common import (
    FEATURES, TARGET, ASOF_FEATURES, SITUATIONAL_FEATURES,
    LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    add_matchup_features, brier_skill_score,
)

DATA_DIR = T.DATA_DIR

CONDITIONS = {
    "pop_scoring_pos_rate": lambda df: (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1),
    "pop_behind_count_rate": lambda df: df["balls_before"] > df["strikes_before"],
    "pop_late_inning_rate": lambda df: df["inning"] >= 7,
}


def load_data_with_extra_cols():
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "runner_on_2b", "runner_on_3b", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def add_one_feature(train_df, val_df, feat_name, cond_fn):
    train_df = train_df.copy()
    val_df = val_df.copy()
    cond_train = cond_fn(train_df)
    rate_true = train_df.loc[cond_train, TARGET].mean()
    rate_false = train_df.loc[~cond_train, TARGET].mean()
    train_df[feat_name] = np.where(cond_train, rate_true, rate_false)
    cond_val = cond_fn(val_df)
    val_df[feat_name] = np.where(cond_val, rate_true, rate_false)
    return train_df, val_df


def main():
    df = load_data_with_extra_cols()

    for feat_name, cond_fn in CONDITIONS.items():
        print(f"\n=== {feat_name} 단독 ===")
        for cutoff, val_season in T.WALK_FORWARD_SPLITS:
            train_df = df[df["season"] <= cutoff]
            val_df = df[df["season"] == val_season]
            train_R = train_df[train_df["game_type"] == "R"].copy()
            val_R = val_df[val_df["game_type"] == "R"].copy()

            train_R_new, val_R_new = add_one_feature(train_R, val_R, feat_name, cond_fn)
            feat_with_new = FEATURES + [feat_name]

            y_val = val_R_new[TARGET].to_numpy()
            p_lgb_b = T.fit_lgb(train_R, val_R, LGB_BASE_PARAMS, FEATURES)
            p_xgb_b = T.fit_xgb(train_R, val_R, XGB_BASE_PARAMS, FEATURES)
            p_cat_b = T.fit_cat(train_R, val_R, CAT_BASE_PARAMS, FEATURES)
            p_base = (p_lgb_b + p_xgb_b + p_cat_b) / 3
            _, bss_base = brier_skill_score(y_val, p_base)

            p_lgb_n = T.fit_lgb(train_R_new, val_R_new, LGB_BASE_PARAMS, feat_with_new)
            p_xgb_n = T.fit_xgb(train_R_new, val_R_new, XGB_BASE_PARAMS, feat_with_new)
            p_cat_n = T.fit_cat(train_R_new, val_R_new, CAT_BASE_PARAMS, feat_with_new)
            p_new = (p_lgb_n + p_xgb_n + p_cat_n) / 3
            _, bss_new = brier_skill_score(y_val, p_new)

            print(f"  [train<={cutoff} val={val_season}] baseline={bss_base:,.0f}  +feat={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
