"""투수별이 아니라 리그 전체(population) 수준의 고정 상수로 조건부 성공률을
만든다(방식 A, 사용자 지시) - 시간에 따라 변하지 않고(드리프트 보정 계열 함정
회피), 투수별로도 안 쪼개서(표본 부족 함정 회피) 딱 2값 target encoding이 된다.
train_df(그 split의 것만, R은 전체/F는 split_regime 신체제만)에서 한 번만 계산해서
고정, val_df에는 그대로 적용한다(시간 무관이므로 매핑 자체가 leak-safe)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import pandas as pd
import train as T
from common import (
    FEATURES, TARGET, ASOF_FEATURES, SITUATIONAL_FEATURES,
    LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    add_matchup_features, brier_skill_score, split_regime,
)

DATA_DIR = T.DATA_DIR

CONDITIONS = {
    "pop_scoring_pos_rate": lambda df: (df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1),
    "pop_behind_count_rate": lambda df: df["balls_before"] > df["strikes_before"],
    "pop_late_inning_rate": lambda df: df["inning"] >= 7,
}
NEW_FEATS = list(CONDITIONS.keys())


def load_data_with_extra_cols():
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "runner_on_2b", "runner_on_3b", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def add_population_features(train_df, val_df):
    """train_df(R은 split_regime 이전 전체, F는 신체제만 - 호출부에서 이미 필터링된
    상태로 넘어옴)에서 딱 하나의 상수를 조건별로 구해서 train/val 둘 다에 적용한다."""
    train_df = train_df.copy()
    val_df = val_df.copy()
    for feat_name, cond_fn in CONDITIONS.items():
        cond_train = cond_fn(train_df)
        rate_true = train_df.loc[cond_train, TARGET].mean()
        rate_false = train_df.loc[~cond_train, TARGET].mean()

        train_df[feat_name] = np.where(cond_train, rate_true, rate_false)
        cond_val = cond_fn(val_df)
        val_df[feat_name] = np.where(cond_val, rate_true, rate_false)
    return train_df, val_df


def main():
    df = load_data_with_extra_cols()
    feat_with_new = FEATURES + NEW_FEATS

    print("=== R walk-forward 3-split, tree ensemble only (population-level fixed-constant features) ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R = train_df[train_df["game_type"] == "R"].copy()
        val_R = val_df[val_df["game_type"] == "R"].copy()

        train_R, val_R = add_population_features(train_R, val_R)

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

        print(f"[train<={cutoff} val={val_season}] baseline={bss_base:,.0f}  +3pop-feats={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
