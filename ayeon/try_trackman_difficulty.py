"""신규 피처: trackman 물리 난이도 지수. trackman_history.csv는 train/test와
row-level로 join 불가능하므로(data_findings 근거), (game_type, pitch_type_group)
단위의 population-level 통계만 leak-safe하게 쓸 수 있다.

이전 시도(공용 rationale 최상단)는 "시즌 리그 평균 구위 지표, 2025는 2024로
carry-forward"였고 실패했다(605 vs 680) - season이 사실상 피처에 섞여 들어가서
(테이블 값이 시즌마다 달라짐) season 직접 사용과 같은 실패 패턴이었다. 이번
설계는 그 실패 원인 자체를 구조적으로 제거한다: "레짐"은 피처 값을 시즌별로
바꾸는 lookup key가 아니라, trackman_history를 어느 기간까지만 신뢰할지 정하는
데이터 필터로만 쓴다. 그 결과 (game_type, pitch_type_group) -> 통계 테이블은
단 하나만 존재하고(시즌 조건 없음), train/val/test 모든 행에 동일하게 적용된다 -
"이 시즌엔 어떤 값" 같은 시즌 의존성이 아예 없으므로 이전 실패 패턴이 구조적으로
재발할 수 없다.

game_type(R/F) 판별은 team 이름의 MIN_ 접두사로(data_findings 근거,
MIN_*-접두 팀 = 퓨처스). R은 train_df의 최신 시즌만(레짐 필터), F는 기존
F_NEW_REGIME_START(2023) 그대로 재사용한다.

피처 하나만 만든다(규칙 7: 새 피처 추가 자체가 -5~-8 구조적 비용을 문다 -
여러 개를 한꺼번에 넣으면 이 비용이 배가된다): 투수의 asof 구종 믹스
(asof_pitcher_fastball/breaking/offspeed_rate, 이미 제공된 공식 피처)로
가중평균한 "구종 무브먼트 변동성"(수평+수직 브레이크의 표준편차 합성) - 타겟의
실패 조건("존에서 크게 벗어난 공")과 직접 연결되는 물리량이라는 가설이다.
"""
import lightgbm as lgb  # import 순서 안전 (env_notes 근거)
import numpy as np
import pandas as pd
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    brier_skill_score, split_regime, F_NEW_REGIME_START,
)

NEW_FEAT = "trackman_mix_break_volatility"
MIN_PREFIX = "MIN_"


def load_trackman():
    cols = ["season", "pitcher_team", "pitch_type_group", "horz_break", "induced_vert_break"]
    tm = pd.read_csv("data/trackman_history.csv", usecols=cols)
    tm["game_type"] = np.where(tm["pitcher_team"].str.startswith(MIN_PREFIX), "F", "R")
    return tm


R_TRACKMAN_WINDOW_SEASONS = 2  # 일반 규칙(특정 구간 전용 예외 아님) - 사용자 지적:
# "결과가 나쁜 구간만 골라 고치면 안 된다"는 원칙에 따라 3구간 전부에 동일하게
# 적용한다. 단일 시즌은 그 해 고유의 우연한 변동을 통계로 흡수할 위험이 있다는
# 가설을 최소 2시즌 트레일링 윈도우로 일반화해서 검증한다.


def build_volatility_table(tm, cutoff):
    """cutoff까지의 trackman만 쓰고(미래 정보 누출 방지), 리그별 '레짐'(신뢰할 최근
    구간)만 남긴다 - R은 최근 R_TRACKMAN_WINDOW_SEASONS개 시즌(3구간 전부 동일 규칙),
    F는 기존 F_NEW_REGIME_START 그대로. (game_type, pitch_type_group) ->
    sqrt(std_horz^2 + std_vert^2) 테이블 하나만 반환한다(시즌 조건 없음 - 그래서
    나중에 시즌별로 값이 안 바뀐다)."""
    tm = tm[tm["season"] <= cutoff]
    r_seasons = sorted(tm.loc[tm["game_type"] == "R", "season"].unique())
    r_window = r_seasons[-R_TRACKMAN_WINDOW_SEASONS:]
    r_part = tm[(tm["game_type"] == "R") & (tm["season"].isin(r_window))]
    f_part = tm[(tm["game_type"] == "F") & (tm["season"] >= F_NEW_REGIME_START)]
    combined = pd.concat([r_part, f_part], ignore_index=True)

    agg = combined.groupby(["game_type", "pitch_type_group"]).agg(
        std_horz=("horz_break", "std"), std_vert=("induced_vert_break", "std")
    )
    agg["volatility"] = np.sqrt(agg["std_horz"] ** 2 + agg["std_vert"] ** 2)
    return agg["volatility"]  # MultiIndex(game_type, pitch_type_group) -> float


def add_feature(df, vol_table, game_type):
    fb = vol_table.get((game_type, "fastball"), np.nan)
    br = vol_table.get((game_type, "breaking"), np.nan)
    os_ = vol_table.get((game_type, "offspeed"), np.nan)
    val = (
        df["asof_pitcher_fastball_rate"] * fb
        + df["asof_pitcher_breaking_rate"] * br
        + df["asof_pitcher_offspeed_rate"] * os_
    )
    df = df.copy()
    df[NEW_FEAT] = val
    return df


def main():
    df = T.load_data()
    tm = load_trackman()
    feat_with_new = FEATURES + [NEW_FEAT]

    print("=== R walk-forward 3-split, tree ensemble only ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R = train_df[train_df["game_type"] == "R"]
        val_R = val_df[val_df["game_type"] == "R"]

        r_seasons_avail = sorted(tm.loc[(tm["game_type"] == "R") & (tm["season"] <= cutoff), "season"].unique())
        r_window_used = r_seasons_avail[-R_TRACKMAN_WINDOW_SEASONS:]
        vol_table = build_volatility_table(tm, cutoff)
        train_R2 = add_feature(train_R, vol_table, "R")
        val_R2 = add_feature(val_R, vol_table, "R")
        y_val = val_R2[TARGET].to_numpy()
        coverage = val_R2[NEW_FEAT].notna().mean()
        print(f"  R trackman window: {r_window_used}")

        p_lgb_b = T.fit_lgb(train_R2, val_R2, LGB_BASE_PARAMS, FEATURES)
        p_xgb_b = T.fit_xgb(train_R2, val_R2, XGB_BASE_PARAMS, FEATURES)
        p_cat_b = T.fit_cat(train_R2, val_R2, CAT_BASE_PARAMS, FEATURES)
        p_base = (p_lgb_b + p_xgb_b + p_cat_b) / 3
        _, bss_base = brier_skill_score(y_val, p_base)

        p_lgb_n = T.fit_lgb(train_R2, val_R2, LGB_BASE_PARAMS, feat_with_new)
        p_xgb_n = T.fit_xgb(train_R2, val_R2, XGB_BASE_PARAMS, feat_with_new)
        p_cat_n = T.fit_cat(train_R2, val_R2, CAT_BASE_PARAMS, feat_with_new)
        p_new = (p_lgb_n + p_xgb_n + p_cat_n) / 3
        _, bss_new = brier_skill_score(y_val, p_new)

        print(f"[train<={cutoff} val={val_season}] coverage={coverage:.1%}  "
              f"baseline={bss_base:,.0f}  +feature={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
