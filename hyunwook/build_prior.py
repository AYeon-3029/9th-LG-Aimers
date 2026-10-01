"""2단계 — 구종군(pitch_type_group) 단위 prior 산출.

설계:
  1) trackman_history.csv 로 구종군별 물리 특성(무브먼트 크기 등)을 집계해
     "난이도 지수"를 만든다. 이건 값 자체가 확률이 아니라, prior 값이 야구적으로
     타당한지 뒷받침하는 근거(도메인 신호)로만 쓴다.
     ※ trackman_history.csv에는 success/middle 라벨이 없고, pitcher_id와도
       매칭되지 않으므로(이전 검증 결과 교집합 0건) 개인화나 직접 확률 계산에는
       쓸 수 없다.
  2) train.csv 는 실제 정답(control_success)이 있는 유일한 데이터이므로,
     여기서 투수별 "구종 사용 비율(pitch mix)"과 "제구 성공률"의 관계를
     가중회귀로 추정해 구종군별 prior 성공률/middle률을 뽑는다.
  3) 1)과 2)를 비교해서 trackman 근거가 train 기반 prior와 방향이 일치하는지
     검증(sanity check)한다.

출력: ./model/prior_table.json — 다음 단계(shrinkage 결합)에서 재사용.
"""

import json
import os

import numpy as np
import pandas as pd

from features import build_matchup_prior_from_df, fit_season_trend

DATA_DIR = "./open/data"
OUT_DIR = "./model"

PITCH_GROUPS = ["fastball", "breaking", "offspeed"]


def build_trackman_difficulty_index():
    """trackman_history.csv → 구종군별 물리 특성 요약 + 상대적 난이도 지수.

    1차 시도(무브먼트 "평균" 크기 기반)는 train.csv 기반 실제 성공률과
    방향이 맞지 않아 폐기했다. "평균 크기"는 그 구종의 전형적 궤적 특성
    (예: 패스트볼의 백스핀에 의한 상승 무브먼트)을 반영할 뿐, 제구의
    "일관성"과는 다른 개념이기 때문이다.

    2차 시도: 무브먼트/릴리스포인트의 "표준편차(변동성)"를 난이도 대리
    지표로 사용한다 — 투구마다 무브먼트나 릴리스 위치가 들쭉날쭉할수록
    원하는 곳에 반복적으로 던지기 어렵다는 의미로 해석 가능하다.
    (단, 이 값은 여러 투수가 섞인 집단 표준편차라 개인의 일관성이 아니라
    "그 구종군을 던질 때 전반적으로 결과가 얼마나 흩어지는가"에 가깝다는
    한계는 남아 있음)
    """
    cols = ["pitch_type_group", "rel_speed", "spin_rate",
            "induced_vert_break", "horz_break", "rel_height", "rel_side"]
    df = pd.read_csv(os.path.join(DATA_DIR, "trackman_history.csv"),
                      encoding="utf-8-sig", usecols=cols)
    df = df[df["pitch_type_group"].isin(PITCH_GROUPS)]

    df["movement_mag"] = np.sqrt(df["induced_vert_break"] ** 2 + df["horz_break"] ** 2)

    summary = df.groupby("pitch_type_group").agg(
        n=("movement_mag", "size"),
        rel_speed_mean=("rel_speed", "mean"),
        spin_rate_mean=("spin_rate", "mean"),
        movement_mag_mean=("movement_mag", "mean"),
        movement_mag_std=("movement_mag", "std"),
        rel_height_std=("rel_height", "std"),
        rel_side_std=("rel_side", "std"),
    )

    # 난이도 지수: 무브먼트 표준편차(변동성)를 0~1로 정규화 (클수록 어려움)
    mm_std = summary["movement_mag_std"]
    summary["difficulty_index_movement_std"] = (
        (mm_std - mm_std.min()) / (mm_std.max() - mm_std.min())
    )

    # 참고용: 릴리스포인트 변동성 기반 난이도 지수
    rel_std = summary["rel_height_std"] + summary["rel_side_std"]
    summary["difficulty_index_release_std"] = (
        (rel_std - rel_std.min()) / (rel_std.max() - rel_std.min())
    )

    return summary.loc[PITCH_GROUPS]


def build_train_group_prior():
    """train.csv → 투수별 마지막 시점 스냅샷으로 구종군별 prior 성공률/middle률 추정.

    asof_pitcher_n 이 가장 큰(=가장 많은 이력이 쌓인) 행을 투수별 대표값으로 쓰고,
    success_rate ~ fastball_rate*b1 + breaking_rate*b2 + offspeed_rate*b3
    형태의 절편 없는 가중최소제곱(WLS)으로 구종군별 평균 성공률을 추정한다.
    (fastball_rate + breaking_rate + offspeed_rate ≈ 1 이므로 절편 없이 적합
    가능하고, 각 계수가 곧 "그 구종군 위주로 던지는 투수의 평균 성공률" 해석이 됨)
    """
    cols = ["pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                         encoding="utf-8-sig", usecols=cols)

    # 투수별 asof_pitcher_n이 가장 큰(가장 누적된) 행만 남긴다 = 그 투수의 최종 스냅샷
    idx = train.groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    snap = train.loc[idx].dropna(subset=[
        "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ])
    # 표본이 너무 적은 투수는 노이즈가 크므로 제외
    snap = snap[snap["asof_pitcher_n"] >= 30]
    print(f"prior 추정에 사용한 투수 수: {len(snap)}")

    X = snap[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
              "asof_pitcher_offspeed_rate"]].to_numpy()
    w = snap["asof_pitcher_n"].to_numpy()

    def weighted_group_rate(y):
        # 가중최소제곱(절편 없음): (X'WX)^-1 X'Wy
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        return dict(zip(PITCH_GROUPS, beta))

    success_prior = weighted_group_rate(snap["asof_pitcher_success_rate"].to_numpy())
    middle_prior = weighted_group_rate(snap["asof_pitcher_middle_rate"].to_numpy())

    global_prior = {
        "success_rate": float(snap["asof_pitcher_success_rate"].mean()),
        "middle_rate": float(snap["asof_pitcher_middle_rate"].mean()),
    }

    return success_prior, middle_prior, global_prior


def build_matchup_prior():
    """투수×타자 쌍(pitcher_id, batter_id) 단위 제구 성공률 사전 계산 (최종 제출용).

    사용자 요청: "pitcher_id별 제구 성공률이 높은 batter_id를 찾아서 가중치로
    반영" — 즉 이 투수가 이 특정 타자를 상대할 때 유독 제구가 잘/안 됐는지를
    나타내는 피처. train.csv 전체(96,133개 고유 쌍, 쌍당 평균 15.3구, 47.5%가
    10구 이상)로 미리 집계해 lookup 테이블로 저장한다. (2025 평가 행은 이
    집계에 전혀 들어가지 않으므로 최종 제출 모델 기준으로는 누출 없음 —
    로컬 검증 시엔 별도로 만들어야 함. features.build_matchup_prior_from_df 참고)

    표본이 작은 쌍이 많으므로(중앙값 9구) 실제 사용 시 shrinkage로 투수의
    전체 성공률(asof_pitcher_success_rate_shrunk)과 섞어써야 함 — features.py
    add_matchup_feature() 참고.
    """
    df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                      encoding="utf-8-sig", usecols=["pitcher_id", "batter_id", "control_success"])
    matchup_prior = build_matchup_prior_from_df(df)
    print(f"매치업 쌍 개수: {len(matchup_prior)}")
    return matchup_prior


def main():
    print("=== 1) trackman_history.csv 기반 구종군 난이도 지수 ===")
    difficulty = build_trackman_difficulty_index()
    print(difficulty[["n", "movement_mag_mean", "movement_mag_std",
                       "rel_height_std", "rel_side_std",
                       "difficulty_index_movement_std",
                       "difficulty_index_release_std"]])

    print("\n=== 2) train.csv 기반 구종군별 prior 성공률/middle률 (WLS) ===")
    success_prior, middle_prior, global_prior = build_train_group_prior()
    print("prior success_rate by group:", success_prior)
    print("prior middle_rate  by group:", middle_prior)
    print("global prior:", global_prior)

    print("\n=== 3) sanity check: 난이도 지수와 prior 성공률의 관계 ===")
    for g in PITCH_GROUPS:
        print(f"  {g:9s} | 난이도(무브먼트std)={difficulty.loc[g, 'difficulty_index_movement_std']:.3f} "
              f"| 난이도(릴리스std)={difficulty.loc[g, 'difficulty_index_release_std']:.3f} "
              f"| prior_success={success_prior[g]:.4f} "
              f"| prior_middle={middle_prior[g]:.4f}")

    print("\n=== 4) 시즌 추세(season trend) 선형회귀 ===")
    # 최종 제출용 모델은 2019~2024 전체로 학습하므로, 여기서도 전체로 fit한다.
    # (walk-forward 검증 스크립트에서는 fold별로 별도로 다시 fit해서 사용함)
    season_target = pd.read_csv(os.path.join(DATA_DIR, "train.csv"),
                                 encoding="utf-8-sig", usecols=["season", "control_success"])
    trend = fit_season_trend(season_target)
    print(f"slope={trend['slope']:.6f}, intercept={trend['intercept']:.4f}")
    print(f"2025 예상 성공률(외삽): {trend['slope']*2025 + trend['intercept']:.4f}")

    print("\n=== 5) 투수×타자 매치업 prior ===")
    matchup_prior = build_matchup_prior()

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "trackman_difficulty_index_movement_std":
            difficulty["difficulty_index_movement_std"].to_dict(),
        "trackman_difficulty_index_release_std":
            difficulty["difficulty_index_release_std"].to_dict(),
        "trackman_group_stats": difficulty[[
            "rel_speed_mean", "spin_rate_mean", "movement_mag_mean",
            "movement_mag_std", "rel_height_std", "rel_side_std",
        ]].to_dict(orient="index"),
        "group_success_prior": success_prior,
        "group_middle_prior": middle_prior,
        "global_prior": global_prior,
        "season_trend": trend,
        "matchup_prior": matchup_prior,
    }
    with open(os.path.join(OUT_DIR, "prior_table.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n저장 완료: {OUT_DIR}/prior_table.json")


if __name__ == "__main__":
    main()
