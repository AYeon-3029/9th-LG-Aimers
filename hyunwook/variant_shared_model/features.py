"""학습(train)과 추론(inference) 양쪽에서 동일하게 재사용하는 피처 엔지니어링 모듈.

핵심 원칙: 여기서 하는 모든 계산은 "이미 갖고 있는 값(prior_table.json, 각 행의
asof_* 컬럼)"을 조회/조합하는 것뿐이다. test.csv 내부 행들끼리 통계를 내거나
새로 집계하는 행위는 절대 하지 않는다 (data_description.md 5)~6)절 누출 금지 규칙).

prior_table.json은 build_prior.py가 train.csv + trackman_history.csv로 미리
계산해 저장해 둔 결과물이다.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

PITCH_GROUPS = ["fastball", "breaking", "offspeed"]

# shrinkage 감쇠 상수: n(표본 수)이 이 값과 비슷할 때 개인 통계와 prior의 비중이
# 5:5가 된다.
# ⚠️ k=85로 튜닝했다가(로컬 2024 검증 708.12, k=50 대비 +17.6) 실제 리더보드
# 제출 결과 오히려 하락(758.37 -> 753.41)해서 50으로 되돌림. 그리드 서치 자체가
# 인접 k값끼리 ±7~15 수준으로 들쭉날쭉했는데, 그 노이즈 중 가장 높은 점 하나를
# 골랐던 게 단일 2024 검증 분할의 우연한 특성에 과적합된 것으로 추정 — 실제
# 미래(2025)엔 재현되지 않음. 이후 하이퍼파라미터는 단일 분할 최고점이 아니라
# 다중 fold(walk-forward) 평균으로 결정해야 함.
DEFAULT_K = 50


def load_prior_table(path="./model/prior_table.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _personal_prior(df, group_prior, global_prior_value):
    """투수별 구종 사용 비율(pitch mix)로 가중평균한 개인화 prior.

    pitch mix가 결측(cold-start)인 행은 global prior로 대체한다.
    """
    mix_cols = {
        "fastball": "asof_pitcher_fastball_rate",
        "breaking": "asof_pitcher_breaking_rate",
        "offspeed": "asof_pitcher_offspeed_rate",
    }
    weighted = sum(df[mix_cols[g]].fillna(0) * group_prior[g] for g in PITCH_GROUPS)
    mix_available = df[list(mix_cols.values())].notna().all(axis=1)
    return weighted.where(mix_available, global_prior_value)


def shrink(rate_col, n_col, prior, k=DEFAULT_K):
    """(n * 개인_rate + k * prior) / (n + k) 형태의 베이지안 shrinkage.

    rate_col이 결측(표본 0)이어도 n=0으로 계산하면 자연스럽게 prior만 남는다.
    """
    rate = rate_col.fillna(0)
    n = n_col.fillna(0)
    return (n * rate + k * prior) / (n + k)


def add_shrinkage_features(df, prior_table, k=DEFAULT_K):
    """asof_pitcher_success_rate / middle_rate 에 shrinkage를 적용한 새 컬럼 추가.

    원본 asof_* 컬럼은 그대로 두고, `_shrunk` 접미사가 붙은 컬럼을 새로 만든다.
    (원본을 남겨두면 모델이 "얼마나 shrink 됐는지"도 간접적으로 학습에 활용 가능)
    """
    df = df.copy()

    success_prior_by_pitcher = _personal_prior(
        df, prior_table["group_success_prior"], prior_table["global_prior"]["success_rate"])
    middle_prior_by_pitcher = _personal_prior(
        df, prior_table["group_middle_prior"], prior_table["global_prior"]["middle_rate"])

    df["asof_pitcher_success_rate_shrunk"] = shrink(
        df["asof_pitcher_success_rate"], df["asof_pitcher_n"], success_prior_by_pitcher, k)
    df["asof_pitcher_middle_rate_shrunk"] = shrink(
        df["asof_pitcher_middle_rate"], df["asof_pitcher_n"], middle_prior_by_pitcher, k)

    # 타자 쪽은 구종 mix 정보가 없으므로 global prior로만 shrink
    df["asof_batter_success_rate_shrunk"] = shrink(
        df["asof_batter_success_rate"], df["asof_batter_n"],
        prior_table["global_prior"]["success_rate"], k)
    df["asof_batter_middle_rate_shrunk"] = shrink(
        df["asof_batter_middle_rate"], df["asof_batter_n"],
        prior_table["global_prior"]["middle_rate"], k)

    # --- reverse/ball/strike rate: group/global prior로 shrink (표본 n은 asof_pitcher_n 공유) ---
    # (1차 시도에서 전체 피처가 늘며 RandomForest max_features='sqrt' 기본값 때문에
    #  hand_matchup/count_state 등의 중요도가 붕괴하는 문제가 있었으나, max_features=None으로
    #  바꾸면서 원인이 해소되어 다시 포함시킴 — train_v3.py 참고)
    df["asof_pitcher_ball_rate_shrunk"] = shrink(
        df["asof_pitcher_ball_rate"], df["asof_pitcher_n"],
        1 - prior_table["global_prior"]["success_rate"], k)  # 대략적 근사 prior
    df["asof_pitcher_strike_rate_shrunk"] = shrink(
        df["asof_pitcher_strike_rate"], df["asof_pitcher_n"],
        prior_table["global_prior"]["success_rate"], k)  # 대략적 근사 prior
    df["asof_pitcher_reverse_rate_shrunk"] = shrink(
        df["asof_pitcher_reverse_rate"], df["asof_pitcher_n"],
        1 - prior_table["global_prior"]["success_rate"], k)  # 대략적 근사 prior

    # --- cold-start 신뢰도 플래그 ---
    df["pitcher_is_cold_start"] = (df["asof_pitcher_n"].fillna(0) < 30).astype(int)
    df["batter_is_cold_start"] = (df["asof_batter_n"].fillna(0) < 30).astype(int)
    df["pitcher_recent_form_missing"] = df["asof_pitcher_prev1_game_success_rate"].isna().astype(int)

    # --- 최근 폼(prev1/3/5): 표본이 없으면 리그 prior가 아니라
    #     "그 투수의 shrink된 통산 성공률/middle률"로 fallback한다.
    #     (표본 크기를 알려주는 컬럼이 따로 없어 n=0/1로 근사: 결측이면 0표본으로 취급)
    for w_games in [1, 3, 5]:
        succ_col = f"asof_pitcher_prev{w_games}_game_success_rate"
        mid_col = f"asof_pitcher_prev{w_games}_game_middle_rate"
        is_missing = df[succ_col].isna()
        pseudo_n = (~is_missing).astype(int)  # 결측이면 0표본, 있으면 1표본 취급(약한 신뢰도)
        df[f"{succ_col}_shrunk"] = shrink(
            df[succ_col], pseudo_n, df["asof_pitcher_success_rate_shrunk"], k=3)
        df[f"{mid_col}_shrunk"] = shrink(
            df[mid_col], pseudo_n, df["asof_pitcher_middle_rate_shrunk"], k=3)

    return df


SHRINKAGE_FEATURE_NAMES = [
    "asof_pitcher_success_rate_shrunk",
    "asof_pitcher_middle_rate_shrunk",
    "asof_batter_success_rate_shrunk",
    "asof_batter_middle_rate_shrunk",
    "asof_pitcher_ball_rate_shrunk",
    "asof_pitcher_strike_rate_shrunk",
    "asof_pitcher_reverse_rate_shrunk",
    "asof_pitcher_prev1_game_success_rate_shrunk",
    "asof_pitcher_prev3_game_success_rate_shrunk",
    "asof_pitcher_prev5_game_success_rate_shrunk",
    "asof_pitcher_prev1_game_middle_rate_shrunk",
    "asof_pitcher_prev3_game_middle_rate_shrunk",
    "asof_pitcher_prev5_game_middle_rate_shrunk",
    "pitcher_is_cold_start",
    "batter_is_cold_start",
    "pitcher_recent_form_missing",
]


# =======================
# 4단계 — Sample.md 조합 피처
# =======================
# 전부 "그 행 안에서만" 계산되는 단순 조합/연산이라 통계 집계가 없고, 따라서
# test.csv 내부 행 간 통계 금지 규칙과 무관하다 (누출 위험 없음). 새로운
# 카디널리티도 늘어나지 않는다 — 각 조합 피처는 원본 컬럼의 값 범위 안에서만
# 만들어지므로 test 시점에 처음 보는 조합이 나올 수 없다.

CATEGORICAL_COMBO_COLS = ["count_state", "hand_matchup", "outs_base_state"]


def add_combo_features(df):
    """Sample.md 조합 피처.

    시도했던 것: 개별 피처 중요도가 0.002 미만인 11개(pitcher_ahead,
    batter_ahead, full_count, two_strike, three_ball, abs_score_diff,
    pitcher_team_leading, pitcher_team_trailing, tie_game, close_game,
    blowout_game)를 제거해봤으나, 오히려 Validation Score가 463.65 → 446.96로
    떨어졌다 (RandomForest는 분기마다 피처를 무작위로 일부만 후보로 뽑기 때문에,
    전체 피처 개수가 줄면 개별 중요도가 낮았던 피처들이 만들던 미세한 상호작용
    효과까지 함께 사라지는 것으로 보임). 그래서 18개 전부 유지하기로 되돌렸다.
    """
    df = df.copy()

    # --- 카운트 조합 ---
    df["count_state"] = (
        df["balls_before"].astype(str) + "-" + df["strikes_before"].astype(str)
    )
    df["pitcher_ahead"] = (df["strikes_before"] > df["balls_before"]).astype(int)
    df["batter_ahead"] = (df["balls_before"] > df["strikes_before"]).astype(int)
    df["full_count"] = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    df["two_strike"] = (df["strikes_before"] == 2).astype(int)
    df["three_ball"] = (df["balls_before"] == 3).astype(int)

    # --- 아웃카운트 × 주자 상황 ---
    df["outs_base_state"] = (
        df["outs_before"].astype(str) + "_" + df["base_state"].astype(str)
    )

    # --- 투수/타자 좌우 매치업 ---
    df["hand_matchup"] = df["pitcher_hand"].astype(str) + "_" + df["batter_hand"].astype(str)

    # --- 점수차 ---
    df["abs_score_diff"] = df["score_diff_pitcher_team"].abs()
    df["pitcher_team_leading"] = (df["score_diff_pitcher_team"] > 0).astype(int)
    df["pitcher_team_trailing"] = (df["score_diff_pitcher_team"] < 0).astype(int)
    df["tie_game"] = (df["score_diff_pitcher_team"] == 0).astype(int)
    df["close_game"] = (df["abs_score_diff"] <= 2).astype(int)
    df["blowout_game"] = (df["abs_score_diff"] >= 5).astype(int)

    # --- 이닝 × 점수차 / 기대승률 / Li interaction ---
    df["inning_x_score_diff"] = df["inning"] * df["score_diff_pitcher_team"]
    df["inning_x_li"] = df["inning"] * df["li"]
    df["win_expectancy_diff"] = df["home_win_expectancy"] - df["away_win_expectancy"]
    df["inning_x_win_expectancy_diff"] = df["inning"] * df["win_expectancy_diff"]
    # 시도: inning × home_win_expectancy / away_win_expectancy를 diff와 별개로
    # 추가해봤으나(708.12 -> 700.96, -7.16) 지금까지 관측된 노이즈 수준(±7~15)과
    # 비슷해 확실한 이득이 없다고 판단, 채택하지 않음.

    # 시도: pitcher_season, pitcher×batter_hand, Li interaction 5개를 추가해
    # 로컬 검증에서 710.00(+1.88)까지 나왔으나, 실제 리더보드 제출 결과 오히려
    # 하락(758.37 -> 753.41)해서 전부 제거함. 특히 pitcher_season은 2025 시즌
    # 자체가 학습에 없어 실제 제출 시엔 전부 미확인 범주(-1)로 뭉개지는데도
    # 로컬(2024도 마찬가지로 미확인 범주)에서 소폭 +가 나왔던 것 자체가
    # 단일 분할 노이즈였을 가능성이 높음.

    return df


COMBO_FEATURE_NAMES = CATEGORICAL_COMBO_COLS + [
    "pitcher_ahead", "batter_ahead", "full_count", "two_strike", "three_ball",
    "abs_score_diff", "pitcher_team_leading", "pitcher_team_trailing",
    "tie_game", "close_game", "blowout_game",
    "inning_x_score_diff", "inning_x_li", "win_expectancy_diff",
    "inning_x_win_expectancy_diff",
]


# =======================
# 5단계 — 시즌 추세 보정 피처
# =======================
# walk-forward 검증에서 발견: control_success 성공률이 2019(56.5%)→2024(48.6%)로
# 꾸준히 하락하는 추세가 있는데, RandomForest는 학습 때 본 적 없는 season 값으로
# 이 선형적 추세를 외삽하지 못한다. 그래서 추세 자체를 트리 밖에서 선형회귀로
# 미리 계산해 "이 시즌의 예상 성공률"을 피처로 만들어 넣어준다 — 트리는 이제
# "외삽"할 필요 없이, 이미 계산된 값을 그냥 참고만 하면 된다.

TREND_FEATURE_NAMES = ["season_trend_success_prior"]


def fit_season_trend(df, season_col="season", target_col="control_success"):
    """시즌별 평균 성공률에 대한 단순 선형회귀 (season -> rate).

    누출 방지: 반드시 "그 시점까지의 학습 데이터"만 넣어서 호출해야 한다.
    (예: 2023을 검증할 땐 2019~2022로만 fit)
    """
    season_rate = df.groupby(season_col)[target_col].mean()
    x = season_rate.index.to_numpy(dtype=float)
    y = season_rate.to_numpy(dtype=float)
    # 최소제곱 직선 fit: y = a*x + b
    a, b = np.polyfit(x, y, deg=1)
    return {"slope": float(a), "intercept": float(b)}


def add_season_trend_feature(df, trend, season_col="season"):
    df = df.copy()
    pred = trend["slope"] * df[season_col] + trend["intercept"]
    df["season_trend_success_prior"] = pred.clip(0, 1)
    return df


# =======================
# 6단계 — 투수×타자 매치업 피처
# =======================
# "이 투수가 이 특정 타자를 상대할 때 유독 제구가 잘/안 됐는가"를 나타내는 피처.
# prior_table["matchup_prior"]는 train.csv 전체로 (pitcher_id, batter_id) 쌍별
# 성공률/표본수를 미리 집계해 둔 lookup 테이블 (build_prior.py 참고).
# 쌍당 표본이 작은 경우가 많으므로(중앙값 9구) 투수의 전체 shrink된 성공률로
# shrinkage한다 — asof_pitcher_success_rate_shrunk가 먼저 계산되어 있어야 함
# (add_shrinkage_features 다음에 호출).

MATCHUP_FEATURE_NAMES = ["pitcher_batter_matchup_success_rate_shrunk"]
MATCHUP_K = 10  # 매치업 쌍의 중앙값 표본 수(9)와 비슷하게 설정


def build_matchup_prior_from_df(df, target_col="control_success"):
    """(pitcher_id, batter_id) 쌍별 성공률/표본수 lookup 테이블 생성.

    ⚠️ 누출 주의: 이 함수에 넘기는 df에 검증/평가 대상 행의 정답이 섞여 있으면,
    그 행 자신의 결과가 자기 피처 계산에 들어가는 target leakage가 발생한다
    (특히 표본 n=1인 쌍은 사실상 정답이 그대로 노출됨). 따라서:
      - 최종 제출용 prior_table.json: train.csv 전체(2019~2024)로 호출 — 2025
        평가 행은 절대 이 함수 호출에 들어가지 않으므로 안전.
      - 로컬 검증(예: 2024를 떼어 평가): 반드시 2019~2023만 넘겨서 별도로
        다시 만들어야 함 (features.fit_season_trend와 동일한 원칙).
    """
    grouped = df.groupby(["pitcher_id", "batter_id"])[target_col].agg(["mean", "count"])
    return {
        f"{p}_{b}": {"n": int(row["count"]), "success_rate": float(row["mean"])}
        for (p, b), row in grouped.iterrows()
    }


def add_matchup_feature(df, matchup_prior, k=MATCHUP_K):
    """검증/추론용. matchup_prior가 이 df의 행을 포함하지 않는 별도 데이터로
    만들어졌을 때만 안전하다 (예: production prior_table은 train 전체로 만들고
    test.csv에 적용 — test 행은 애초에 그 테이블 구성에 안 들어갔으므로 안전.
    로컬 검증도 마찬가지로 2019~2023으로 만든 테이블을 2024에 적용하는 건 안전).
    학습 데이터 자기 자신에 적용할 땐 반드시 add_matchup_feature_loo를 써야 한다.
    """
    df = df.copy()
    keys = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)

    n_map = {key: v["n"] for key, v in matchup_prior.items()}
    rate_map = {key: v["success_rate"] for key, v in matchup_prior.items()}
    matchup_n = keys.map(n_map).fillna(0)
    matchup_rate = keys.map(rate_map)  # 없는 쌍은 NaN -> shrink()에서 n=0으로 자연 처리

    df["pitcher_batter_matchup_success_rate_shrunk"] = shrink(
        matchup_rate, matchup_n, df["asof_pitcher_success_rate_shrunk"], k)
    return df


def add_matchup_feature_loo(df, target_col="control_success", k=MATCHUP_K):
    """학습 데이터 자기 자신에 매치업 피처를 붙일 때 쓰는 leave-one-out 버전.

    표본이 매우 작은 쌍이 많아(중앙값 9구, n=1인 쌍도 3.9%) 그냥 자기 쌍의
    평균을 피처로 쓰면 자기 정답이 그대로 노출되는 셈이라 (n=1이면 feature ==
    자기 target), 반드시 "이 쌍의 다른 행들"만으로 계산해야 한다.

    데이터에 exact한 투구 순서/시각이 없어 완벽한 시간순 walk-forward는
    불가능하므로, 대신 "자기 자신만 제외한 나머지 전체"로 계산하는
    leave-one-out 방식을 쓴다 — 표준적인 CV-safe target encoding 기법이며,
    최소한 가장 심각한 형태의 누출(자기 자신 포함)은 완전히 제거한다.
    """
    df = df.copy()
    grp = df.groupby(["pitcher_id", "batter_id"])[target_col]
    total_n = grp.transform("count")
    total_sum = grp.transform("sum")
    loo_n = total_n - 1
    loo_sum = total_sum - df[target_col]
    loo_rate = loo_sum / loo_n.replace(0, np.nan)

    df["pitcher_batter_matchup_success_rate_shrunk"] = shrink(
        loo_rate.astype(float), loo_n.clip(lower=0), df["asof_pitcher_success_rate_shrunk"], k)
    return df


# --- Li(Leverage Index) 가중 버전 ---
# 단순 빈도 평균 대신, 각 투구를 그 투구가 일어난 상황의 중요도(li)로 가중해서
# 평균을 낸다 — "이 투수-타자 매치업이 중요한 상황에서 유독 잘/안 됐는가"를
# 반영. li가 클수록 그 투구의 결과가 가중평균에 더 크게 기여한다.
# 가중치 총합(sum of li)을 "유효 표본 수"로 써서 shrinkage 강도를 정한다.

MATCHUP_LI_FEATURE_NAMES = ["pitcher_batter_matchup_li_weighted_success_rate_shrunk"]


def build_matchup_li_prior_from_df(df, target_col="control_success", li_col="li"):
    """(pitcher_id, batter_id) 쌍별 Li 가중 성공률 lookup 테이블.

    build_matchup_prior_from_df와 동일한 누출 규칙 적용 — 검증/평가 행이
    이 df에 섞여 있으면 안 됨.
    """
    w = df[li_col].clip(lower=0.01)  # li=0인 완전 무의미한 상황이 가중치 0이 되어
                                       # 분모를 0으로 만들지 않도록 최소값 보정
    weighted_success = df[target_col] * w
    grouped = pd.DataFrame({
        "pitcher_id": df["pitcher_id"], "batter_id": df["batter_id"],
        "w": w, "ws": weighted_success,
    }).groupby(["pitcher_id", "batter_id"]).sum()
    return {
        f"{p}_{b}": {"total_li": float(row["w"]), "li_weighted_success_rate": float(row["ws"] / row["w"])}
        for (p, b), row in grouped.iterrows()
    }


def add_matchup_li_feature(df, matchup_li_prior, k=MATCHUP_K):
    """검증/추론용 lookup (add_matchup_feature와 동일한 안전 규칙)."""
    df = df.copy()
    keys = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)
    w_map = {key: v["total_li"] for key, v in matchup_li_prior.items()}
    rate_map = {key: v["li_weighted_success_rate"] for key, v in matchup_li_prior.items()}
    total_li = keys.map(w_map).fillna(0)
    li_rate = keys.map(rate_map)

    df["pitcher_batter_matchup_li_weighted_success_rate_shrunk"] = shrink(
        li_rate, total_li, df["asof_pitcher_success_rate_shrunk"], k)
    return df


def add_matchup_li_feature_loo(df, target_col="control_success", li_col="li", k=MATCHUP_K):
    """학습 데이터 자기 자신용 leave-one-out Li 가중 버전."""
    df = df.copy()
    w = df[li_col].clip(lower=0.01)
    weighted_success = df[target_col] * w

    grp_w = w.groupby([df["pitcher_id"], df["batter_id"]])
    grp_ws = weighted_success.groupby([df["pitcher_id"], df["batter_id"]])
    total_w = grp_w.transform("sum")
    total_ws = grp_ws.transform("sum")

    loo_w = total_w - w
    loo_ws = total_ws - weighted_success
    loo_rate = loo_ws / loo_w.replace(0, np.nan)

    df["pitcher_batter_matchup_li_weighted_success_rate_shrunk"] = shrink(
        loo_rate.astype(float), loo_w.clip(lower=0), df["asof_pitcher_success_rate_shrunk"], k)
    return df


# =======================
# 7단계 — CatBoost 네이티브 범주형 처리 래퍼
# =======================
# CatBoost는 OrdinalEncoder로 미리 수치화하지 않고 원본 범주형 컬럼(문자열)을
# 그대로 받아 ordered target encoding을 내부적으로 수행하는데, 이게 LightGBM의
# OrdinalEncoder 방식보다 이 데이터에서 훨씬 강력했다(다중 fold 평균 1473.75 ->
# 1565.34, +91.6 — train_v14_blend.py 실험). script.py의 model.predict_proba(X)
# 인터페이스(X: FEATURES 컬럼을 가진 DataFrame)를 그대로 유지하면서 CatBoost의
# 전처리(수치형 median 대치 + 범주형 문자열 캐스팅)를 감싸는 래퍼.
class CatBoostWrapper:
    """{model, features, prior_table} 번들에 그대로 넣어 pickle 가능한 CatBoost 래퍼.

    fit()에서 NUM_COLS의 median을 저장해두고, predict_proba() 호출 시에도
    학습 때와 동일한 median으로 결측을 채운다(school test 시점 재계산 없음 —
    누출 방지 원칙 유지).
    """

    def __init__(self, cat_cols, num_cols, **catboost_params):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.catboost_params = catboost_params
        self.model = CatBoostClassifier(**catboost_params)
        self.medians_ = None

    def _transform(self, X):
        X = X.copy()
        X[self.num_cols] = X[self.num_cols].fillna(self.medians_)
        for c in self.cat_cols:
            X[c] = X[c].astype(str)
        return X

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        self.medians_ = X[self.num_cols].median()
        Xt = self._transform(X)
        cat_idx = [Xt.columns.get_loc(c) for c in self.cat_cols]
        train_pool = Pool(Xt, y, cat_features=cat_idx)

        eval_pool = None
        if eval_set is not None:
            X_eval, y_eval = eval_set
            Xe_t = self._transform(X_eval)
            eval_pool = Pool(Xe_t, y_eval, cat_features=cat_idx)

        fit_kwargs = {}
        if eval_pool is not None:
            fit_kwargs["eval_set"] = eval_pool
            fit_kwargs["use_best_model"] = True
        if early_stopping_rounds is not None:
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
        self.model.fit(train_pool, **fit_kwargs)
        return self

    def predict_proba(self, X):
        Xt = self._transform(X)
        return self.model.predict_proba(Xt)

    @property
    def best_iteration_(self):
        return self.model.get_best_iteration()


# =======================
# 8단계 — 리그(game_type)별 "직전 시즌" shrinkage prior (팀원 아이디어 결합)
# =======================
# 팀원(ayeon) 파이프라인에서 검증된 아이디어: 우리 shrinkage(정적/전역 prior)와
# 달리, prior 자체를 "직전 시즌의 같은 game_type 평균"으로 매년 갱신한다.
# leak-safe(그 시점까지의 학습 데이터로만 계산) + 레짐(R/F) 조건부라, 정적
# 전역 prior보다 그 해의 분포에 더 가깝다. 우리 기존 shrinkage 피처와는
# 별개로 "추가 피처"로만 넣는다 — 모델 구조(공유/분리)와 무관하게 결합 가능.
def compute_league_season_means(df, league_col="game_type", season_col="season",
                                 target_col="control_success"):
    """학습 데이터(is_train 마스크로 걸러진 것)만으로 (game_type, season)별
    평균 성공률을 계산 — 반드시 그 시점까지의 데이터만 넣어서 호출할 것."""
    return df.groupby([league_col, season_col])[target_col].mean().to_dict()


def add_league_shrunk_feature(df, season_means, league_col="game_type", season_col="season",
                               rate_col="asof_pitcher_success_rate", n_col="asof_pitcher_n", k=100):
    """각 행의 (game_type, season-1) prior로 asof_pitcher_success_rate를 shrink.
    직전 시즌 데이터가 없는 경우(첫 시즌)는 원본 rate를 그대로 사용."""
    df = df.copy()
    key = list(zip(df[league_col], df[season_col] - 1))
    prior = np.array([season_means.get(k2, np.nan) for k2 in key], dtype=np.float64)
    n = df[n_col].to_numpy(dtype=np.float64)
    raw = df[rate_col].to_numpy(dtype=np.float64)
    shrunk = (n * raw + k * prior) / (n + k)
    df["league_season_shrunk_pitcher_rate"] = np.where(np.isnan(prior), raw, shrunk)
    return df


LEAGUE_SHRUNK_FEATURE_NAMES = ["league_season_shrunk_pitcher_rate"]
