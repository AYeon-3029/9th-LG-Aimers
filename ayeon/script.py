"""제출용 추론 스크립트 - R(1군)/F(퓨처스) 완전 분리 모델을 game_type으로
라우팅해 예측한다. 각 리그마다 LightGBM + XGBoost + CatBoost + DCNv2 균등
가중(1/4) 앙상블.

평가 서버가 자동 실행한다. model/ 아래 lgb/xgb/cat_model_{R,F}.*와
dcn_model_{R,F}_seed{0..N-1}.pt(R은 3개, F는 5개 - DCNv2는 시드 편차가 커서
독립 재학습 후 평균낸다, common.py 근거)를 각자의 네이티브 포맷으로 불러와
data/test.csv를 행별 game_type에 따라 해당 리그 앙상블로 예측하고,
output/submission.csv 로 생성한다. game_type 라우팅은 그 행 자체의 입력
변수만 사용하므로 (다른 행이나 테스트셋 분포를 이용한 보정이 아니므로)
rule.md 4항의 "각 행은 독립적인 예측 대상" 원칙을 위반하지 않는다.

shrunk_pitcher_rate는 model/shrink_season_means.csv(학습 데이터로만 계산된
(game_type, season) 평균 테이블)를 읽어 각 행 자신의 (season-1) 값을 leak-safe
prior로 사용한다 - 이 테이블도 그 행 자체(및 학습 시 확정된 상수)만으로 결정되므로
같은 원칙을 위반하지 않는다. R의 트리 3종만 이 피처를 실제로 쓴다(FEATURES) -
F는 검증셋을 여러 구간으로 쪼개보니 shrinkage 효과가 구간마다 뒤집혀서 철회했다
(FEATURES_F, common.py/train.py 근거 참고). shrunk_pitcher_rate 자체는 모든
행에 대해 계산해 두지만 F 쪽 모델 입력에서는 빠진다.

트리 세 모델은 결측치를 학습 시 결정된 분기 방향으로 그대로 처리하므로 asof_*의
cold-start NaN을 별도로 대치하지 않는다. DCNv2는 표준화 후 0으로 채우고 결측
지시 피처를 덧붙인다(학습 때와 동일하게 model/dcn_stats_{R,F}.npz의 통계 사용).

Platt scaling은 시도했다가 뺐다 - 연도별로 개별 적합하면 계수 부호가
뒤집히고 한 해에 적합해 다른 해에 적용하면 오히려 악화됐다(common.py 근거
참고, LB 772.38 -> 763 회귀의 유력한 원인 중 하나로 의심됨). 4-way 앙상블
확률을 보정 없이 그대로 제출한다.

zip 내부에서 로컬 모듈 import 없이 단독 실행돼야 하므로 common.py를 참조하지
않고 학습 때와 동일한 피처 정의/DCNv2 구조를 이 파일 안에 직접 둔다.
(train.py/common.py의 정의와 반드시 일치해야 함)
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TEST_PATH = BASE_DIR / "data" / "test.csv"
OUTPUT_PATH = BASE_DIR / "output" / "submission.csv"

ASOF_FEATURES = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
MATCHUP_FEATURES = [
    "matchup_1_1",
    "matchup_1_2",
    "matchup_2_1",
    "matchup_2_2",
    "same_hand",
]
SITUATIONAL_FEATURES = [
    "balls_before",
    "strikes_before",
    "inning",
    "home_win_expectancy",
    "num_runners_on",
    "li",
]
SHRUNK_FEATURES = ["shrunk_pitcher_rate"]
SHRINK_K = 100

# **772.38 재현 통제 실험 중 - R shrinkage도 임시로 뺐다** (common.py 근거
# 참고: F를 되돌려도 LB가 763대에서 안 움직여서, 남은 차이인 R shrinkage를
# 마저 빼고 772.38과 최대한 동일한 조건으로 제출해 원인을 가른다). R/F 둘 다
# shrinkage 없이 같은 피처셋(FEATURES)을 쓴다 - DCN_FEATURES도 자동으로
# 동일해진다.
FEATURES = ASOF_FEATURES + MATCHUP_FEATURES + SITUATIONAL_FEATURES
FEATURES_F = FEATURES
DCN_FEATURES = FEATURES


def add_matchup_features(df):
    df = df.copy()
    for a in (1, 2):
        for b in (1, 2):
            df[f"matchup_{a}_{b}"] = (
                (df["pitcher_hand"] == a) & (df["batter_hand"] == b)
            ).astype(np.float64)
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64)
    return df


def add_shrunk_pitcher_feature(df, season_means, k=SHRINK_K):
    """train.py/common.py의 add_shrunk_pitcher_feature와 동일 로직. season_means는
    model/shrink_season_means.csv(학습 데이터로만 계산됨)에서 로드한 것을 쓴다 -
    각 행은 자기 (season - 1)의 (game_type) 평균을 leak-safe prior로 사용한다."""
    df = df.copy()
    key = list(zip(df["game_type"], df["season"] - 1))
    prior = np.array([season_means.get(k2, np.nan) for k2 in key], dtype=np.float64)
    n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    raw = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    shrunk = (n * raw + k * prior) / (n + k)
    shrunk[np.isnan(raw) | np.isnan(prior)] = np.nan
    df["shrunk_pitcher_rate"] = shrunk
    return df


# DCNv2 - train.py/common.py의 정의와 반드시 일치해야 함 (구조가 다르면
# state_dict 로딩이 깨진다). 시드별 편차가 커서(common.py 근거) N_SEEDS개를
# 독립 재학습해 예측을 평균 낸다 - 772.38 재현 실험으로 잠깐 1(단일시드)로
# 내렸다가 F의 단일시드 불안정성(bss_F 33~514)이 재발해 무효 결과를 낸 걸
# 확인하고 원복했다(common.py 근거). common.py의 DCN_R_N_SEEDS/
# DCN_F_N_SEEDS와 반드시 같이 바꿔야 한다(model/dcn_model_*_seed*.pt 파일
# 개수와도 일치해야 함 - train.py가 실제로 몇 개를 저장했는지 model/ 확인).
DCN_R_PARAMS = dict(cross_layers=2, deep_dims=(256, 128), dropout=0.1)
# 2026-08-22 3->5시드로 확장했다가 2026-08-23 3으로 철회 (common.py
# DCN_R_N_SEEDS 근거 참고) - DCN 단독으로는 3구간 전부 개선(+11/+14/+3)이었지만
# 4모델 블렌드로 재확인하니 유효한 두 구간에서 일관되게 -2였다(다양성 손실).
DCN_R_N_SEEDS = 3
# 2026-08-22 재탐색 채택했다가 2026-08-23 철회 (common.py DCN_F_PARAMS 근거
# 참고) - cross_layers=0은 DCN 단독 성능(366->462)은 개선했지만 4모델 블렌드
# bss_F는 531->526로 악화시켰다(다양성 손실, 볼록결합 재가중치로도 회복 안 됨).
# 원래 값으로 복귀.
DCN_F_PARAMS = dict(cross_layers=2, deep_dims=(32, 16), dropout=0.2)
DCN_F_N_SEEDS = 5

# pitcher/batter 저차원 임베딩 - 5번째 앙상블 멤버(R 2026-08-23 채택, F
# 2026-08-24 채택, common.py EmbedMLP/EMBED_R_*/EMBED_F_* 근거 참고). R 3구간
# walk-forward, 5-way 블렌드 기준 4-way 대비 델타 +15/+7/+4(3구간 전부 양수,
# optuna 트리 파라미터와 조합해도 상쇄 없음: combined_test.py). F는 whole-split
# bss_F 531->543(+12, 독립된 두 실행에서 재현). R/F 설정은 완전히 독립적이다 -
# 이름에 반드시 R/F 접두사를 유지할 것(과거 LGB_BASE_PARAMS가 F로 새어들어갔던
# 사고를 EMBED에서도 반복하지 않기 위함).
EMBED_R_DIM = 8
EMBED_R_HIDDEN = (64, 32)
EMBED_R_DROPOUT = 0.1
EMBED_R_N_SEEDS = 3

EMBED_F_DIM = 8
EMBED_F_HIDDEN = (16, 8)
EMBED_F_DROPOUT = 0.2
EMBED_F_N_SEEDS = 5


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    def __init__(self, in_dim, cross_layers=3, deep_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])
        deep, prev = [], in_dim
        for d in deep_dims:
            deep += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        self.deep = nn.Sequential(*deep)
        self.out = nn.Linear(in_dim + deep_dims[-1], 1)

    def forward(self, x):
        xc = x
        for layer in self.cross_layers:
            xc = layer(x, xc)
        xd = self.deep(x)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


class EmbedMLP(nn.Module):
    """train.py/common.py의 EmbedMLP와 반드시 동일 구조. index 0=unknown(미확인
    ID, 예: 2025 신인). embed_dim/hidden/dropout은 기본값 없이 호출부에서 항상
    명시적으로 전달한다(R/F가 다른 값을 쓰므로 암묵적 기본값에 의존하지 않음)."""

    def __init__(self, n_raw, n_pitchers, n_batters, embed_dim, hidden, dropout):
        super().__init__()
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, embed_dim)
        self.batter_emb = nn.Embedding(n_batters + 1, embed_dim)
        layers, prev = [], n_raw + 2 * embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_raw, pid, bid):
        x = torch.cat([x_raw, self.pitcher_emb(pid), self.batter_emb(bid)], dim=1)
        return self.mlp(x).squeeze(-1)


def load_id_vocab(suffix, id_col):
    """train.py가 저장한 model/embed_{pitcher,batter}_vocab_{suffix}.csv를
    dict(raw_id -> index)로 로드한다. 어휘는 학습 데이터로만 만들어졌으므로
    (train.py build_id_vocab 근거) 여기서 test.csv 값을 섞어 재계산하지 않는다."""
    name = "pitcher" if id_col == "pitcher_id" else "batter"
    vocab_df = pd.read_csv(MODEL_DIR / f"embed_{name}_vocab_{suffix}.csv")
    return dict(zip(vocab_df[id_col], vocab_df["idx"]))


def encode_ids(series, vocab):
    return series.map(vocab).fillna(0).to_numpy(dtype=np.int64)


def predict_embed(X, suffix, n_seeds, embed_dim, hidden, dropout):
    """X는 그 리그의 FEATURES + pitcher_id/batter_id 컬럼을 가진 DataFrame.
    DCN과 동일한 전처리 통계(dcn_stats_{suffix}.npz)를 재사용한다(같은
    피처셋, common.py 근거). embed_dim/hidden/dropout은 호출부(R/F)에서 명시
    전달 - EMBED_R_*/EMBED_F_*가 암묵적으로 섞이지 않게 한다."""
    stats_npz = np.load(str(MODEL_DIR / f"dcn_stats_{suffix}.npz"))
    stats = {"mean": stats_npz["mean"], "std": stats_npz["std"], "miss_cols": stats_npz["miss_cols"]}
    Xn = torch.tensor(apply_dcn_preprocessing(X[DCN_FEATURES], stats))

    pitcher_vocab = load_id_vocab(suffix, "pitcher_id")
    batter_vocab = load_id_vocab(suffix, "batter_id")
    pid = torch.tensor(encode_ids(X["pitcher_id"], pitcher_vocab))
    bid = torch.tensor(encode_ids(X["batter_id"], batter_vocab))

    preds = []
    for seed in range(n_seeds):
        model = EmbedMLP(Xn.shape[1], len(pitcher_vocab), len(batter_vocab), embed_dim, hidden, dropout)
        model.load_state_dict(torch.load(str(MODEL_DIR / f"embed_model_{suffix}_seed{seed}.pt"), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(Xn, pid, bid)).numpy())
    return np.mean(preds, axis=0)


DCN_CLIP = 5.0


def apply_dcn_preprocessing(X_df, stats):
    """train.py/common.py와 반드시 동일 로직. X_df는 DCN_FEATURES 컬럼만 가진
    DataFrame이어야 한다(shrunk_pitcher_rate 제외 - predict_dcn에서 슬라이싱).
    클리핑 없으면 F처럼 표본이 작은 subset에서 극단적 z-score가 CrossLayer를
    통해 학습을 붕괴시킨다."""
    X = X_df.to_numpy(dtype=np.float64)
    nan_mask = np.isnan(X)
    Xn = (X - stats["mean"]) / stats["std"]
    Xn = np.clip(Xn, -DCN_CLIP, DCN_CLIP)
    Xn[nan_mask] = 0.0
    miss_cols = stats["miss_cols"]
    if len(miss_cols) > 0:
        miss_ind = nan_mask[:, miss_cols].astype(np.float64)
        Xn = np.concatenate([Xn, miss_ind], axis=1)
    return Xn.astype(np.float32)


def predict_dcn(X, suffix, dcn_params, n_seeds):
    """X는 그 리그의 트리 피처셋(FEATURES 또는 FEATURES_F) 컬럼을 가진
    DataFrame - DCN_FEATURES로 다시 슬라이싱한다(R은 shrinkage를 뺀 부분집합,
    F는 이미 같은 리스트라 항등 연산). n_seeds개의 독립 학습된 state_dict를
    각각 예측한 뒤 평균 낸다."""
    stats_npz = np.load(str(MODEL_DIR / f"dcn_stats_{suffix}.npz"))
    stats = {"mean": stats_npz["mean"], "std": stats_npz["std"], "miss_cols": stats_npz["miss_cols"]}
    Xn = torch.tensor(apply_dcn_preprocessing(X[DCN_FEATURES], stats))

    preds = []
    for seed in range(n_seeds):
        model = DCNv2(Xn.shape[1], **dcn_params)
        model.load_state_dict(torch.load(str(MODEL_DIR / f"dcn_model_{suffix}_seed{seed}.pt"), map_location="cpu"))
        model.eval()
        with torch.no_grad():
            preds.append(torch.sigmoid(model(Xn)).numpy())
    return np.mean(preds, axis=0)


def predict_group(df_slice, suffix, dcn_params, dcn_n_seeds, feat_list, embed_params=None):
    """df_slice는 feat_list 컬럼(+ embed_params가 주어지면 pitcher_id/batter_id도)을
    가진 DataFrame. embed_params=None이면 기존 4-way, 아니면(R 2026-08-23 채택,
    F 2026-08-24 채택) 임베딩을 5번째 멤버로 추가한 5-way 평균을 반환한다.
    embed_params=(n_seeds, embed_dim, hidden, dropout)."""
    X = df_slice[feat_list]
    X_np = X.to_numpy(dtype=np.float64)

    lgb_booster = lgb.Booster(model_file=str(MODEL_DIR / f"lgb_model_{suffix}.txt"))
    p_lgb = lgb_booster.predict(X_np)

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / f"xgb_model_{suffix}.json"))
    p_xgb = xgb_model.predict_proba(X)[:, 1]

    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_DIR / f"cat_model_{suffix}.cbm"))
    p_cat = cat_model.predict_proba(X)[:, 1]

    p_dcn = predict_dcn(X, suffix, dcn_params, dcn_n_seeds)

    if embed_params is None:
        return (p_lgb + p_xgb + p_cat + p_dcn) / 4

    embed_n_seeds, embed_dim, embed_hidden, embed_dropout = embed_params
    p_embed = predict_embed(df_slice, suffix, embed_n_seeds, embed_dim, embed_hidden, embed_dropout)
    return (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5


def main():
    cols = (
        ["row_id", "game_type", "season"] + ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["pitcher_hand", "batter_hand", "pitcher_id", "batter_id"]
    )
    df = pd.read_csv(TEST_PATH, usecols=cols)
    df = add_matchup_features(df)

    means_df = pd.read_csv(MODEL_DIR / "shrink_season_means.csv")
    season_means = {
        (row.game_type, row.season): row.mean for row in means_df.itertuples()
    }
    df = add_shrunk_pitcher_feature(df, season_means)

    proba = np.empty(len(df), dtype=np.float64)
    is_R = (df["game_type"] == "R").to_numpy()
    embed_params_R = (EMBED_R_N_SEEDS, EMBED_R_DIM, EMBED_R_HIDDEN, EMBED_R_DROPOUT)
    embed_params_F = (EMBED_F_N_SEEDS, EMBED_F_DIM, EMBED_F_HIDDEN, EMBED_F_DROPOUT)
    groups = (
        ("R", is_R, DCN_R_PARAMS, DCN_R_N_SEEDS, FEATURES, embed_params_R),
        ("F", ~is_R, DCN_F_PARAMS, DCN_F_N_SEEDS, FEATURES_F, embed_params_F),
    )
    for suffix, mask, dcn_params, dcn_n_seeds, feat_list, embed_params in groups:
        if not mask.any():
            continue
        proba[mask] = predict_group(df.loc[mask], suffix, dcn_params, dcn_n_seeds, feat_list, embed_params)

    submission = pd.DataFrame({"row_id": df["row_id"], "control_success": proba})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"saved: {OUTPUT_PATH} ({len(submission):,} rows)")


if __name__ == "__main__":
    main()
