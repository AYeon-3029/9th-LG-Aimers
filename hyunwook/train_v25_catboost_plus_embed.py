"""25단계 — 검증된 pitcher/batter 임베딩을 CatBoost 입력 피처로 융합 (블렌딩 아님).

배경: 오늘 시도한 세 번의 "신경망이 최종 결정까지 담당" 구조(FT-Transformer 2회,
NODE 3회 크래시)가 전부 막혔다 — 공통점은 경사하강법으로 끝까지 학습하는 신경망이
이 태스크의 BSS 지표와 안 맞는다는 것. 반면 "최종 결정은 CatBoost가 담당"하는
구조는 지금까지 이 문제를 겪은 적이 없다.

이 스크립트의 프레임: 신경망으로 CatBoost를 대체하지 않고, pitcher/batter 임베딩만
신경망으로 미리 학습해서 CatBoost의 추가 수치형 입력으로 넣는다.
  - ayeon 파이프라인에서 이미 R 기준으로 검증된 메커니즘(EMBED_R_*, 저차원
    nn.Embedding, 경사하강법 학습)을 재사용하되, ayeon의 저장된 가중치는 그대로
    가져다 쓰지 않는다 — 그건 2019~2024 전체로 학습된 프로덕션 임베딩이라, 이
    스크립트의 val=2024 스크리닝에 그대로 쓰면 2024 정보가 임베딩에 이미 녹아든
    채로 새는 셈(리크)이다. 대신 hyunwook님의 정확한 fold(train<2024)로 임베딩을
    새로 학습한다.
  - CatBoost 원시 범주형으로 pitcher_id/batter_id를 넣었다가 실패한 것(6.19,
    -29~-174)과도, 신경망 종단학습(오늘 3번 불안정)과도 다른 세 번째 경로 —
    "학습된 저차원 표현을 트리 모델의 평범한 수치형 피처로 제공"이다.

임베딩 사전학습: pitcher_id/batter_id를 nn.Embedding(dim=8, 0=unknown)으로,
hyunwook님의 수치형 72피처(표준화)와 concat 후 작은 MLP로 control_success를 예측하며
학습(ayeon EmbedMLP와 동일 구조). 목적은 이 MLP 자체의 예측력이 아니라 임베딩
벡터를 얻는 것 - epoch 수는 ayeon의 R 임베딩에서 이미 확인된 안정적인 값(6, R
DCN/embedding류가 이 프로젝트에서 유일하게 시드 안정성을 보인 학습 방식)을 그대로
재사용한다(재도출하지 않음 - 시간 절약, 이 학습 방식 자체가 이미 안정적이라고
검증됐으므로 재확인 비용이 낮은 가정).

CatBoost 쪽은 항상 902.04를 만든 정확한 하이퍼파라미터(depth=6, lr=0.02, l2=3.0)를
그대로 유지 - 이번 실험의 유일한 변수는 "임베딩 8+8=16개 수치형 컬럼 추가 여부"뿐.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

EMBED_DIM = 8
EMBED_HIDDEN = (64, 32)
EMBED_DROPOUT = 0.1
EMBED_BATCH_SIZE = 4096
EMBED_EPOCHS = 6  # ayeon EMBED_R_FINAL_EPOCHS 재사용 (재도출 안 함, 근거는 상단 docstring)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATBOOST_PARAMS = dict(
    iterations=3000, learning_rate=0.02, depth=6, l2_leaf_reg=3.0,
    random_seed=42, loss_function="Logloss", eval_metric="Logloss",
    verbose=False, thread_count=-1,
)


def bss(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def build_id_vocab(train_df, col):
    return {v: i + 1 for i, v in enumerate(sorted(train_df[col].unique()))}


def encode_ids(df, col, vocab):
    return df[col].map(vocab).fillna(0).to_numpy(dtype=np.int64)


class EmbedMLP(nn.Module):
    def __init__(self, n_raw, n_pitchers, n_batters, embed_dim=EMBED_DIM,
                 hidden=EMBED_HIDDEN, dropout=EMBED_DROPOUT):
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


def prep_base(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]  # pitcher_id/batter_id 포함(현재 902점 구성과 동일)

    train_df = train.loc[is_train].reset_index(drop=True)
    val_df = train.loc[is_val].reset_index(drop=True)
    return train_df, val_df, FEATURES, NUM_COLS


def train_embeddings(train_df, NUM_COLS):
    """train_df(train<val_season)만으로 pitcher/batter 임베딩 학습. 리크 없음."""
    pitcher_vocab = build_id_vocab(train_df, "pitcher_id")
    batter_vocab = build_id_vocab(train_df, "batter_id")
    pid = encode_ids(train_df, "pitcher_id", pitcher_vocab)
    bid = encode_ids(train_df, "batter_id", batter_vocab)

    num = train_df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    medians = num.median()
    num = num.fillna(medians)
    means, stds = num.mean(), num.std().replace(0, 1.0)
    x_raw = ((num - means) / stds).to_numpy(dtype=np.float32)
    y = train_df[TARGET].to_numpy(dtype=np.float32)

    x_raw_t = torch.as_tensor(x_raw, device=DEVICE)
    pid_t = torch.as_tensor(pid, dtype=torch.long, device=DEVICE)
    bid_t = torch.as_tensor(bid, dtype=torch.long, device=DEVICE)
    y_t = torch.as_tensor(y, device=DEVICE)

    torch.manual_seed(42)
    model = EmbedMLP(x_raw.shape[1], len(pitcher_vocab), len(batter_vocab)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    n = x_raw_t.shape[0]
    for epoch in range(1, EMBED_EPOCHS + 1):
        t = time.time()
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        for start in range(0, n, EMBED_BATCH_SIZE):
            idx = perm[start:start + EMBED_BATCH_SIZE]
            opt.zero_grad()
            logit = model(x_raw_t[idx], pid_t[idx], bid_t[idx])
            loss = loss_fn(logit, y_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        print(f"  [embed] epoch={epoch} | loss={total_loss/n:.5f} | {time.time()-t:.1f}s")

    pitcher_emb_w = model.pitcher_emb.weight.detach().cpu().numpy()
    batter_emb_w = model.batter_emb.weight.detach().cpu().numpy()
    return pitcher_vocab, batter_vocab, pitcher_emb_w, batter_emb_w


def add_embed_columns(df, pitcher_vocab, batter_vocab, pitcher_emb_w, batter_emb_w):
    df = df.copy()
    pid = encode_ids(df, "pitcher_id", pitcher_vocab)
    bid = encode_ids(df, "batter_id", batter_vocab)
    p_vecs = pitcher_emb_w[pid]
    b_vecs = batter_emb_w[bid]
    p_cols = [f"pitcher_embed_{i}" for i in range(p_vecs.shape[1])]
    b_cols = [f"batter_embed_{i}" for i in range(b_vecs.shape[1])]
    for i, c in enumerate(p_cols):
        df[c] = p_vecs[:, i]
    for i, c in enumerate(b_cols):
        df[c] = b_vecs[:, i]
    return df, p_cols + b_cols


def run_catboost(train_df, val_df, features, cat_cols):
    num_cols = [c for c in features if c not in cat_cols]
    wrapper = CatBoostWrapper(cat_cols, num_cols, **CATBOOST_PARAMS)
    wrapper.fit(train_df[features], train_df[TARGET],
                eval_set=(val_df[features], val_df[TARGET]), early_stopping_rounds=100)
    pred = wrapper.predict_proba(val_df[features])[:, 1]
    return bss(pred, val_df[TARGET].to_numpy()), wrapper.best_iteration_


def main():
    print(f"device={DEVICE}")
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    val_season = 2024
    train_df, val_df, FEATURES, NUM_COLS = prep_base(raw_train, prior_table, val_season)
    print(f"train={len(train_df)} | val={len(val_df)} | 피처={len(FEATURES)}")

    print("\n=== baseline: CatBoost만 (902점 구성) ===")
    t = time.time()
    base_score, base_iter = run_catboost(train_df, val_df, FEATURES, CAT_COLS)
    print(f"  Score={base_score:.2f} (best_iter={base_iter}) | {time.time()-t:.1f}s")

    print("\n=== pitcher/batter 임베딩 학습 (train<2024만, 리크 없음) ===")
    pitcher_vocab, batter_vocab, p_w, b_w = train_embeddings(train_df, NUM_COLS)
    print(f"  n_pitchers={len(pitcher_vocab)} | n_batters={len(batter_vocab)} | dim={EMBED_DIM}")

    train_df2, embed_cols = add_embed_columns(train_df, pitcher_vocab, batter_vocab, p_w, b_w)
    val_df2, _ = add_embed_columns(val_df, pitcher_vocab, batter_vocab, p_w, b_w)
    FEATURES_PLUS = FEATURES + embed_cols

    print("\n=== CatBoost + 임베딩 16개 피처 (같은 902점 하이퍼파라미터) ===")
    t = time.time()
    plus_score, plus_iter = run_catboost(train_df2, val_df2, FEATURES_PLUS, CAT_COLS)
    print(f"  Score={plus_score:.2f} (best_iter={plus_iter}) | {time.time()-t:.1f}s")

    print(f"\n=== 결과 (val={val_season}) ===")
    print(f"  CatBoost baseline        : {base_score:8.2f}")
    print(f"  CatBoost + 임베딩 피처   : {plus_score:8.2f}")
    print(f"  차이: {plus_score - base_score:+.2f}")


if __name__ == "__main__":
    main()
