"""신규 아이디어: pitcher_id/batter_id 저차원 임베딩을 5번째 앙상블 멤버로 추가.

CatBoost 범주형 ID(rejection 기록, common.py 144번째 항목)는 ordered target
statistics가 시즌에 묶인 우연한 패턴을 암기해 R 3구간 전부 대폭 악화(-29/-115/
-122)됐다. 이번 시도는 그와 다르게, (1) 저차원(8)으로 용량을 강하게 제한하고
(2) target statistic이 아니라 gradient descent로 학습되는 분산 표현(embedding)을
쓰고 (3) 미확인 ID는 별도 unknown 벡터(index 0)로 처리한다 - 그래도 같은 실패
패턴(시즌/상대에 묶인 암기)이 재발할 위험은 있어 모니터링 대상.

오늘 배운 "개별 성능 vs 블렌드 기여도는 다르다" 교훈을 처음부터 반영해서, 이
스크립트는 솔로 성능이 아니라 곧바로 **기존 4모델(LGB/XGB/CAT/DCN) 블렌드 대비
5번째 멤버로 추가했을 때의 blend BSS 변화**를 R 3구간 walk-forward로 측정한다.
DCN처럼 시드 편차가 클 수 있으므로 3시드 평균(R DCN과 동일 관례)을 쓴다.
"""
import train as T
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import (
    FEATURES, TARGET, ASOF_FEATURES, SITUATIONAL_FEATURES, DCN_FEATURES,
    LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    fit_dcn_preprocessing, apply_dcn_preprocessing, brier_skill_score,
    split_regime, add_matchup_features,
)

DATA_DIR = T.DATA_DIR
EMBED_DIM = 8
EMBED_BATCH_SIZE = 4096
EMBED_N_SEEDS = 3  # R DCN과 동일 관례


def load_data_with_ids():
    cols = (
        ASOF_FEATURES + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "pitcher_id", "batter_id", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


class EmbedMLP(nn.Module):
    def __init__(self, n_raw, n_pitchers, n_batters, embed_dim=EMBED_DIM, hidden=(64, 32), dropout=0.1):
        super().__init__()
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, embed_dim)  # 0 = unknown
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


def build_vocab(train_df, col):
    return {v: i + 1 for i, v in enumerate(sorted(train_df[col].unique()))}  # 0=unknown


def encode_ids(df, col, vocab):
    return df[col].map(vocab).fillna(0).to_numpy(dtype=np.int64)


def fit_embed_model(train_df, val_df, pitcher_vocab, batter_vocab, seed=0, max_epochs=100, patience=8):
    torch.manual_seed(seed)
    stats = fit_dcn_preprocessing(train_df)
    Xtr_raw = torch.tensor(apply_dcn_preprocessing(train_df, stats))
    Xval_raw = torch.tensor(apply_dcn_preprocessing(val_df, stats))
    pid_tr = torch.tensor(encode_ids(train_df, "pitcher_id", pitcher_vocab))
    bid_tr = torch.tensor(encode_ids(train_df, "batter_id", batter_vocab))
    pid_val = torch.tensor(encode_ids(val_df, "pitcher_id", pitcher_vocab))
    bid_val = torch.tensor(encode_ids(val_df, "batter_id", batter_vocab))
    ytr = torch.tensor(train_df[TARGET].to_numpy(dtype=np.float32))
    yval = torch.tensor(val_df[TARGET].to_numpy(dtype=np.float32))

    model = EmbedMLP(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    n = Xtr_raw.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, EMBED_BATCH_SIZE):
            idx = perm[i:i + EMBED_BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_raw[idx], pid_tr[idx], bid_tr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_raw, pid_val, bid_val), yval).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xval_raw, pid_val, bid_val)).numpy()


def fit_embed_multiseed(train_df, val_df, pitcher_vocab, batter_vocab, n_seeds=EMBED_N_SEEDS):
    preds = [fit_embed_model(train_df, val_df, pitcher_vocab, batter_vocab, seed=s) for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def main():
    df = load_data_with_ids()

    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        tr_R, _ = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        y_val = val_R[TARGET].to_numpy()

        p_lgb = T.fit_lgb(tr_R, val_R, LGB_BASE_PARAMS, FEATURES)
        p_xgb = T.fit_xgb(tr_R, val_R, XGB_BASE_PARAMS, FEATURES)
        p_cat = T.fit_cat(tr_R, val_R, CAT_BASE_PARAMS, FEATURES)
        p_dcn = T.fit_dcn_multiseed(tr_R, val_R, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS)

        pitcher_vocab = build_vocab(tr_R, "pitcher_id")
        batter_vocab = build_vocab(tr_R, "batter_id")
        unseen_pitcher = (~val_R["pitcher_id"].isin(pitcher_vocab)).mean()
        unseen_batter = (~val_R["batter_id"].isin(batter_vocab)).mean()
        p_embed = fit_embed_multiseed(tr_R, val_R, pitcher_vocab, batter_vocab)

        p_4way = (p_lgb + p_xgb + p_cat + p_dcn) / 4
        p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5

        _, bss_embed_solo = brier_skill_score(y_val, p_embed)
        _, bss_4way = brier_skill_score(y_val, p_4way)
        _, bss_5way = brier_skill_score(y_val, p_5way)

        print(
            f"[train<={cutoff} val={val_season}] n_pitchers={len(pitcher_vocab)} "
            f"n_batters={len(batter_vocab)} unseen_pitcher={unseen_pitcher:.1%} "
            f"unseen_batter={unseen_batter:.1%}"
        )
        print(
            f"  embed solo bss={bss_embed_solo:,.0f}  4-way(current) bss={bss_4way:,.0f}  "
            f"5-way(+embed) bss={bss_5way:,.0f}  delta={bss_5way - bss_4way:+.0f}"
        )


if __name__ == "__main__":
    main()
