"""F pitcher/batter 임베딩 차원을 비대칭으로 시험한다(사용자 지시 우선순위 3,
F 임베딩 결과 위에서 이어서). F는 n_pitchers=240, n_batters=269(비율 1.12)로
차이가 크지 않지만, 전수조사 없이 2개 조합만 테스트: (6,10)과 그 반대(10,6) -
방향 자체가 의미 있는지(클래스 수 비율과 일치하는 쪽이 더 나은지) 같이 확인한다.
whole-split을 결정 기준으로, 월별 4구간은 참고용으로 같이 본다(F 안전장치 유지)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import torch
import torch.nn as nn
import train as T
from common import (
    FEATURES_F, TARGET, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, DCN_LR, DCN_WEIGHT_DECAY,
    brier_skill_score, split_regime, build_id_vocab, encode_ids,
    fit_dcn_preprocessing, apply_dcn_preprocessing,
)

EMBED_F_HIDDEN = (16, 8)
EMBED_F_DROPOUT = 0.2
EMBED_F_BATCH_SIZE = 1024
EMBED_F_N_SEEDS = 5
F_SEGMENTS = T.F_SEGMENTS

DIM_CANDIDATES = [(8, 8), (6, 10), (10, 6)]  # (pitcher_dim, batter_dim)


class EmbedMLPAsym(nn.Module):
    def __init__(self, n_raw, n_pitchers, n_batters, pitcher_dim, batter_dim, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT):
        super().__init__()
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, pitcher_dim)
        self.batter_emb = nn.Embedding(n_batters + 1, batter_dim)
        layers, prev = [], n_raw + pitcher_dim + batter_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_raw, pid, bid):
        x = torch.cat([x_raw, self.pitcher_emb(pid), self.batter_emb(bid)], dim=1)
        return self.mlp(x).squeeze(-1)


def fit_embed_asym(train_df, val_df, pitcher_vocab, batter_vocab, pitcher_dim, batter_dim, seed=0, max_epochs=100, patience=8):
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

    model = EmbedMLPAsym(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab), pitcher_dim, batter_dim)
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()
    n = Xtr_raw.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, EMBED_F_BATCH_SIZE):
            idx = perm[i:i + EMBED_F_BATCH_SIZE]
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


def fit_embed_asym_multiseed(train_df, val_df, pitcher_vocab, batter_vocab, pitcher_dim, batter_dim, n_seeds=EMBED_F_N_SEEDS):
    preds = [fit_embed_asym(train_df, val_df, pitcher_vocab, batter_vocab, pitcher_dim, batter_dim, seed=s) for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def segment_report(y, p, game_month, label):
    seg_bss = []
    for months in F_SEGMENTS:
        mask = np.isin(game_month, months)
        if mask.sum() == 0:
            continue
        _, b = brier_skill_score(y[mask], p[mask])
        seg_bss.append(b)
    print(f"  {label}: segments={[round(b) for b in seg_bss]}")


def main():
    df = T.load_data()
    cutoff, val_season = 2023, 2024
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]
    _, tr_F = split_regime(train_df)
    val_F = val_df[val_df["game_type"] == "F"]
    y_val = val_F[TARGET].to_numpy()
    game_month = val_F["game_month"].to_numpy()

    p_lgb = T.fit_lgb(tr_F, val_F, LGB_F_PARAMS, FEATURES_F)
    p_xgb = T.fit_xgb(tr_F, val_F, XGB_F_PARAMS, FEATURES_F)
    p_cat = T.fit_cat(tr_F, val_F, CAT_F_PARAMS, FEATURES_F)
    p_dcn = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)
    p_4way = (p_lgb + p_xgb + p_cat + p_dcn) / 4
    _, bss_4way = brier_skill_score(y_val, p_4way)
    print(f"4-way (current) whole-split bss_F={bss_4way:,.0f}")
    segment_report(y_val, p_4way, game_month, "4-way (current)")

    pitcher_vocab = build_id_vocab(tr_F, "pitcher_id")
    batter_vocab = build_id_vocab(tr_F, "batter_id")

    for pdim, bdim in DIM_CANDIDATES:
        p_embed = fit_embed_asym_multiseed(tr_F, val_F, pitcher_vocab, batter_vocab, pdim, bdim)
        p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5
        _, bss_5way = brier_skill_score(y_val, p_5way)
        print(f"\n(pitcher_dim={pdim}, batter_dim={bdim})  whole-split bss_F={bss_5way:,.0f}  delta vs 4-way={bss_5way - bss_4way:+.0f}")
        segment_report(y_val, p_5way, game_month, f"dim=({pdim},{bdim})")


if __name__ == "__main__":
    main()
