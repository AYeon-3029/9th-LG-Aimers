"""embed_dim(현재 고정값 8)을 스윕해서 더 나은 값이 있는지 확인한다. 트리
3종+DCN 예측은 embed_dim과 무관하므로 스플릿당 한 번만 계산해서 재사용하고,
embed_dim 후보별로 임베딩만 다시 학습한다(비용 절감). 5-way 블렌드 레벨에서
직접 비교한다(오늘 확립한 규칙 - 처음부터 블렌드 기준으로 검증)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import torch
import torch.nn as nn
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    EMBED_HIDDEN, EMBED_DROPOUT, EMBED_BATCH_SIZE, EMBED_R_N_SEEDS,
    DCN_LR, DCN_WEIGHT_DECAY,
    brier_skill_score, split_regime, build_id_vocab, encode_ids,
    fit_dcn_preprocessing, apply_dcn_preprocessing, EmbedMLP,
)

EMBED_DIM_CANDIDATES = [4, 8, 16, 24, 32]


def fit_embed_dim(train_df, val_df, pitcher_vocab, batter_vocab, embed_dim, seed, max_epochs=100, patience=8):
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

    model = EmbedMLP(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab), embed_dim=embed_dim, hidden=EMBED_HIDDEN, dropout=EMBED_DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
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


def fit_embed_dim_multiseed(train_df, val_df, pitcher_vocab, batter_vocab, embed_dim, n_seeds=EMBED_R_N_SEEDS):
    preds = [fit_embed_dim(train_df, val_df, pitcher_vocab, batter_vocab, embed_dim, seed=s) for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def main():
    df = T.load_data()
    results = {d: [] for d in EMBED_DIM_CANDIDATES}
    baseline4way = []

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
        p_4way = (p_lgb + p_xgb + p_cat + p_dcn) / 4
        _, bss_4way = brier_skill_score(y_val, p_4way)
        baseline4way.append(bss_4way)

        pitcher_vocab = build_id_vocab(tr_R, "pitcher_id")
        batter_vocab = build_id_vocab(tr_R, "batter_id")

        print(f"\n[train<={cutoff} val={val_season}] 4-way baseline={bss_4way:,.0f}")
        for dim in EMBED_DIM_CANDIDATES:
            p_embed = fit_embed_dim_multiseed(tr_R, val_R, pitcher_vocab, batter_vocab, dim)
            p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5
            _, bss_5way = brier_skill_score(y_val, p_5way)
            results[dim].append(bss_5way)
            print(f"  embed_dim={dim:>3}  5-way bss={bss_5way:,.0f}  delta vs 4-way={bss_5way - bss_4way:+.0f}")

    print("\n=== SUMMARY ===")
    print(f"4-way baseline AVG: {np.mean(baseline4way):,.1f}  per-split={[round(v) for v in baseline4way]}")
    for dim in EMBED_DIM_CANDIDATES:
        avg = np.mean(results[dim])
        print(f"embed_dim={dim:>3}  AVG 5-way={avg:,.1f}  delta vs baseline={avg - np.mean(baseline4way):+.1f}  per-split={[round(v) for v in results[dim]]}")


if __name__ == "__main__":
    main()
