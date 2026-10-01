"""F 전용 SSL: R+F 전체 train_df(라벨/game_type 제외)로 denoising autoencoder를
pretrain한 뒤, encoder를 F 라벨(tr_F)로만 fine-tune해서 F의 5/6번째 앙상블 멤버로
추가한다. 예전에 시도했던 "R+F 섞어서 SSL"(노이즈 수준, +10 std 13~14)과 다른 점은
그때는 아마 전체를 pretrain+fine-tune 다 같이 했을 것이고, 이번엔 F 라벨 fine-tune
단계만 F로 좁힌다는 것 - "SSL은 라벨 없는 표현학습이라 R/F 레짐 문제가 원천적으로
안 새어든다"는 설계 논리(사용자 근거)를 따른다.

**중요한 설계 선택**: pretrain 데이터는 train_df<=cutoff로 제한한다(walk-forward
분할 무결성 유지) - 사용자 제안은 "train.csv 전체"였지만, 미래 시즌(예: val=2024)의
피처 분포까지 라벨 없이 미리 보게 되면 walk-forward가 원래 막으려는 종류의 정보
누출(feature-distribution 수준)이 생긴다. 이번 세션 내내 지켜온 규칙과 동일하게
적용한다.

평가: F 4-way(임베딩 포함 5-way) 블렌드에 6번째 멤버로 추가해서 whole-split
bss_F로만 판정(단독 성능 착시 방지, 사용자 지시)."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import torch
import torch.nn as nn
import train as T
from common import (
    FEATURES_F, TARGET, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
    EMBED_F_DIM, EMBED_F_HIDDEN, EMBED_F_DROPOUT, EMBED_F_BATCH_SIZE, EMBED_F_N_SEEDS,
    DCN_LR, DCN_WEIGHT_DECAY,
    brier_skill_score, split_regime, build_id_vocab,
    fit_dcn_preprocessing, apply_dcn_preprocessing,
)

F_SEGMENTS = T.F_SEGMENTS

MASK_RATE = 0.175  # 15~20% 중간값
PRETRAIN_BATCH_SIZE = 4096
PRETRAIN_MAX_EPOCHS = 60
PRETRAIN_PATIENCE = 6
FINE_TUNE_N_SEEDS = 5  # F 관례(표본 작을수록 시드 편차 커서 다중 시드 평균)
FINE_TUNE_MAX_EPOCHS = 100
FINE_TUNE_PATIENCE = 8
FINE_TUNE_BATCH_SIZE = 1024
ENCODER_HIDDEN = 128
BOTTLENECK = 64
HEAD_HIDDEN = 32


class DenoisingAutoencoder(nn.Module):
    def __init__(self, n_features, hidden=ENCODER_HIDDEN, bottleneck=BOTTLENECK):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x_corrupted):
        z = self.encoder(x_corrupted)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)


class FineTuneHead(nn.Module):
    def __init__(self, encoder, bottleneck=BOTTLENECK, hidden=HEAD_HIDDEN):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z).squeeze(-1)


def pretrain_autoencoder(Xtr_pre, seed=0):
    """Xtr_pre: 이미 표준화된 (n, n_features) 텐서. 마지막 10%를 재구성 손실
    조기종료용 held-out으로 뗀다(다운스트림 F val과 무관, 순수 SSL 진단용)."""
    torch.manual_seed(seed)
    n = Xtr_pre.shape[0]
    perm = torch.randperm(n)
    n_val = int(n * 0.1)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    X_ptr, X_pval = Xtr_pre[train_idx], Xtr_pre[val_idx]

    model = DenoisingAutoencoder(Xtr_pre.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    n_tr = X_ptr.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(PRETRAIN_MAX_EPOCHS):
        model.train()
        perm_ep = torch.randperm(n_tr)
        for i in range(0, n_tr, PRETRAIN_BATCH_SIZE):
            idx = perm_ep[i:i + PRETRAIN_BATCH_SIZE]
            xb = X_ptr[idx]
            mask = (torch.rand_like(xb) < MASK_RATE).float()
            xb_corrupted = xb * (1 - mask)
            opt.zero_grad()
            recon = model(xb_corrupted)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mask_val = (torch.rand_like(X_pval) < MASK_RATE).float()
            recon_val = model(X_pval * (1 - mask_val))
            val_loss = loss_fn(recon_val, X_pval).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= PRETRAIN_PATIENCE:
                break
    model.load_state_dict(best_state)
    print(f"    pretrain seed={seed}: best_epoch~{epoch+1-bad_epochs}  recon_val_loss={best_val_loss:.5f}")
    return model.encoder


def fine_tune(encoder_state, n_features, Xtr, ytr, Xval, yval, seed=0):
    torch.manual_seed(seed)
    encoder = nn.Sequential(
        nn.Linear(n_features, ENCODER_HIDDEN), nn.ReLU(),
        nn.Linear(ENCODER_HIDDEN, BOTTLENECK), nn.ReLU(),
    )
    encoder.load_state_dict(encoder_state)
    model = FineTuneHead(encoder)
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n = Xtr.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(FINE_TUNE_MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, FINE_TUNE_BATCH_SIZE):
            idx = perm[i:i + FINE_TUNE_BATCH_SIZE]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= FINE_TUNE_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xval)).numpy()


def main():
    df = T.load_data()
    cutoff, val_season = 2023, 2024
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]
    tr_R, tr_F = split_regime(train_df)
    val_F = val_df[val_df["game_type"] == "F"]
    y_val_F = val_F[TARGET].to_numpy()
    game_month = val_F["game_month"].to_numpy()

    # --- SSL pretrain: R+F 전체 train_df(<=cutoff), 라벨/game_type 제외 ---
    print(f"pretrain pool: R+F combined, n={len(train_df):,} (train<=2023, walk-forward 무결성 유지)")
    pretrain_stats = fit_dcn_preprocessing(train_df)
    Xpre_np = apply_dcn_preprocessing(train_df, pretrain_stats)
    Xpre = torch.tensor(Xpre_np)
    print(f"  n_features(전처리 후)={Xpre.shape[1]}")

    encoder = pretrain_autoencoder(Xpre, seed=0)
    encoder_state = encoder.state_dict()

    # --- fine-tune: F 라벨만, 같은 전처리 통계 재사용 ---
    Xtr_F = torch.tensor(apply_dcn_preprocessing(tr_F, pretrain_stats))
    Xval_F = torch.tensor(apply_dcn_preprocessing(val_F, pretrain_stats))
    ytr_F = torch.tensor(tr_F[TARGET].to_numpy(dtype=np.float32))
    yval_F = torch.tensor(y_val_F.astype(np.float32))

    preds_ssl = []
    for seed in range(FINE_TUNE_N_SEEDS):
        p = fine_tune(encoder_state, Xpre.shape[1], Xtr_F, ytr_F, Xval_F, yval_F, seed=seed)
        preds_ssl.append(p)
    p_ssl = np.mean(preds_ssl, axis=0)
    _, bss_ssl_solo = brier_skill_score(y_val_F, p_ssl)
    print(f"\nSSL fine-tuned solo bss_F={bss_ssl_solo:,.0f}")

    # --- 기존 F 4-way + 임베딩(5-way) 블렌드 재현 ---
    p_lgb = T.fit_lgb(tr_F, val_F, LGB_F_PARAMS, FEATURES_F)
    p_xgb = T.fit_xgb(tr_F, val_F, XGB_F_PARAMS, FEATURES_F)
    p_cat = T.fit_cat(tr_F, val_F, CAT_F_PARAMS, FEATURES_F)
    p_dcn = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)
    pitcher_vocab = build_id_vocab(tr_F, "pitcher_id")
    batter_vocab = build_id_vocab(tr_F, "batter_id")
    p_embed = T.fit_embed_multiseed(
        tr_F, val_F, pitcher_vocab, batter_vocab, EMBED_F_N_SEEDS,
        EMBED_F_DIM, EMBED_F_HIDDEN, EMBED_F_DROPOUT, EMBED_F_BATCH_SIZE,
    )
    p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5
    _, bss_5way = brier_skill_score(y_val_F, p_5way)

    p_6way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed + p_ssl) / 6
    _, bss_6way = brier_skill_score(y_val_F, p_6way)

    print(f"\n5-way (current production) whole-split bss_F={bss_5way:,.0f}")
    print(f"6-way (+SSL) whole-split bss_F={bss_6way:,.0f}  delta={bss_6way - bss_5way:+.0f}")

    print("\nsegment breakdown (참고용, 판정은 whole-split 기준):")
    for months in F_SEGMENTS:
        mask = np.isin(game_month, months)
        if mask.sum() == 0:
            continue
        _, b5 = brier_skill_score(y_val_F[mask], p_5way[mask])
        _, b6 = brier_skill_score(y_val_F[mask], p_6way[mask])
        print(f"  {months}: 5-way={b5:,.0f}  6-way(+SSL)={b6:,.0f}  delta={b6-b5:+.0f}")


if __name__ == "__main__":
    main()
