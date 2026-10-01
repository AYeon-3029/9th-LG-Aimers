"""F(퓨처스)에 pitcher/batter 임베딩을 5번째 앙상블 멤버로 시도한다(R에서 검증된
메커니즘을 F로 확장, 사용자 지시 우선순위 1). 안전장치:
  1) whole-split + 월별 4구간 breakdown 둘 다 확인 (F DCN 아키텍처 실패 재발 방지 -
     당시 월별로는 좋아 보였는데 whole-split 블렌드에서 손해였다)
  2) unseen ID 비율 사전 확인 완료(47.1%/43.2% - R보다 훨씬 높음, 별도 스크립트로 확인)
  3) F 전용 임베딩 상수를 R과 완전히 독립적으로 정의 - LGB_F_PARAMS 등에서 이미
     겪은 "R 하이퍼파라미터가 F로 조용히 새어들어가는" 버그를 반복하지 않는다
  4) 이 스크립트는 로컬 검증까지만 - 실제 프로덕션 반영/제출은 결과 확인 후 별도 결정
"""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import torch
import torch.nn as nn
import train as T
from common import (
    FEATURES_F, TARGET, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
    DCN_LR, DCN_WEIGHT_DECAY,
    brier_skill_score, split_regime, build_id_vocab, encode_ids,
    fit_dcn_preprocessing, apply_dcn_preprocessing, EmbedMLP,
)

# F 전용 임베딩 설정 - R의 EMBED_DIM/EMBED_HIDDEN/EMBED_R_N_SEEDS를 절대 참조하지 않고
# 완전히 독립적으로 하드코딩한다 (F 표본이 훨씬 작으므로 처음부터 작은 네트워크로 시작 -
# DCN에서 F가 R보다 훨씬 작은 네트워크를 원했던 것과 같은 논리)
EMBED_F_DIM = 8
EMBED_F_HIDDEN = (16, 8)
EMBED_F_DROPOUT = 0.2
EMBED_F_BATCH_SIZE = 1024
EMBED_F_N_SEEDS = 5  # F DCN과 동일 관례 - 표본 작을수록 시드 편차 커서 더 많은 시드 평균

F_SEGMENTS = T.F_SEGMENTS


def fit_embed_f(train_df, val_df, pitcher_vocab, batter_vocab, seed=0, max_epochs=100, patience=8):
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

    model = EmbedMLP(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab), embed_dim=EMBED_F_DIM, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT)
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


def fit_embed_f_multiseed(train_df, val_df, pitcher_vocab, batter_vocab, n_seeds=EMBED_F_N_SEEDS):
    preds = [fit_embed_f(train_df, val_df, pitcher_vocab, batter_vocab, seed=s) for s in range(n_seeds)]
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
    cutoff, val_season = 2023, 2024  # F 신호가 있는 유일한 구간
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

    pitcher_vocab = build_id_vocab(tr_F, "pitcher_id")
    batter_vocab = build_id_vocab(tr_F, "batter_id")
    p_embed = fit_embed_f_multiseed(tr_F, val_F, pitcher_vocab, batter_vocab)
    _, bss_embed_solo = brier_skill_score(y_val, p_embed)
    p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5
    _, bss_5way = brier_skill_score(y_val, p_5way)

    print(f"[train<={cutoff} val={val_season}] n_train_F={len(tr_F):,}  n_pitchers={len(pitcher_vocab)}  n_batters={len(batter_vocab)}")
    print(f"embed solo bss={bss_embed_solo:,.0f}")
    print(f"4-way (current) whole-split bss_F={bss_4way:,.0f}")
    print(f"5-way (+F embed) whole-split bss_F={bss_5way:,.0f}  delta={bss_5way - bss_4way:+.0f}")
    print()
    segment_report(y_val, p_4way, game_month, "4-way (current)")
    segment_report(y_val, p_5way, game_month, "5-way (+F embed)")


if __name__ == "__main__":
    main()
