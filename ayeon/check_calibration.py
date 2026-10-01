"""792.24+ 구조(R Optuna 트리+임베딩 5-way, F 임베딩 5-way) 전체에 대해 캘리브레이션을
처음으로 재점검한다 - R/F 분리, DCN 재탐색, 임베딩 추가가 전부 반영된 이후 한 번도
확인 안 됐다. train<=2023->val=2024(진짜 held-out)의 예측으로 ECE + 구간별
reliability 테이블을 R/F 각각 만든다."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import train as T
from common import (
    FEATURES, FEATURES_F, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
    EMBED_R_DIM, EMBED_R_HIDDEN, EMBED_R_DROPOUT, EMBED_R_BATCH_SIZE, EMBED_R_N_SEEDS,
    EMBED_F_DIM, EMBED_F_HIDDEN, EMBED_F_DROPOUT, EMBED_F_BATCH_SIZE, EMBED_F_N_SEEDS,
    split_regime, build_id_vocab, brier_skill_score,
)

N_BINS = 10


def compute_ece_and_table(y, p, n_bins=N_BINS, label=""):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    print(f"\n[{label}] n={len(y):,}  mean_pred={p.mean():.4f}  actual_rate={y.mean():.4f}")
    print(f"  {'bin':>12}  {'n':>8}  {'mean_pred':>10}  {'actual':>8}  {'gap':>8}")
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = mask.sum()
        if n_b == 0:
            continue
        mean_pred_b = p[mask].mean()
        actual_b = y[mask].mean()
        gap = mean_pred_b - actual_b
        ece += (n_b / len(y)) * abs(gap)
        print(f"  {bins[b]:.2f}-{bins[b+1]:.2f}  {n_b:>8,}  {mean_pred_b:>10.4f}  {actual_b:>8.4f}  {gap:>+8.4f}")
    print(f"  ECE = {ece:.5f}")
    return ece


def main():
    df = T.load_data()
    cutoff, val_season = 2023, 2024
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]
    tr_R, tr_F = split_regime(train_df)
    val_R = val_df[val_df["game_type"] == "R"]
    val_F = val_df[val_df["game_type"] == "F"]

    pitcher_vocab_R = build_id_vocab(tr_R, "pitcher_id")
    batter_vocab_R = build_id_vocab(tr_R, "batter_id")
    embed_config_R = dict(
        pitcher_vocab=pitcher_vocab_R, batter_vocab=batter_vocab_R, n_seeds=EMBED_R_N_SEEDS,
        embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE,
    )
    preds_R = T.fit_ensemble_predictions(
        tr_R, val_R, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
        DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS, FEATURES, embed_config=embed_config_R,
    )
    p_R = preds_R.mean(axis=1)
    y_R = val_R[TARGET].to_numpy()

    pitcher_vocab_F = build_id_vocab(tr_F, "pitcher_id")
    batter_vocab_F = build_id_vocab(tr_F, "batter_id")
    embed_config_F = dict(
        pitcher_vocab=pitcher_vocab_F, batter_vocab=batter_vocab_F, n_seeds=EMBED_F_N_SEEDS,
        embed_dim=EMBED_F_DIM, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT, batch_size=EMBED_F_BATCH_SIZE,
    )
    preds_F = T.fit_ensemble_predictions(
        tr_F, val_F, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
        DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, FEATURES_F, embed_config=embed_config_F,
    )
    p_F = preds_F.mean(axis=1)
    y_F = val_F[TARGET].to_numpy()

    _, bss_R = brier_skill_score(y_R, p_R)
    _, bss_F = brier_skill_score(y_F, p_F)
    print(f"sanity check (should match known production numbers): bss_R={bss_R:,.0f}  bss_F={bss_F:,.0f}")

    compute_ece_and_table(y_R, p_R, label="R (train<=2023 -> val=2024)")
    compute_ece_and_table(y_F, p_F, label="F (train<=2023 -> val=2024)")


if __name__ == "__main__":
    main()
