"""R 임베딩의 시드 수(현재 3)를 스윕한다 - R DCN이 3->5시드로 늘었을 때 앙상블
다양성 손실로 실패했던 것과 같은 패턴이 임베딩에도 재현되는지 확인(사용자 지시
우선순위 2). 트리 3종+DCN 예측은 시드 수와 무관하므로 스플릿당 한 번만 계산해서
재사용한다."""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    brier_skill_score, split_regime, build_id_vocab,
)

SEED_CANDIDATES = [1, 3, 5, 7]


def main():
    df = T.load_data()
    results = {n: [] for n in SEED_CANDIDATES}
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

        # 시드 7개까지 한 번에 학습해서 앞에서부터 잘라 쓴다(중복 학습 방지)
        max_seeds = max(SEED_CANDIDATES)
        all_seed_preds = [T.fit_embed(tr_R, val_R, pitcher_vocab, batter_vocab, seed=s) for s in range(max_seeds)]

        print(f"\n[train<={cutoff} val={val_season}] 4-way baseline={bss_4way:,.0f}")
        for n in SEED_CANDIDATES:
            p_embed = np.mean(all_seed_preds[:n], axis=0)
            p_5way = (p_lgb + p_xgb + p_cat + p_dcn + p_embed) / 5
            _, bss_5way = brier_skill_score(y_val, p_5way)
            results[n].append(bss_5way)
            print(f"  embed_n_seeds={n}  5-way bss={bss_5way:,.0f}  delta vs 4-way={bss_5way - bss_4way:+.0f}")

    print("\n=== SUMMARY ===")
    print(f"4-way baseline AVG: {np.mean(baseline4way):,.1f}  per-split={[round(v) for v in baseline4way]}")
    for n in SEED_CANDIDATES:
        avg = np.mean(results[n])
        print(f"embed_n_seeds={n}  AVG 5-way={avg:,.1f}  delta vs baseline={avg - np.mean(baseline4way):+.1f}  per-split={[round(v) for v in results[n]]}")


if __name__ == "__main__":
    main()
