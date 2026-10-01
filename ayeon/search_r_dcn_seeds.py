"""3순위: R DCNv2 앙상블 시드 수 확장(3->5) 검증.

F는 이미 5시드 평균(DCN_F_N_SEEDS=5, 33->366으로 극적 안정화, common.py 근거)을
쓰는데 R은 3시드(DCN_R_N_SEEDS=3, 1시드630->3시드674)에 머물러 있다 - 순수
분산 감소 시도이므로(용량/구조 변경이 아님) 레벨 보정 계열 실패 패턴과 무관하고
리스크가 낮다(2026-08-22 논의 근거, memory: lgaimers-model-state 우선순위 3).

R 3구간 walk-forward 전부에서 확인한다(F와 달리 R은 3구간 전부 유효, memory:
lgaimers-validation-lessons #1). 시드 0~4를 한 번씩만 학습해서, 처음 3개
평균(현재 프로덕션)과 5개 평균(제안)을 같은 학습 결과로 비교한다(중복 학습 없음).
"""
import train as T  # lightgbm을 가장 먼저 import해야 하는 이 환경의 이슈 회피
import numpy as np
from common import (
    FEATURES, TARGET, add_shrunk_pitcher_feature, compute_season_means,
    split_regime, brier_skill_score, DCN_R_PARAMS, DCN_R_BATCH_SIZE,
)

MAX_SEEDS = 5


def main():
    df = T.load_data()
    bss3_list, bss5_list = [], []
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]

        season_means = compute_season_means(train_df)
        train_df = add_shrunk_pitcher_feature(train_df, season_means)
        val_df = add_shrunk_pitcher_feature(val_df, season_means)

        tr_R, _ = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        y_R = val_R[TARGET].to_numpy()

        preds = [T.fit_dcn(tr_R, val_R, DCN_R_PARAMS, DCN_R_BATCH_SIZE, seed=s) for s in range(MAX_SEEDS)]
        p3 = np.mean(preds[:3], axis=0)
        p5 = np.mean(preds, axis=0)
        _, bss3 = brier_skill_score(y_R, p3)
        _, bss5 = brier_skill_score(y_R, p5)
        bss3_list.append(bss3)
        bss5_list.append(bss5)
        per_seed = [brier_skill_score(y_R, p)[1] for p in preds]
        print(f"[train<={cutoff} val={val_season}] n_train_R={len(tr_R):,}  "
              f"3-seed={bss3:,.0f}  5-seed={bss5:,.0f}  delta={bss5-bss3:+.0f}  "
              f"per-seed={[round(b) for b in per_seed]}")

    print(f"[SUMMARY] R DCN 3-seed avg BSS = {np.mean(bss3_list):,.0f}  per-split={[round(v) for v in bss3_list]}")
    print(f"[SUMMARY] R DCN 5-seed avg BSS = {np.mean(bss5_list):,.0f}  per-split={[round(v) for v in bss5_list]}")


if __name__ == "__main__":
    main()
