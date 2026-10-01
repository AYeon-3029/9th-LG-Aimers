"""R DCN 시드 확장(3->5)과 F DCN 아키텍처 변경 중 어느 쪽이 4-way 앙상블 레벨의
희석 효과(AVG 994->993, bss_F 531->526)를 만드는지 재학습 없이 분리한다.

_stack_cache/split_*.npz에는 구(舊) 설정(DCN_R_N_SEEDS=3, DCN_F_PARAMS 구아키텍처)
으로 계산된 LGB/XGB/CAT/DCN 4열 예측이 그대로 남아있다. LGB/XGB/CAT는
TREE_SEED=0으로 완전히 결정론적이라(common.py 근거) 재학습 없이 그대로 재사용
가능하고, DCN 열(구)도 "구성 조합"의 기준값으로 그대로 쓴다. 여기서는 신규 DCN
(R 5시드, F 신아키텍처)만 다시 학습해서 4가지 조합을 만든다:
  1) 구R DCN + 구F DCN (기준선 - _stack_cache의 avg와 일치해야 함, 검산용)
  2) 신R DCN + 구F DCN (R 변경 단독 효과)
  3) 구R DCN + 신F DCN (F 변경 단독 효과)
  4) 신R DCN + 신F DCN (최종 프로덕션 - 방금 전체 재학습 결과와 일치해야 함, 검산용)
"""
import train as T
import numpy as np
from common import (
    TARGET, add_shrunk_pitcher_feature, compute_season_means, split_regime,
    brier_skill_score, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
)

# 구 설정 (common.py 변경 전, _stack_cache 생성 당시) - 하드코딩해서 재현
OLD_DCN_R_PARAMS = dict(cross_layers=2, deep_dims=(256, 128), dropout=0.1)
OLD_DCN_R_N_SEEDS = 3
OLD_DCN_F_PARAMS = dict(cross_layers=2, deep_dims=(32, 16), dropout=0.2)
OLD_DCN_F_N_SEEDS = 5

SPLITS = T.WALK_FORWARD_SPLITS


def main():
    df = T.load_data()
    results = {label: [] for label in ["old/old", "newR/oldF", "oldR/newF", "new/new"]}
    f_results = {label: [] for label in ["old/old", "newR/oldF", "oldR/newF", "new/new"]}

    for cutoff, val_season in SPLITS:
        d = np.load(str(T.CACHE_DIR / f"split_{cutoff}_{val_season}.npz"))
        preds_R_old, y_R = d["preds_R"], d["y_R"]  # (n,4): lgb,xgb,cat,dcn(old)
        preds_F_old, y_F = d["preds_F"], d["y_F"]
        y_all = np.concatenate([y_R, y_F])

        # 신규 DCN만 재학습 (LGB/XGB/CAT는 캐시 재사용)
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        season_means = compute_season_means(train_df)
        train_df = add_shrunk_pitcher_feature(train_df, season_means)
        val_df = add_shrunk_pitcher_feature(val_df, season_means)
        tr_R, tr_F = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        val_F = val_df[val_df["game_type"] == "F"]

        p_dcn_R_new = T.fit_dcn_multiseed(tr_R, val_R, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS)
        if len(val_F) > 0:
            p_dcn_F_new = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)
        else:
            p_dcn_F_new = preds_F_old[:, 3]  # F 없는 구간 - 자리만 채움(사용 안 됨)

        combos = {
            "old/old": (preds_R_old[:, 3], preds_F_old[:, 3]),
            "newR/oldF": (p_dcn_R_new, preds_F_old[:, 3]),
            "oldR/newF": (preds_R_old[:, 3], p_dcn_F_new),
            "new/new": (p_dcn_R_new, p_dcn_F_new),
        }

        print(f"\n[train<={cutoff} val={val_season}]")
        for label, (dcn_R, dcn_F) in combos.items():
            p_R = (preds_R_old[:, 0] + preds_R_old[:, 1] + preds_R_old[:, 2] + dcn_R) / 4
            p_F = (preds_F_old[:, 0] + preds_F_old[:, 1] + preds_F_old[:, 2] + dcn_F) / 4
            _, bss_R = brier_skill_score(y_R, p_R)
            _, bss_F = brier_skill_score(y_F, p_F) if len(y_F) else (None, float("nan"))
            _, bss_all = brier_skill_score(y_all, np.concatenate([p_R, p_F]))
            results[label].append(bss_all)
            if len(y_F) > 0:
                f_results[label].append(bss_F)
            print(f"  {label:10s}  bss_R={bss_R:,.0f}  bss_F={bss_F:,.0f}  ENSEMBLE={bss_all:,.0f}")

    print("\n[SUMMARY across 3 splits]")
    for label in results:
        print(f"  {label:10s}  AVG ENSEMBLE={np.mean(results[label]):,.0f}  per-split={[round(v) for v in results[label]]}")
    print("\n[SUMMARY F-only, split3 (only valid F split)]")
    for label in f_results:
        if f_results[label]:
            print(f"  {label:10s}  bss_F={f_results[label][0]:,.0f}")


if __name__ == "__main__":
    main()
