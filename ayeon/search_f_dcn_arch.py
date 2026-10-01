"""2순위: F(퓨처스) DCNv2 아키텍처 재탐색.

R의 DCN 아키텍처는 그리드서치(common.py 근거)로 cross_layers=2,
deep_dims=(256,128)을 확정했지만, F의 현재 값(cross_layers=2, deep_dims=(32,16))은
한 번도 이렇게 재탐색된 적이 없다 - 처음 정할 때 "R 기준값을 작게 스케일"한
추정치였을 뿐이다.

F는 신체제 데이터가 1년(2024)뿐이라 walk-forward 3구간이 안 되므로(HANDOFF.md
5-2절, memory: lgaimers-validation-lessons #2), train<=2023->val=2024 딱 한
구간만 쓰되 val_F(2024)를 월 단위 4구간(3-4/5-6/7-8/9-10월)으로 쪼개 방향
일관성을 반드시 확인한다 - 전체 평균 하나만 보고 판단하지 않는다.

탐색 단계는 시드 3개 평균(R 아키텍처 탐색 때와 동일 관례, common.py 근거)으로
비용을 억제하고, 최종 후보만 프로덕션 설정(DCN_F_N_SEEDS=5)으로 다시 검증한다.
"""
import train as T  # lightgbm을 가장 먼저 import해야 하는 이 환경의 이슈 회피
import numpy as np
from common import (
    FEATURES_F, TARGET, add_shrunk_pitcher_feature, compute_season_means,
    split_regime, brier_skill_score, DCN_F_BATCH_SIZE,
)

SEARCH_N_SEEDS = 5  # 최종 확인 - 프로덕션 시드 수(DCN_F_N_SEEDS=5)와 맞춘다
CANDIDATES = [
    ("baseline (cross=2, dims=(32,16))", dict(cross_layers=2, deep_dims=(32, 16), dropout=0.2)),
    ("winner (cross=0, dims=(16,8))", dict(cross_layers=0, deep_dims=(16, 8), dropout=0.2)),
]


def main():
    df = T.load_data()
    cutoff, val_season = 2023, 2024
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]

    season_means = compute_season_means(train_df)
    train_df = add_shrunk_pitcher_feature(train_df, season_means)
    val_df = add_shrunk_pitcher_feature(val_df, season_means)

    _, tr_F = split_regime(train_df)
    val_F = val_df[val_df["game_type"] == "F"]
    y_F = val_F[TARGET].to_numpy()
    game_month_F = val_F["game_month"].to_numpy()
    print(f"n_train_F={len(tr_F):,}  n_val_F={len(val_F):,}")

    for name, params in CANDIDATES:
        preds = [T.fit_dcn(tr_F, val_F, params, DCN_F_BATCH_SIZE, seed=s) for s in range(SEARCH_N_SEEDS)]
        p = np.mean(preds, axis=0)
        _, bss_all = brier_skill_score(y_F, p)
        seg_str = []
        seg_bss = []
        for months in T.F_SEGMENTS:
            mask = np.isin(game_month_F, months)
            if mask.sum() == 0:
                continue
            _, b = brier_skill_score(y_F[mask], p[mask])
            seg_bss.append(b)
            seg_str.append(f"{months}={b:,.0f}")
        n_pos = sum(1 for b in seg_bss if b > 0)
        print(f"[{name}] overall={bss_all:,.0f}  segments: {' '.join(seg_str)}  (n_seg_nonzero={n_pos}/{len(seg_bss)})")


if __name__ == "__main__":
    main()
