"""R 내부 재분리 - 1단계 저비용 진단(2안, 사용자 지시): 역할 라벨을 만들지 않고
그 행 자체의 inning 값(이미 제공된 공식 피처)만으로 early(<=5이닝)/late(6이닝+)
두 구간으로 즉시 라우팅한다. 신호가 있는지만 빠르게 확인하는 목적 - 신호 있으면
투수별 지속 역할 라벨(1안, asof 기반)로 정교화, 없으면 이 방향 자체를 접는다
(사용자 지시).

트리는 이미 inning을 피처로 갖고 있어 상호작용을 스스로 학습할 수 있으므로,
"따로 쪼갠 모델"이 "한 모델 안에서 inning을 피처로 쓰는 것"보다 실제로 더
나은지가 핵심 질문이다(R/F 분리가 game_type을 피처로 넣는 것보다 나았던 것과
같은 종류의 질문). R/F 분리 때와 달리 F만큼 표본이 작지 않다는 것도 확인한다
(사용자 지시 - 가장 작은 서브그룹이 F 신체제 5.5만행 대비 얼마나 되는지).
"""
import lightgbm as lgb  # import 순서 안전
import numpy as np
import train as T
from common import FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, brier_skill_score, split_regime

INNING_THRESHOLD = 5  # early: inning<=5, late: inning>=6 - R 전체 기준 56%/44%로 비교적 균형


def main():
    df = T.load_data()

    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        tr_R, _ = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        y_val = val_R[TARGET].to_numpy()

        # 베이스라인: 단일 R 모델(현재 프로덕션과 동일 방식, inning은 이미 FEATURES 안에 있음)
        p_lgb = T.fit_lgb(tr_R, val_R, LGB_BASE_PARAMS, FEATURES)
        p_xgb = T.fit_xgb(tr_R, val_R, XGB_BASE_PARAMS, FEATURES)
        p_cat = T.fit_cat(tr_R, val_R, CAT_BASE_PARAMS, FEATURES)
        p_base = (p_lgb + p_xgb + p_cat) / 3
        _, bss_base = brier_skill_score(y_val, p_base)

        # early/late 완전 분리 - 각자 독립적으로 트리 3종 학습
        tr_early, tr_late = tr_R[tr_R["inning"] <= INNING_THRESHOLD], tr_R[tr_R["inning"] > INNING_THRESHOLD]
        val_early, val_late = val_R[val_R["inning"] <= INNING_THRESHOLD], val_R[val_R["inning"] > INNING_THRESHOLD]

        def fit_subset(tr_sub, val_sub):
            pl = T.fit_lgb(tr_sub, val_sub, LGB_BASE_PARAMS, FEATURES)
            px = T.fit_xgb(tr_sub, val_sub, XGB_BASE_PARAMS, FEATURES)
            pc = T.fit_cat(tr_sub, val_sub, CAT_BASE_PARAMS, FEATURES)
            return (pl + px + pc) / 3

        p_early = fit_subset(tr_early, val_early)
        p_late = fit_subset(tr_late, val_late)
        y_early, y_late = val_early[TARGET].to_numpy(), val_late[TARGET].to_numpy()
        p_split_combined = np.concatenate([p_early, p_late])
        y_split_combined = np.concatenate([y_early, y_late])
        _, bss_split = brier_skill_score(y_split_combined, p_split_combined)

        print(
            f"[train<={cutoff} val={val_season}] n_train_early={len(tr_early):,} n_train_late={len(tr_late):,}  "
            f"baseline(single R)={bss_base:,.0f}  split(early+late)={bss_split:,.0f}  delta={bss_split - bss_base:+.0f}"
        )

    print(f"\nF 신체제 표본 참고치: 55,696행 (train<=2023 기준)")


if __name__ == "__main__":
    main()
