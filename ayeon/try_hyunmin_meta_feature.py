"""Hyunmin의 OOF 예측 점수(hyunmin_pred_score) 딱 1개 컬럼만 우리 FEATURES에
추가해서 R 3구간 walk-forward, 트리 3종으로 스크리닝한다(사용자 지시 - 10개 피처는
섞지 말고 이 컬럼 하나만).

주의: hyunmin_train_meta.csv는 train.csv 전체 행을 커버하므로, 우리 walk-forward의
val_season에 해당하는 행도 이 파일에 포함돼 있다. 만약 그의 OOF가 랜덤 K-fold로
생성된 거라면(시즌 드리프트 누출 의심 - 모든 시즌에서 비현실적으로 높은 bss,
사전 확인 완료) val 구간에서도 이 컬럼이 실제 2025 제출 시나리오보다 유리하게
작용할 수 있다 - 로컬 결과를 곧이곧대로 못 믿을 수 있다는 걸 감안하고 해석할 것."""
import lightgbm as lgb  # import 순서 안전
import pandas as pd
import train as T
from common import FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, brier_skill_score, split_regime

NEW_FEAT = "hyunmin_pred_score"


def main():
    df = T.load_data()
    row_ids = pd.read_csv(T.DATA_DIR / "train.csv", usecols=["row_id"])
    df = df.reset_index(drop=True)
    df["row_id"] = row_ids["row_id"]

    meta = pd.read_csv("../hyunmin/hyunmin_train_meta.csv")
    df = df.merge(meta, on="row_id", how="left")
    print(f"unmatched after merge: {df[NEW_FEAT].isna().sum()}")

    feat_with_new = FEATURES + [NEW_FEAT]

    print("=== R walk-forward 3-split, tree ensemble only (+hyunmin_pred_score) ===")
    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]
        train_R, _ = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        y_val = val_R[TARGET].to_numpy()

        p_lgb_b = T.fit_lgb(train_R, val_R, LGB_BASE_PARAMS, FEATURES)
        p_xgb_b = T.fit_xgb(train_R, val_R, XGB_BASE_PARAMS, FEATURES)
        p_cat_b = T.fit_cat(train_R, val_R, CAT_BASE_PARAMS, FEATURES)
        p_base = (p_lgb_b + p_xgb_b + p_cat_b) / 3
        _, bss_base = brier_skill_score(y_val, p_base)

        p_lgb_n = T.fit_lgb(train_R, val_R, LGB_BASE_PARAMS, feat_with_new)
        p_xgb_n = T.fit_xgb(train_R, val_R, XGB_BASE_PARAMS, feat_with_new)
        p_cat_n = T.fit_cat(train_R, val_R, CAT_BASE_PARAMS, feat_with_new)
        p_new = (p_lgb_n + p_xgb_n + p_cat_n) / 3
        _, bss_new = brier_skill_score(y_val, p_new)

        print(f"[train<={cutoff} val={val_season}] baseline={bss_base:,.0f}  +hyunmin_score={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")


if __name__ == "__main__":
    main()
