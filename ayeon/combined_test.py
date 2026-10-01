"""Optuna 재탐색 트리 파라미터 + pitcher/batter 임베딩(5번째 멤버)을 함께 테스트한다.
두 개선이 서로 다른 축(트리 하이퍼파라미터 vs 새 앙상블 멤버)이라 상쇄 없이
합산될 가능성이 높다는 가설을 R 3구간 walk-forward로 직접 확인한다.

세 가지를 함께 비교:
  baseline   = 원래 트리 파라미터 + DCN (4-way, 현재 프로덕션)
  optuna     = optuna 트리 파라미터 + DCN (4-way, 이미 확인: +12/+9/+1)
  combined   = optuna 트리 파라미터 + DCN + embedding (5-way, 신규)
"""
import json
import numpy as np
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    brier_skill_score, split_regime,
)
from try_pitcher_batter_embedding import (
    load_data_with_ids, build_vocab, fit_embed_multiseed,
)

with open("optuna_r_results.json") as f:
    best = json.load(f)["best_params"]

lgb_params = dict(
    LGB_BASE_PARAMS, num_leaves=best["lgb_num_leaves"], min_child_samples=best["lgb_min_child_samples"],
    learning_rate=best["lgb_lr"], feature_fraction=best["lgb_feature_fraction"], bagging_fraction=best["lgb_bagging_fraction"],
)
xgb_params = dict(
    XGB_BASE_PARAMS, max_depth=best["xgb_max_depth"], min_child_weight=best["xgb_min_child_weight"],
    learning_rate=best["xgb_lr"], subsample=best["xgb_subsample"], colsample_bytree=best["xgb_colsample"],
)
cat_params = dict(
    CAT_BASE_PARAMS, depth=best["cat_depth"], l2_leaf_reg=best["cat_l2"],
    learning_rate=best["cat_lr"], subsample=best["cat_subsample"], rsm=best["cat_rsm"],
)

df = load_data_with_ids()
base_list, optuna_list, combined_list = [], [], []
for cutoff, val_season in T.WALK_FORWARD_SPLITS:
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]
    tr_R, _ = split_regime(train_df)
    val_R = val_df[val_df["game_type"] == "R"]
    y_val = val_R[TARGET].to_numpy()

    p_dcn = T.fit_dcn_multiseed(tr_R, val_R, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS)

    p_lgb_base = T.fit_lgb(tr_R, val_R, LGB_BASE_PARAMS, FEATURES)
    p_xgb_base = T.fit_xgb(tr_R, val_R, XGB_BASE_PARAMS, FEATURES)
    p_cat_base = T.fit_cat(tr_R, val_R, CAT_BASE_PARAMS, FEATURES)
    p_base = (p_lgb_base + p_xgb_base + p_cat_base + p_dcn) / 4
    _, bss_base = brier_skill_score(y_val, p_base)

    p_lgb_new = T.fit_lgb(tr_R, val_R, lgb_params, FEATURES)
    p_xgb_new = T.fit_xgb(tr_R, val_R, xgb_params, FEATURES)
    p_cat_new = T.fit_cat(tr_R, val_R, cat_params, FEATURES)
    p_optuna = (p_lgb_new + p_xgb_new + p_cat_new + p_dcn) / 4
    _, bss_optuna = brier_skill_score(y_val, p_optuna)

    pitcher_vocab = build_vocab(tr_R, "pitcher_id")
    batter_vocab = build_vocab(tr_R, "batter_id")
    p_embed = fit_embed_multiseed(tr_R, val_R, pitcher_vocab, batter_vocab)
    p_combined = (p_lgb_new + p_xgb_new + p_cat_new + p_dcn + p_embed) / 5
    _, bss_combined = brier_skill_score(y_val, p_combined)

    base_list.append(bss_base)
    optuna_list.append(bss_optuna)
    combined_list.append(bss_combined)
    print(
        f"[train<={cutoff} val={val_season}] baseline={bss_base:,.0f}  "
        f"optuna-4way={bss_optuna:,.0f}(+{bss_optuna - bss_base:.0f})  "
        f"combined-5way={bss_combined:,.0f}(+{bss_combined - bss_base:.0f} vs base, "
        f"{bss_combined - bss_optuna:+.0f} vs optuna-only)"
    )

print(f"\nAVG baseline={np.mean(base_list):,.1f}  AVG optuna={np.mean(optuna_list):,.1f}  AVG combined={np.mean(combined_list):,.1f}")
print(f"combined vs baseline: {np.mean(combined_list) - np.mean(base_list):+.1f}")
print(f"combined vs optuna-only (embedding's marginal contribution): {np.mean(combined_list) - np.mean(optuna_list):+.1f}")
