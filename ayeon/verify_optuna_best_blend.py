"""optuna_r_search.py의 최고 trial(tree-3 블렌드 기준)이 실제 4모델(DCN 포함)
블렌드에서도 개선인지 재검증한다 - 오늘의 핵심 교훈(부분 앙상블 개선이 전체
블렌드 개선을 보장하지 않음)을 그대로 적용."""
import json
import numpy as np
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    brier_skill_score, split_regime,
)

with open("optuna_r_results.json") as f:
    results = json.load(f)
best = results["best_params"]

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

df = T.load_data()
baseline_list, new_list = [], []
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
    p_4way_base = (p_lgb_base + p_xgb_base + p_cat_base + p_dcn) / 4
    _, bss_base = brier_skill_score(y_val, p_4way_base)

    p_lgb_new = T.fit_lgb(tr_R, val_R, lgb_params, FEATURES)
    p_xgb_new = T.fit_xgb(tr_R, val_R, xgb_params, FEATURES)
    p_cat_new = T.fit_cat(tr_R, val_R, cat_params, FEATURES)
    p_4way_new = (p_lgb_new + p_xgb_new + p_cat_new + p_dcn) / 4
    _, bss_new = brier_skill_score(y_val, p_4way_new)

    baseline_list.append(bss_base)
    new_list.append(bss_new)
    print(f"[train<={cutoff} val={val_season}] 4-way baseline={bss_base:,.0f}  4-way optuna-tree={bss_new:,.0f}  delta={bss_new - bss_base:+.0f}")

print(f"\nAVG baseline={np.mean(baseline_list):,.1f}  AVG optuna-tree={np.mean(new_list):,.1f}  delta={np.mean(new_list) - np.mean(baseline_list):+.1f}")
