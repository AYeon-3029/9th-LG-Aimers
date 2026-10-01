"""LGB/XGB/CAT(R 전용) 하이퍼파라미터를 Optuna(TPE)로 체계적으로 재탐색한다.
지금까지는 대부분 수동으로 몇 개 값만 시도했다 - R 3구간 walk-forward의
3-tree 균등 블렌드 평균 BSS를 목적함수로 베이지안 최적화를 돌린다.

주의: 여기서 찾은 최적값은 "3-tree 블렌드" 기준이다. 오늘 R DCN 시드/F 아키텍처
사례에서 확인했듯 개별/부분 블렌드에서 좋아도 최종 4모델 블렌드에서는 다르게
움직일 수 있으므로, 여기서 나온 후보는 반드시 walk_forward_check(4모델 전체)로
재검증한 뒤에만 채택한다 - 이 스크립트 자체는 채택 여부를 결정하지 않는다.

현재 프로덕션 값(LGB/XGB/CAT_BASE_PARAMS)을 trial 0으로 enqueue해서 탐색 결과와
직접 비교 가능하게 한다.
"""
import json
import numpy as np
import optuna
import train as T
from common import (
    FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
    brier_skill_score, split_regime,
)

N_TRIALS = 30

df = T.load_data()
SPLITS_DATA = []
for cutoff, val_season in T.WALK_FORWARD_SPLITS:
    train_df = df[df["season"] <= cutoff]
    val_df = df[df["season"] == val_season]
    tr_R, _ = split_regime(train_df)
    val_R = val_df[val_df["game_type"] == "R"]
    SPLITS_DATA.append((tr_R, val_R))
    print(f"split train<={cutoff} val={val_season}: n_train_R={len(tr_R):,} n_val_R={len(val_R):,}")


def objective(trial):
    lgb_params = dict(
        LGB_BASE_PARAMS,
        num_leaves=trial.suggest_int("lgb_num_leaves", 15, 63),
        min_child_samples=trial.suggest_int("lgb_min_child_samples", 50, 400),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.08, log=True),
        feature_fraction=trial.suggest_float("lgb_feature_fraction", 0.4, 0.9),
        bagging_fraction=trial.suggest_float("lgb_bagging_fraction", 0.5, 1.0),
    )
    xgb_params = dict(
        XGB_BASE_PARAMS,
        max_depth=trial.suggest_int("xgb_max_depth", 3, 8),
        min_child_weight=trial.suggest_int("xgb_min_child_weight", 50, 400),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.4, 0.9),
    )
    cat_params = dict(
        CAT_BASE_PARAMS,
        depth=trial.suggest_int("cat_depth", 3, 8),
        l2_leaf_reg=trial.suggest_float("cat_l2", 10, 400, log=True),
        learning_rate=trial.suggest_float("cat_lr", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("cat_subsample", 0.5, 1.0),
        rsm=trial.suggest_float("cat_rsm", 0.4, 0.9),
    )

    bss_list = []
    for tr_R, val_R in SPLITS_DATA:
        p_lgb = T.fit_lgb(tr_R, val_R, lgb_params, FEATURES)
        p_xgb = T.fit_xgb(tr_R, val_R, xgb_params, FEATURES)
        p_cat = T.fit_cat(tr_R, val_R, cat_params, FEATURES)
        p = (p_lgb + p_xgb + p_cat) / 3
        _, bss = brier_skill_score(val_R[TARGET].to_numpy(), p)
        bss_list.append(bss)

    trial.set_user_attr("per_split", bss_list)
    print(f"[trial {trial.number}] per_split={[round(b) for b in bss_list]} avg={np.mean(bss_list):,.1f}")
    return float(np.mean(bss_list))


study = optuna.create_study(direction="maximize", study_name="r_tree_search")
study.enqueue_trial({
    "lgb_num_leaves": 31, "lgb_min_child_samples": 200, "lgb_lr": 0.03,
    "lgb_feature_fraction": 0.6, "lgb_bagging_fraction": 0.8,
    "xgb_max_depth": 5, "xgb_min_child_weight": 200, "xgb_lr": 0.03,
    "xgb_subsample": 0.8, "xgb_colsample": 0.6,
    "cat_depth": 5, "cat_l2": 200, "cat_lr": 0.03,
    "cat_subsample": 0.8, "cat_rsm": 0.6,
})
study.optimize(objective, n_trials=N_TRIALS)

print("\n=== DONE ===")
print("baseline (trial 0, current production params) value:", study.trials[0].value)
print("BEST VALUE:", study.best_value)
print("BEST PARAMS:", json.dumps(study.best_params, indent=2))
print("BEST TRIAL per_split:", study.best_trial.user_attrs.get("per_split"))

with open("optuna_r_results.json", "w") as f:
    json.dump({
        "baseline_value": study.trials[0].value,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params, "per_split": t.user_attrs.get("per_split")}
            for t in study.trials
        ],
    }, f, indent=2)
