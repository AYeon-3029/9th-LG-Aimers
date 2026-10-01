"""F(퓨처스) LGB/XGB/CAT 하이퍼파라미터를 Optuna(TPE)로 재탐색한다. R과 달리
목적함수를 처음부터 whole-split 4-way(DCN 포함) 블렌드 BSS로 잡는다 - F DCN
아키텍처 건에서 "월별로는 좋아 보였는데 whole-split 블렌드에서는 손해"였던
실패를 반복하지 않기 위해서다(오늘자 model-state 기록 참고). F는 신체제
데이터가 1년(2024)뿐이라 walk-forward가 구조적으로 train<=2023->val=2024
한 구간만 가능하다(validation-lessons 근거) - 월별 4구간 breakdown은 방향
일관성 참고용으로만 같이 출력하고, 실제 채택 판단(및 Optuna의 최적화 목표)은
whole-split 숫자로만 한다.

F_PARAMS는 이미 R_BASE_PARAMS와 완전히 독립돼 있으므로(오늘 R 재탐색 때 고친
것) 여기서 R을 건드릴 걱정 없이 안전하게 실험할 수 있다."""
import json
import numpy as np
import optuna
import train as T
from common import (
    FEATURES_F, TARGET, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
    brier_skill_score, split_regime,
)

N_TRIALS = 30
F_SEGMENTS = T.F_SEGMENTS

df = T.load_data()
train_df = df[df["season"] <= 2023]
val_df = df[df["season"] == 2024]
_, tr_F = split_regime(train_df)
val_F = val_df[val_df["game_type"] == "F"]
y_F = val_F[TARGET].to_numpy()
game_month_F = val_F["game_month"].to_numpy()
print(f"n_train_F={len(tr_F):,}  n_val_F={len(val_F):,}")

# DCN은 이번 탐색 대상이 아니다(오늘 이미 원래 아키텍처로 확정) - 한 번만 학습해서
# 모든 trial이 재사용한다(재학습 비용 절감, 트리 파라미터와 독립적).
p_dcn_F = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)


def objective(trial):
    lgb_params = dict(
        LGB_F_PARAMS,
        num_leaves=trial.suggest_int("lgb_num_leaves", 5, 31),
        min_child_samples=trial.suggest_int("lgb_min_child_samples", 50, 400),
        learning_rate=trial.suggest_float("lgb_lr", 0.01, 0.08, log=True),
        feature_fraction=trial.suggest_float("lgb_feature_fraction", 0.4, 0.9),
        bagging_fraction=trial.suggest_float("lgb_bagging_fraction", 0.5, 1.0),
    )
    xgb_params = dict(
        XGB_F_PARAMS,
        max_depth=trial.suggest_int("xgb_max_depth", 2, 6),
        min_child_weight=trial.suggest_int("xgb_min_child_weight", 50, 400),
        learning_rate=trial.suggest_float("xgb_lr", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("xgb_subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("xgb_colsample", 0.4, 0.9),
    )
    cat_params = dict(
        CAT_F_PARAMS,
        depth=trial.suggest_int("cat_depth", 2, 6),
        l2_leaf_reg=trial.suggest_float("cat_l2", 10, 400, log=True),
        learning_rate=trial.suggest_float("cat_lr", 0.01, 0.08, log=True),
        subsample=trial.suggest_float("cat_subsample", 0.5, 1.0),
        rsm=trial.suggest_float("cat_rsm", 0.4, 0.9),
    )

    p_lgb = T.fit_lgb(tr_F, val_F, lgb_params, FEATURES_F)
    p_xgb = T.fit_xgb(tr_F, val_F, xgb_params, FEATURES_F)
    p_cat = T.fit_cat(tr_F, val_F, cat_params, FEATURES_F)
    p_blend = (p_lgb + p_xgb + p_cat + p_dcn_F) / 4
    _, bss_whole = brier_skill_score(y_F, p_blend)

    seg_bss = []
    for months in F_SEGMENTS:
        mask = np.isin(game_month_F, months)
        if mask.sum() == 0:
            continue
        _, b = brier_skill_score(y_F[mask], p_blend[mask])
        seg_bss.append(b)
    trial.set_user_attr("seg_bss", seg_bss)
    print(f"[trial {trial.number}] whole-split bss_F={bss_whole:,.0f}  segments={[round(b) for b in seg_bss]}")
    return float(bss_whole)


study = optuna.create_study(direction="maximize", study_name="f_tree_search")
study.enqueue_trial({
    "lgb_num_leaves": 15, "lgb_min_child_samples": 100, "lgb_lr": 0.03,
    "lgb_feature_fraction": 0.6, "lgb_bagging_fraction": 0.8,
    "xgb_max_depth": 3, "xgb_min_child_weight": 100, "xgb_lr": 0.03,
    "xgb_subsample": 0.8, "xgb_colsample": 0.6,
    "cat_depth": 3, "cat_l2": 100, "cat_lr": 0.03,
    "cat_subsample": 0.8, "cat_rsm": 0.6,
})
study.optimize(objective, n_trials=N_TRIALS)

print("\n=== DONE ===")
print("baseline (trial 0, current production F params) whole-split bss_F:", study.trials[0].value)
print("baseline segments:", study.trials[0].user_attrs.get("seg_bss"))
print("BEST whole-split bss_F:", study.best_value)
print("BEST PARAMS:", json.dumps(study.best_params, indent=2))
print("BEST TRIAL segments:", study.best_trial.user_attrs.get("seg_bss"))

with open("optuna_f_results.json", "w") as f:
    json.dump({
        "baseline_value": study.trials[0].value,
        "baseline_segments": study.trials[0].user_attrs.get("seg_bss"),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_segments": study.best_trial.user_attrs.get("seg_bss"),
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params, "seg_bss": t.user_attrs.get("seg_bss")}
            for t in study.trials
        ],
    }, f, indent=2)
