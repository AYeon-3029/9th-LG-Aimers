"""optuna_r_search.py의 최적 트리 파라미터를 프로덕션(fit_final)에 반영하기 전에,
LGB_FINAL_N_ESTIMATORS/XGB_FINAL_N_ESTIMATORS/CAT_FINAL_N_ESTIMATORS와 동일한
방식(train<=2023 -> val=2024 early stopping 기준 best_iteration)으로 새
파라미터에 맞는 반복 횟수를 다시 구한다 - 학습률이 바뀌면(특히 CAT: 0.03->0.059)
같은 고정 반복 횟수를 재사용하는 게 안전하지 않다."""
import lightgbm as lgb
import json
import xgboost as xgb
from catboost import CatBoostClassifier
import train as T
from common import FEATURES, TARGET, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, split_regime

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

df = T.load_data()
train_df = df[df["season"] <= 2023]
val_df = df[df["season"] == 2024]
tr_R, _ = split_regime(train_df)
val_R = val_df[val_df["game_type"] == "R"]

X, y = tr_R[FEATURES].to_numpy(), tr_R[TARGET].to_numpy()
Xv, yv = val_R[FEATURES].to_numpy(), val_R[TARGET].to_numpy()

lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=2000, verbose=-1)
lgb_model.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="binary_logloss",
              feature_name=list(FEATURES), callbacks=[lgb.early_stopping(50, verbose=False)])
print("LGB best_iteration:", lgb_model.best_iteration_)

xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=2000, early_stopping_rounds=50, verbosity=0)
xgb_model.fit(tr_R[FEATURES], tr_R[TARGET], eval_set=[(val_R[FEATURES], val_R[TARGET])], verbose=False)
print("XGB best_iteration:", xgb_model.best_iteration)

cat_model = CatBoostClassifier(**cat_params, iterations=2000, early_stopping_rounds=50)
cat_model.fit(tr_R[FEATURES], tr_R[TARGET], eval_set=(val_R[FEATURES], val_R[TARGET]))
print("CAT best_iteration:", cat_model.get_best_iteration())
