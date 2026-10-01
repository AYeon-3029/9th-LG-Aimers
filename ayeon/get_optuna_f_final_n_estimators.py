"""F Optuna 최적 파라미터에 맞는 *_F_FINAL_N_ESTIMATORS를 재도출한다 - R과 동일한
방식(train<=2023 -> val=2024 early stopping 기준 best_iteration)."""
import lightgbm as lgb
import json
import xgboost as xgb
from catboost import CatBoostClassifier
import train as T
from common import FEATURES_F, TARGET, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS, split_regime

with open("optuna_f_results.json") as f:
    best = json.load(f)["best_params"]

lgb_params = dict(
    LGB_F_PARAMS, num_leaves=best["lgb_num_leaves"], min_child_samples=best["lgb_min_child_samples"],
    learning_rate=best["lgb_lr"], feature_fraction=best["lgb_feature_fraction"], bagging_fraction=best["lgb_bagging_fraction"],
)
xgb_params = dict(
    XGB_F_PARAMS, max_depth=best["xgb_max_depth"], min_child_weight=best["xgb_min_child_weight"],
    learning_rate=best["xgb_lr"], subsample=best["xgb_subsample"], colsample_bytree=best["xgb_colsample"],
)
cat_params = dict(
    CAT_F_PARAMS, depth=best["cat_depth"], l2_leaf_reg=best["cat_l2"],
    learning_rate=best["cat_lr"], subsample=best["cat_subsample"], rsm=best["cat_rsm"],
)

df = T.load_data()
train_df = df[df["season"] <= 2023]
val_df = df[df["season"] == 2024]
_, tr_F = split_regime(train_df)
val_F = val_df[val_df["game_type"] == "F"]

X, y = tr_F[FEATURES_F].to_numpy(), tr_F[TARGET].to_numpy()
Xv, yv = val_F[FEATURES_F].to_numpy(), val_F[TARGET].to_numpy()

lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=2000, verbose=-1)
lgb_model.fit(X, y, eval_X=Xv, eval_y=yv, eval_metric="binary_logloss",
              feature_name=list(FEATURES_F), callbacks=[lgb.early_stopping(50, verbose=False)])
print("LGB_F best_iteration:", lgb_model.best_iteration_)

xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=2000, early_stopping_rounds=50, verbosity=0)
xgb_model.fit(tr_F[FEATURES_F], tr_F[TARGET], eval_set=[(val_F[FEATURES_F], val_F[TARGET])], verbose=False)
print("XGB_F best_iteration:", xgb_model.best_iteration)

cat_model = CatBoostClassifier(**cat_params, iterations=2000, early_stopping_rounds=50)
cat_model.fit(tr_F[FEATURES_F], tr_F[TARGET], eval_set=(val_F[FEATURES_F], val_F[TARGET]))
print("CAT_F best_iteration:", cat_model.get_best_iteration())
