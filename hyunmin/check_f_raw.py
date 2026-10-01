"""팀원 F 모델의 bss_F=0이 진짜 심각한 붕괴인지, 아니면 baseline과 거의 동률이라
클리핑된 것뿐인지 raw Brier과 예측 분포로 직접 확인한다. train<=2023->val=2024
(F 신호가 있는 유일한 구간)만, 버그 수정판으로."""
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

DATA_DIR = "./data"
CAT_COLS = ["top_bottom", "base_state"]


def add_teammate_features(df):
    df = df.copy()
    df['crisis_pressure'] = df['li'] * (df['num_runners_on'] + 1)
    df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(int)
    df['pitcher_ahead'] = (df['strikes_before'] > df['balls_before']).astype(int)
    df['batter_ahead'] = (df['balls_before'] > df['strikes_before']).astype(int)
    max_rate = df[['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate']].max(axis=1)
    df['max_pitch_rate'] = max_rate.fillna(0)
    df['stubbornness_index'] = df['max_pitch_rate'] * df['li']
    df['control_struggle_index'] = df['asof_pitcher_ball_rate'].fillna(0) * df['li']
    df['is_scoring_pos'] = ((df['runner_on_2b'] == 1) | (df['runner_on_3b'] == 1)).astype(int)
    df['pitcher_batter_diff'] = df['asof_pitcher_success_rate'].fillna(0.5) - df['asof_batter_success_rate'].fillna(0.5)
    df['recent_trend'] = (df['asof_pitcher_prev1_game_success_rate'].fillna(df['asof_pitcher_success_rate']) - df['asof_pitcher_success_rate'].fillna(0.5)).fillna(0)
    df['is_same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)
    for c in CAT_COLS:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    return df


full_df = pd.read_csv(f"{DATA_DIR}/train.csv")
full_df = add_teammate_features(full_df)
DROP_COLS = ["row_id", "game_type", "season", "control_success"]
tm_features = [c for c in full_df.columns if c not in DROP_COLS]

train_df = full_df[full_df["season"] <= 2023]
val_df = full_df[full_df["season"] == 2024]
tr_F_buggy = train_df[train_df["game_type"] == "F"]                      # 버그: 필터 없음
tr_F_fixed = train_df[(train_df["game_type"] == "F") & (train_df["season"] >= 2023)]  # 수정판
val_F = val_df[val_df["game_type"] == "F"]
y_val = val_F["control_success"].to_numpy()

print(f"n_train_F buggy(unfiltered)={len(tr_F_buggy):,}  fixed(2023+)={len(tr_F_fixed):,}  n_val_F={len(val_F):,}")
print(f"actual val_F success rate: {y_val.mean():.4f}")

baseline_pred = np.full(len(y_val), y_val.mean())
baseline_brier = brier_score_loss(y_val, baseline_pred)
print(f"baseline (mean-only) Brier: {baseline_brier:.6f}")

for label, tr_F in [("BUGGY(unfiltered)", tr_F_buggy), ("FIXED(2023+ only)", tr_F_fixed)]:
    X = tr_F[tm_features]
    y = tr_F["control_success"]
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), [c for c in tm_features if c not in CAT_COLS]),
    ])
    X_pre = preprocessor.fit_transform(X)
    lgbm = LGBMClassifier(n_estimators=150, learning_rate=0.05, min_child_samples=50, random_state=42, n_jobs=4, verbose=-1)
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, min_child_weight=50, random_state=42, n_jobs=4, tree_method='hist', eval_metric='logloss')
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, min_samples_leaf=50, l2_regularization=0.1, random_state=42)
    lgbm.fit(X_pre, y)
    xgb.fit(X_pre, y)
    hgb.fit(X_pre, y)

    Xv_pre = preprocessor.transform(val_F[tm_features])
    p_lgb = lgbm.predict_proba(Xv_pre)[:, 1]
    p_xgb = xgb.predict_proba(Xv_pre)[:, 1]
    p_hgb = hgb.predict_proba(Xv_pre)[:, 1]
    p_blend = (p_lgb + p_xgb + p_hgb) / 3.0

    print(f"\n=== {label} (n_train_F={len(tr_F):,}) ===")
    for name, p in [("LGB", p_lgb), ("XGB", p_xgb), ("HGB", p_hgb), ("1:1:1 blend", p_blend)]:
        brier = brier_score_loss(y_val, p)
        bss_raw = 100000 * (1 - brier / baseline_brier)  # 클리핑 없이
        print(f"  {name:12s}  pred_mean={p.mean():.4f}  pred_std={p.std():.4f}  "
              f"pred_range=[{p.min():.4f},{p.max():.4f}]  Brier={brier:.6f}  "
              f"bss_raw(unclipped)={bss_raw:,.0f}  bss_clipped={max(0,bss_raw):,.0f}")
