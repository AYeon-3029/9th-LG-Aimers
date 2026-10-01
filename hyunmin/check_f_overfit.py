"""팀원 F(수정판) 모델이 왜 baseline을 못 넘는지 진단: train(in-sample) Brier vs
val(held-out) Brier 격차를 본다 - 격차가 크면 과적합, 격차가 작으면 다른 원인
(하이퍼파라미터 문제가 아니라 애초에 신호 자체가 이 표본에서 안 잡히는 것)이다.
우리 F 하이퍼파라미터(LGB num_leaves=15/min_child=100 등, 훨씬 보수적)와도 직접 대조."""
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
print(f"n_features={len(tm_features)}")

train_df = full_df[full_df["season"] <= 2023]
val_df = full_df[full_df["season"] == 2024]
tr_F = train_df[(train_df["game_type"] == "F") & (train_df["season"] >= 2023)]
val_F = val_df[val_df["game_type"] == "F"]
y_val = val_F["control_success"].to_numpy()
y_tr = tr_F["control_success"].to_numpy()

baseline_val = brier_score_loss(y_val, np.full(len(y_val), y_val.mean()))
baseline_tr = brier_score_loss(y_tr, np.full(len(y_tr), y_tr.mean()))

X = tr_F[tm_features]
y = tr_F["control_success"]
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
    ("num", SimpleImputer(strategy="median"), [c for c in tm_features if c not in CAT_COLS]),
])
X_pre = preprocessor.fit_transform(X)
Xv_pre = preprocessor.transform(val_F[tm_features])

print("\n=== teammate 원래 설정 (덜 보수적) ===")
models = {
    "LGB(teammate)": LGBMClassifier(n_estimators=150, learning_rate=0.05, min_child_samples=50, random_state=42, n_jobs=4, verbose=-1),
    "XGB(teammate)": XGBClassifier(n_estimators=150, learning_rate=0.05, min_child_weight=50, random_state=42, n_jobs=4, tree_method='hist', eval_metric='logloss'),
    "HGB(teammate)": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, min_samples_leaf=50, l2_regularization=0.1, random_state=42),
}
for name, m in models.items():
    m.fit(X_pre, y)
    p_tr = m.predict_proba(X_pre)[:, 1]
    p_val = m.predict_proba(Xv_pre)[:, 1]
    brier_tr = brier_score_loss(y_tr, p_tr)
    brier_val = brier_score_loss(y_val, p_val)
    bss_tr = 100000 * (1 - brier_tr / baseline_tr)
    bss_val = 100000 * (1 - brier_val / baseline_val)
    print(f"  {name:16s}  TRAIN bss(unclipped)={bss_tr:>7,.0f}  VAL bss(unclipped)={bss_val:>7,.0f}  gap={bss_tr-bss_val:>7,.0f}")

print("\n=== 우리 F 하이퍼파라미터로 교체(num_leaves=15/min_child_samples=100 등, 훨씬 보수적) ===")
models_conservative = {
    "LGB(ours-style)": LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=15, min_child_samples=100, random_state=42, n_jobs=4, verbose=-1),
    "XGB(ours-style)": XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=3, min_child_weight=100, random_state=42, n_jobs=4, tree_method='hist', eval_metric='logloss'),
}
for name, m in models_conservative.items():
    m.fit(X_pre, y)
    p_tr = m.predict_proba(X_pre)[:, 1]
    p_val = m.predict_proba(Xv_pre)[:, 1]
    brier_tr = brier_score_loss(y_tr, p_tr)
    brier_val = brier_score_loss(y_val, p_val)
    bss_tr = 100000 * (1 - brier_tr / baseline_tr)
    bss_val = 100000 * (1 - brier_val / baseline_val)
    print(f"  {name:16s}  TRAIN bss(unclipped)={bss_tr:>7,.0f}  VAL bss(unclipped)={bss_val:>7,.0f}  gap={bss_tr-bss_val:>7,.0f}")
