"""팀원 파이프라인을 R/F 통합(단일 모델, 전체 train_df로 한 번에 학습) 버전으로
walk-forward 재현한다 - "800점 제출이 사실 통합 학습이었을 수도 있다"는 가설을
확인하기 위한 추가 데이터 포인트. game_type은 원래도 FEATURES에서 빠져있으므로
그대로 유지(리그 정보 없이 통합 학습). 채점만 R/F로 나눠서 본다."""
from lightgbm import LGBMClassifier
import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

import sys
sys.path.insert(0, "../LGAimers")
import train as T
from common import TARGET, brier_skill_score

DATA_DIR = "./data"
ID_COL = "row_id"
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


def main():
    t0 = time.time()
    full_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    full_df = add_teammate_features(full_df)
    # 'Futures' 버그 수정 적용(F만) - 원래 그들의 의도(2023년 이전 F 제거)를 살리되
    # R/F를 통합 학습하는 구조이므로 이 필터는 "F 오염 행만 통째로 빼는" 형태로 적용
    DROP_COLS = [ID_COL, "game_type", "season", TARGET]
    tm_features = [c for c in full_df.columns if c not in DROP_COLS]
    print(f"loaded in {time.time()-t0:.0f}s, {len(full_df):,} rows, {len(tm_features)} features")

    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        t_split = time.time()
        train_df = full_df[full_df["season"] <= cutoff]
        # 버그 수정판: F는 2023+ 만 남기고, R은 전부 포함 - 통합 학습셋 하나로 합침
        train_df = train_df[(train_df["game_type"] == "R") | (train_df["season"] >= 2023)]
        val_df = full_df[full_df["season"] == val_season]

        X = train_df[tm_features]
        y = train_df[TARGET]
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

        Xv_pre = preprocessor.transform(val_df[tm_features])
        p_lgb = lgbm.predict_proba(Xv_pre)[:, 1]
        p_xgb = xgb.predict_proba(Xv_pre)[:, 1]
        p_hgb = hgb.predict_proba(Xv_pre)[:, 1]
        p_val = (p_lgb + p_xgb + p_hgb) / 3.0

        is_R = (val_df["game_type"] == "R").to_numpy()
        y_val = val_df[TARGET].to_numpy()

        _, bss_all = brier_skill_score(y_val, p_val)
        _, bss_R = brier_skill_score(y_val[is_R], p_val[is_R])
        if (~is_R).sum() > 0:
            _, bss_F = brier_skill_score(y_val[~is_R], p_val[~is_R])
        else:
            bss_F = float("nan")

        print(f"[train<={cutoff} val={val_season}] n_train={len(train_df):,} (R+F unified)  "
              f"POOLED={bss_all:,.0f}  R={bss_R:,.0f}  F={bss_F:,.0f}  (took {time.time()-t_split:.0f}s)")


if __name__ == "__main__":
    main()
