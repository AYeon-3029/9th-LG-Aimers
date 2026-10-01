"""teammate/script.py의 정확히 같은 피처 엔지니어링/모델 설정을, 우리 R walk-forward
분할(train<=2021/22/23 -> val=2022/23/24)로 재현한다. 지금은 'Futures' 버그를 그대로
둔 채(사용자 지시 - 먼저 있는 그대로 블렌딩 효과부터 본다) 3개 구간 각각에서:
  1) teammate 단독(LGB+XGB+HistGB 1:1:1) BSS
  2) 우리 모델 단독(기존 프로덕션 R 5-way/F 4-way) BSS
  3) 단순 평균 블렌드 BSS
를 R/F 각각, 그리고 pooled로 비교한다.
"""
from lightgbm import LGBMClassifier  # import 순서 안전
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
from common import (
    FEATURES, FEATURES_F, TARGET, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS,
    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, LGB_BASE_PARAMS, XGB_BASE_PARAMS,
    CAT_BASE_PARAMS, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
    brier_skill_score, split_regime, build_id_vocab,
)

DATA_DIR = "./data"
ID_COL = "row_id"
CAT_COLS = ["top_bottom", "base_state"]
FIX_FUTURES_BUG = True  # 'Futures'->'F' 버그 수정 버전으로 재검증


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


def fit_teammate_league(train_df, val_df, features):
    X = train_df[features]
    y = train_df[TARGET]
    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), [c for c in features if c not in CAT_COLS]),
    ])
    X_pre = preprocessor.fit_transform(X)

    lgbm = LGBMClassifier(n_estimators=150, learning_rate=0.05, min_child_samples=50, random_state=42, n_jobs=4, verbose=-1)
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.05, min_child_weight=50, random_state=42, n_jobs=4, tree_method='hist', eval_metric='logloss')
    hgb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, min_samples_leaf=50, l2_regularization=0.1, random_state=42)
    lgbm.fit(X_pre, y)
    xgb.fit(X_pre, y)
    hgb.fit(X_pre, y)

    Xv_pre = preprocessor.transform(val_df[features])
    p = (lgbm.predict_proba(Xv_pre)[:, 1] + xgb.predict_proba(Xv_pre)[:, 1] + hgb.predict_proba(Xv_pre)[:, 1]) / 3.0
    return p


def main():
    t0 = time.time()
    print("loading full train.csv (all columns, teammate pipeline needs them all)...")
    full_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    full_df = add_teammate_features(full_df)
    DROP_COLS = [ID_COL, "game_type", "season", TARGET]
    tm_features = [c for c in full_df.columns if c not in DROP_COLS]
    print(f"loaded in {time.time()-t0:.0f}s, {len(full_df):,} rows, {len(tm_features)} teammate features")

    # 우리 모델용 로더(커스텀 컬럼셋)도 이미 로드된 full_df에서 재사용 - 중복 read 방지
    our_df = full_df.rename(columns={})  # 동일 df, common.py add_matchup_features만 추가 적용
    from common import add_matchup_features
    our_df = add_matchup_features(our_df)

    for cutoff, val_season in T.WALK_FORWARD_SPLITS:
        t_split = time.time()
        train_df = full_df[full_df["season"] <= cutoff]
        val_df = full_df[full_df["season"] == val_season]
        our_train = our_df[our_df["season"] <= cutoff]
        our_val = our_df[our_df["season"] == val_season]

        results = {}
        for lg in ["R", "F"]:
            tr_lg = train_df[train_df["game_type"] == lg]
            val_lg = val_df[val_df["game_type"] == lg]
            if len(val_lg) == 0:
                continue
            y_val = val_lg[TARGET].to_numpy()

            # 'Futures'->'F' 버그 수정판: F는 신체제(season>=2023)만 학습에 사용
            if lg == "F" and FIX_FUTURES_BUG:
                tr_lg_new = tr_lg[tr_lg["season"] >= 2023]
                tr_lg = tr_lg_new if len(tr_lg_new) > 0 else tr_lg
            p_tm = fit_teammate_league(tr_lg, val_lg, tm_features)
            _, bss_tm = brier_skill_score(y_val, p_tm)

            if lg == "R":
                our_tr_lg, _ = split_regime(our_train)  # R은 그대로, split_regime은 F만 필터링
            else:
                _, our_tr_lg = split_regime(our_train)  # F는 반드시 신체제(season>=2023)만 - 프로덕션과 동일
            our_val_lg = our_val[our_val["game_type"] == lg]
            if lg == "R":
                pitcher_vocab = build_id_vocab(our_tr_lg, "pitcher_id")
                batter_vocab = build_id_vocab(our_tr_lg, "batter_id")
                preds_ours = T.fit_ensemble_predictions(
                    our_tr_lg, our_val_lg, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
                    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS, FEATURES,
                    embed_config=(pitcher_vocab, batter_vocab, 3),
                )
            else:
                preds_ours = T.fit_ensemble_predictions(
                    our_tr_lg, our_val_lg, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
                    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, FEATURES_F,
                )
            p_ours = preds_ours.mean(axis=1)
            _, bss_ours = brier_skill_score(y_val, p_ours)

            p_blend = (p_tm + p_ours) / 2
            _, bss_blend = brier_skill_score(y_val, p_blend)

            results[lg] = (y_val, p_tm, p_ours, p_blend)
            print(f"[train<={cutoff} val={val_season}] {lg}: teammate={bss_tm:,.0f}  ours={bss_ours:,.0f}  "
                  f"avg-blend={bss_blend:,.0f}  (delta vs ours={bss_blend-bss_ours:+.0f})")

        if "R" in results and "F" in results:
            y_all = np.concatenate([results["R"][0], results["F"][0]])
            for i, name in enumerate(["teammate", "ours", "avg-blend"]):
                p_all = np.concatenate([results["R"][i+1], results["F"][i+1]])
                _, bss_all = brier_skill_score(y_all, p_all)
                print(f"  POOLED {name}: {bss_all:,.0f}")
        print(f"  (split took {time.time()-t_split:.0f}s)")


if __name__ == "__main__":
    main()
