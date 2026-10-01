"""1단계: 현민 vs 우리(현재 프로덕션, R=Optuna트리+임베딩 5-way / F=임베딩 5-way)
예측을 R/F 각각 walk-forward로 재생성하고, 결과를 캐싱한 뒤 볼록결합(비음수,
합=1, Brier 직접최소화) 최적 가중치를 구한다. F는 'Futures'->'F' 버그를
수정한 버전을 쓴다(이미 확정된 판단 - common.py 로그 근거)."""
from lightgbm import LGBMClassifier  # import 순서 안전
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
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
    EMBED_R_DIM, EMBED_R_HIDDEN, EMBED_R_DROPOUT, EMBED_R_BATCH_SIZE, EMBED_R_N_SEEDS,
    EMBED_F_DIM, EMBED_F_HIDDEN, EMBED_F_DROPOUT, EMBED_F_BATCH_SIZE, EMBED_F_N_SEEDS,
    brier_skill_score, split_regime, build_id_vocab, add_matchup_features,
)

DATA_DIR = "./data"
ID_COL = "row_id"
CAT_COLS = ["top_bottom", "base_state"]
F_SEGMENTS = T.F_SEGMENTS
CACHE_PATH = "convex_blend_cache.npz"


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
    return (lgbm.predict_proba(Xv_pre)[:, 1] + xgb.predict_proba(Xv_pre)[:, 1] + hgb.predict_proba(Xv_pre)[:, 1]) / 3.0


def convex_weight(p1, p2, y):
    """비음수, 합=1 제약으로 Brier 직접 최소화 - 기존에 R/F 앙상블 가중치
    검증에 썼던 것과 동일한 방법론."""
    def objective(w):
        p = w[0] * p1 + w[1] * p2
        return np.mean((p - y) ** 2)
    result = minimize(
        objective, x0=[0.5, 0.5], bounds=[(0, 1), (0, 1)], method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    return result.x


def generate_and_cache():
    t0 = time.time()
    print("loading full train.csv...")
    full_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    full_df = add_teammate_features(full_df)
    DROP_COLS = [ID_COL, "game_type", "season", TARGET]
    tm_features = [c for c in full_df.columns if c not in DROP_COLS]
    our_df = add_matchup_features(full_df)
    print(f"loaded in {time.time()-t0:.0f}s")

    cache = {}
    for split_i, (cutoff, val_season) in enumerate(T.WALK_FORWARD_SPLITS):
        t_split = time.time()
        train_df = full_df[full_df["season"] <= cutoff]
        val_df = full_df[full_df["season"] == val_season]
        our_train = our_df[our_df["season"] <= cutoff]
        our_val = our_df[our_df["season"] == val_season]

        for lg in ["R", "F"]:
            tr_lg = train_df[train_df["game_type"] == lg]
            val_lg = val_df[val_df["game_type"] == lg]
            if len(val_lg) == 0:
                continue
            y_val = val_lg[TARGET].to_numpy()
            game_month = val_lg["game_month"].to_numpy()

            if lg == "F":
                tr_lg_new = tr_lg[tr_lg["season"] >= 2023]  # 'Futures' 버그 수정판
                tr_lg = tr_lg_new if len(tr_lg_new) > 0 else tr_lg
            p_tm = fit_teammate_league(tr_lg, val_lg, tm_features)

            if lg == "R":
                our_tr_lg, _ = split_regime(our_train)
            else:
                _, our_tr_lg = split_regime(our_train)
            our_val_lg = our_val[our_val["game_type"] == lg]

            if lg == "R":
                pitcher_vocab = build_id_vocab(our_tr_lg, "pitcher_id")
                batter_vocab = build_id_vocab(our_tr_lg, "batter_id")
                embed_config = dict(
                    pitcher_vocab=pitcher_vocab, batter_vocab=batter_vocab, n_seeds=EMBED_R_N_SEEDS,
                    embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE,
                )
                preds_ours = T.fit_ensemble_predictions(
                    our_tr_lg, our_val_lg, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS,
                    DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS, FEATURES, embed_config=embed_config,
                )
            else:
                pitcher_vocab = build_id_vocab(our_tr_lg, "pitcher_id")
                batter_vocab = build_id_vocab(our_tr_lg, "batter_id")
                embed_config = dict(
                    pitcher_vocab=pitcher_vocab, batter_vocab=batter_vocab, n_seeds=EMBED_F_N_SEEDS,
                    embed_dim=EMBED_F_DIM, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT, batch_size=EMBED_F_BATCH_SIZE,
                )
                preds_ours = T.fit_ensemble_predictions(
                    our_tr_lg, our_val_lg, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
                    DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, FEATURES_F, embed_config=embed_config,
                )
            p_ours = preds_ours.mean(axis=1)

            cache[f"{lg}_{split_i}_y"] = y_val
            cache[f"{lg}_{split_i}_p_tm"] = p_tm
            cache[f"{lg}_{split_i}_p_ours"] = p_ours
            cache[f"{lg}_{split_i}_month"] = game_month

            _, bss_tm = brier_skill_score(y_val, p_tm)
            _, bss_ours = brier_skill_score(y_val, p_ours)
            print(f"[split {split_i}: train<={cutoff} val={val_season}] {lg}: teammate={bss_tm:,.0f}  ours(current prod)={bss_ours:,.0f}")

        print(f"  (split took {time.time()-t_split:.0f}s)")

    np.savez(CACHE_PATH, **cache)
    print(f"cached to {CACHE_PATH}")
    return cache


def main():
    cache = generate_and_cache()

    print("\n=== R: 3구간 각각 볼록결합 가중치 ===")
    r_weights = []
    for split_i in range(3):
        key = f"R_{split_i}_y"
        if key not in cache:
            continue
        y = cache[f"R_{split_i}_y"]
        p_tm = cache[f"R_{split_i}_p_tm"]
        p_ours = cache[f"R_{split_i}_p_ours"]
        w = convex_weight(p_tm, p_ours, y)
        r_weights.append(w)
        _, bss_tm = brier_skill_score(y, p_tm)
        _, bss_ours = brier_skill_score(y, p_ours)
        _, bss_opt = brier_skill_score(y, w[0] * p_tm + w[1] * p_ours)
        print(f"  split {split_i}: w_teammate={w[0]:.3f}  w_ours={w[1]:.3f}  "
              f"bss(teammate)={bss_tm:,.0f}  bss(ours)={bss_ours:,.0f}  bss(optimal blend)={bss_opt:,.0f}  "
              f"delta vs ours={bss_opt-bss_ours:+.0f}")

    print("\n=== F: 유일 유효 구간(split 2, train<=2023->val=2024) 볼록결합 가중치 ===")
    y = cache["F_2_y"]
    p_tm = cache["F_2_p_tm"]
    p_ours = cache["F_2_p_ours"]
    month = cache["F_2_month"]
    w = convex_weight(p_tm, p_ours, y)
    _, bss_tm = brier_skill_score(y, p_tm)
    _, bss_ours = brier_skill_score(y, p_ours)
    p_opt = w[0] * p_tm + w[1] * p_ours
    _, bss_opt = brier_skill_score(y, p_opt)
    print(f"  w_teammate={w[0]:.3f}  w_ours={w[1]:.3f}  bss(teammate)={bss_tm:,.0f}  "
          f"bss(ours)={bss_ours:,.0f}  bss(optimal blend, whole-split)={bss_opt:,.0f}  delta vs ours={bss_opt-bss_ours:+.0f}")

    print("  F segment breakdown (참고용, 판정은 whole-split 기준):")
    for months in F_SEGMENTS:
        mask = np.isin(month, months)
        if mask.sum() == 0:
            continue
        _, b_ours = brier_skill_score(y[mask], p_ours[mask])
        _, b_opt = brier_skill_score(y[mask], p_opt[mask])
        print(f"    {months}: ours={b_ours:,.0f}  optimal-blend={b_opt:,.0f}  delta={b_opt-b_ours:+.0f}")


if __name__ == "__main__":
    main()
