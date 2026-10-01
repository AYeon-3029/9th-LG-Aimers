"""8단계 — Optuna로 LightGBM 정규화 파라미터(reg_alpha, reg_lambda, min_split_gain) 탐색.

지금까지 num_leaves/min_child_samples/learning_rate는 수동으로만 맞췄고
정규화 파라미터는 기본값(0)을 그대로 썼다 — 여기를 체계적으로 탐색한다.

⚠️ 단일 2024 분할 검증이 실제 리더보드와 반대 방향으로 나온 전례가 세 번
있었기 때문에(README 6.10~6.12절), 목적함수를 단일 분할이 아니라
train_v6의 walk-forward 다중 fold(2022+2024 평균, 2023은 ABS 단절로 제외)로
잡는다 — 그래야 단일 분할 노이즈에 파라미터가 과적합되는 걸 최대한 피할 수 있다.

피처 구성/범주형 처리는 현재 채택 구성(A: 팀ID만 범주형, k=50 — 유일하게
실제 리더보드에서 검증된 조합)을 그대로 고정하고, 정규화 파라미터만 바꾼다.
"""

import time

import lightgbm
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from features import (
    CATEGORICAL_COMBO_COLS,
    COMBO_FEATURE_NAMES,
    SHRINKAGE_FEATURE_NAMES,
    TREND_FEATURE_NAMES,
    add_combo_features,
    add_season_trend_feature,
    add_shrinkage_features,
    fit_season_trend,
    load_prior_table,
)

DATA_DIR = "./open/data"
ID = "row_id"
TARGET = "control_success"

VAL_SEASONS = [2022, 2024]  # 2023은 ABS 단절로 판단 기준에서 제외 (v6와 동일)
K = 50  # 현재 채택 shrinkage 강도
BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS

N_TRIALS = 30


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def run_fold(raw_train, prior_table, val_season, params):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train, y_train = train.loc[is_train, FEATURES], train.loc[is_train, TARGET]
    X_val, y_val = train.loc[is_val, FEATURES], train.loc[is_val, TARGET]

    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])
    pre.fit(X_train)
    Xtr_t = pre.transform(X_train)
    Xval_t = pre.transform(X_val)

    clf = LGBMClassifier(
        n_estimators=3000, learning_rate=0.02, num_leaves=63, min_child_samples=50,
        subsample=1.0, colsample_bytree=1.0, random_state=42, n_jobs=-1, verbose=-1,
        **params,
    )
    clf.fit(Xtr_t, y_train, eval_X=Xval_t, eval_y=y_val, eval_metric="binary_logloss",
            callbacks=[lightgbm.early_stopping(100, verbose=False)])

    pred = clf.predict_proba(Xval_t)[:, 1]
    return score(pred, y_val), clf.best_iteration_


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    # 기준선(현재 채택, reg 파라미터 전부 기본값 0) — 탐색 전에 먼저 재확인
    base_scores = {}
    for val_season in VAL_SEASONS:
        s, it = run_fold(raw_train, prior_table, val_season, dict(reg_alpha=0.0, reg_lambda=0.0, min_split_gain=0.0))
        base_scores[val_season] = s
        print(f"[기준선] val={val_season} | best_iter={it:4d} | Score={s:8.2f}")
    base_avg = sum(base_scores.values()) / len(base_scores)
    print(f"[기준선] 평균: {base_avg:.2f}\n")

    def objective(trial):
        params = dict(
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 0.5),
        )
        t = time.time()
        scores = {}
        for val_season in VAL_SEASONS:
            s, _ = run_fold(raw_train, prior_table, val_season, params)
            scores[val_season] = s
        avg = sum(scores.values()) / len(scores)
        print(f"  trial={trial.number:3d} | {params} | 2022={scores[2022]:.2f} 2024={scores[2024]:.2f} "
              f"| avg={avg:.2f} | {time.time()-t:.1f}s")
        return avg

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n=== 최적 파라미터 ===")
    print(study.best_params)
    print(f"best avg={study.best_value:.2f}  (기준선 대비 {study.best_value - base_avg:+.2f})")

    print("\n=== 상위 5개 trial ===")
    df = study.trials_dataframe().sort_values("value", ascending=False).head(5)
    print(df[["number", "value", "params_reg_alpha", "params_reg_lambda", "params_min_split_gain"]])


if __name__ == "__main__":
    main()
