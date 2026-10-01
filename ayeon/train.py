"""제출용 최종 모델 학습 - R(1군)/F(퓨처스) 완전 분리 모델, 각각
LightGBM + XGBoost + CatBoost + DCNv2 균등 가중(1/4) 앙상블.

1) 시즌 단위 walk-forward holdout(train<=2021/val=2022, <=2022/2023, <=2023/2024)으로
   개별 모델 및 앙상블의 실제 일반화 성능(2025 외삽에 가까운 추정치)을 확인하고
2) 전체 train.csv로 R/F 각각 트리 3종 + DCNv2(R은 3시드, F는 5시드 독립
   재학습 후 평균)를 학습해 각자의 네이티브 포맷으로 저장한다.

R/F를 분리하는 이유(common.py 참고): 공유 트리 하나로 R+F를 같이 예측하면
F(2022->2023 레벨 반전)를 살리려는 시도가 전부 R에 collateral damage를 입혔다
(가중치/피처/shrinkage 세 방식 다 실패). R/F를 아예 별도 모델로 분리하고
game_type으로 라우팅하면 이 간섭 자체가 사라져 walk-forward train<=2023->val=2024
기준 bss_all 510 -> 701로 개선된다. F는 구레짐(2023 이전) 행을 섞으면 분리해도
여전히 무너지므로 신체제만 쓴다(split_regime 참고).

DCNv2를 4번째 멤버로 추가하는 이유: 트리와 오차 패턴이 달라 블렌딩하면 추가로
개선된다(701 -> 715, R/F 둘 다 개선, common.py 근거 참고).

Platt scaling은 시도했다가 완전히 뺐다 - 연도별로 개별 적합하면 계수 부호
자체가 뒤집히고(R: 2023 intercept=+0.017, 2024 intercept=-0.028), 한 해에
적합해 다른 해에 적용하면 오히려 악화됐다(2023->2024 적용 시 bss_R
703->676). "같은 해 안에서" 통과한 리키지 없는 검증은 그 해 고유의 잔차를
잡은 것이었지 안정적인 모델 편향이 아니었다 - 실제 배포(2024에 적합 ->
2025에 적용)와 정확히 같은 구조로 실패를 재현했으므로 신뢰할 수 없다고
판단했다. LB 772.38 -> 763 회귀의 유력한 원인 중 하나로 의심된다
(common.py 근거 참고, walk_forward_check에는 진단용 출력만 남겨둠).

shrunk_pitcher_rate(common.py의 add_shrunk_pitcher_feature)는 각 행 자신의
(season-1) 시즌 (game_type) 평균을 leak-safe prior로 써서 asof_pitcher_n이
작을수록 그쪽으로 shrink한 값이다. season_means는 반드시 그 시점의 train_df
에서만 계산해야 하고(walk_forward_check는 매 구간마다 다시 계산), 최종 학습은
전체 데이터 기준으로 계산한 테이블을 model/shrink_season_means.csv로 저장해
script.py가 동일하게 재사용한다. R은 FEATURES(shrinkage 포함)를 쓰지만 F는
FEATURES_F(shrinkage 제외)를 쓴다 - F 검증셋을 여러 구간으로 쪼개보니
shrinkage 효과가 구간마다 뒤집혀서(-4/+5/+41/-19) 철회했다(common.py 근거
참고). F 하이퍼파라미터 재탐색도 같은 이유로 철회해 원래 값으로 되돌렸다.
DCN_FEATURES는 원래도 shrinkage를 뺀 상태였으므로(다중공선성 버그 수정,
common.py 근거) F는 이제 트리 3종+DCN 네 멤버 전부 같은 피처셋을 쓴다.

무작위 K-fold/랜덤 holdout을 쓰지 않는 이유: 이 데이터는 시즌이 지날수록
control_success 비율이 계속 변하는 시계열이라(data_analysis_report.md 1절),
무작위로 나누면 검증셋에 학습셋과 같은 시즌이 섞여 실제 성능(완전히 못 본
미래 시즌 2025)을 크게 과대평가한다 - model.py의 경고 참고.

트리 세 모델을 pickle/joblib이 아니라 각자의 네이티브 포맷으로 저장하는 이유:
이전에 sklearn 파이프라인을 joblib으로 저장했다가 학습 환경과 평가 서버의
scikit-learn 버전이 달라 언피클링이 깨진 적이 있어서(SimpleImputer._fill_dtype
관련 AttributeError), 버전에 의존하지 않는 포맷을 쓴다. DCNv2는 state_dict만
저장한다(torch.save 자체가 버전 의존적인 pickle이지만, 평가 서버에 torch가
고정 버전으로 이미 설치돼 있어 nn.Module 클래스 정의만 script.py에 동일하게
두면 안전하다).
"""

import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import (
    ASOF_FEATURES,
    CAT_BASE_PARAMS,
    CAT_F_FINAL_N_ESTIMATORS,
    CAT_F_PARAMS,
    CAT_FINAL_N_ESTIMATORS,
    DCN_F_BATCH_SIZE,
    DCN_F_FINAL_EPOCHS,
    DCN_F_N_SEEDS,
    DCN_F_PARAMS,
    DCN_LR,
    DCN_R_BATCH_SIZE,
    DCN_R_FINAL_EPOCHS,
    DCN_R_N_SEEDS,
    DCN_R_PARAMS,
    DCN_WEIGHT_DECAY,
    DCNv2,
    EMBED_F_BATCH_SIZE,
    EMBED_F_DIM,
    EMBED_F_DROPOUT,
    EMBED_F_FINAL_EPOCHS,
    EMBED_F_HIDDEN,
    EMBED_F_N_SEEDS,
    EMBED_R_BATCH_SIZE,
    EMBED_R_DIM,
    EMBED_R_DROPOUT,
    EMBED_R_FINAL_EPOCHS,
    EMBED_R_HIDDEN,
    EMBED_R_N_SEEDS,
    EmbedMLP,
    FEATURES,
    FEATURES_F,
    LGB_BASE_PARAMS,
    LGB_F_FINAL_N_ESTIMATORS,
    LGB_F_PARAMS,
    LGB_FINAL_N_ESTIMATORS,
    SITUATIONAL_FEATURES,
    TARGET,
    XGB_BASE_PARAMS,
    XGB_F_FINAL_N_ESTIMATORS,
    XGB_F_PARAMS,
    XGB_FINAL_N_ESTIMATORS,
    add_matchup_features,
    add_shrunk_pitcher_feature,
    apply_dcn_preprocessing,
    brier_skill_score,
    build_id_vocab,
    compute_season_means,
    encode_ids,
    fit_dcn_preprocessing,
    fit_platt,
    split_regime,
)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "model"

WALK_FORWARD_SPLITS = [(2021, 2022), (2022, 2023), (2023, 2024)]

# F는 신체제 데이터가 1년(2024)뿐이라 walk-forward 자체가 안 되므로, 마지막
# 구간(train<=2023->val=2024)에서 val_F를 월 단위 4구간으로 쪼개 방향 일관성을
# 확인한다(HANDOFF.md 5-2절 규칙). game_month는 모델 피처가 아니라 이 검증
# 전용으로만 로드한다.
F_SEGMENTS = [(3, 4), (5, 6), (7, 8), (9, 10)]

# OOF 스태킹: 균등(1/4) 평균 대신 로지스틱회귀 메타러너로 4개 베이스 모델의
# 블렌딩 가중치를 데이터가 스스로 정하게 한다. 메타러너는 train_df 내부
# K-fold OOF 예측으로 학습하고(fit_stack_meta), 실제 val_df 예측에는 val_df를
# 전혀 보지 않은 상태로 학습된 이 메타러너를 프로덕션 베이스 모델(전체
# train_df로 학습, DCN은 다중시드 평균)의 예측에 적용한다 - "조합 방식만
# 바꾸는" 변경이라 레벨 드리프트 문제와 무관하다(2026-08-21 논의 근거).
STACK_N_FOLDS = 5
STACK_SEED = 0

# 1차 검증(2026-08-21, 무정규화 C=1.0)에서 R 3구간 전부 스태킹이 나빠졌고
# (-27/-45/-39) 메타러너 계수 부호가 구간마다 뒤집히는 다중공선성 증상이
# 뚜렷했다 - L2 정규화를 강하게 걸어 재검증한다. sklearn 기본 C=1.0은 대량
# 표본(수십만~백만)+[0,1] 스케일 확률 피처에서는 로그우도 항이 정규화 항을
# 압도해 사실상 무정규화와 같았다. OOF/프로덕션 예측(재학습 비용이 큰 부분)은
# 스플릿당 한 번만 계산해서 캐싱하므로, 여러 C 후보를 한 번에 스윕해도 메타
# 러너 재적합(가벼움)만 반복될 뿐 추가 재학습 비용은 거의 없다.
STACK_C_CANDIDATES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003]

# generate_oof_preds/fit_ensemble_predictions 결과(재학습 비용이 큰 부분)를
# 캐싱해서, 메타러너의 C 등 다운스트림 하이퍼파라미터만 바꿔 재검증할 때
# 전체 재학습 없이 캐시에서 즉시 재현할 수 있게 한다.
CACHE_DIR = Path(__file__).parent / "_stack_cache"


def load_data():
    cols = (
        ASOF_FEATURES
        + SITUATIONAL_FEATURES
        + ["season", "game_type", "game_month", "pitcher_hand", "batter_hand",
           "pitcher_id", "batter_id", TARGET]
    )
    df = pd.read_csv(DATA_DIR / "train.csv", usecols=cols)
    return add_matchup_features(df)


def fit_lgb(train_df, val_df, params, feat_list=FEATURES):
    # 이 환경(lightgbm==4.7.0 + pandas==2.0.3, Windows)에서는 pandas DataFrame/Series를
    # 그대로 .fit()에 넘기면 LGBM_DatasetSetField에서 access violation이 난다(narwhals
    # 버전을 여러 개 시도해도 재현됨 - 원인 미상의 wheel 레벨 버그로 추정). numpy 배열로
    # 변환해서 넘기면 문제없이 동작하고, feature_name을 명시해 저장된 모델의
    # feature_name()이 여전히 FEATURES와 이름까지 일치하도록 유지한다.
    model = lgb.LGBMClassifier(**params, n_estimators=2000, verbose=-1)
    model.fit(
        train_df[feat_list].to_numpy(),
        train_df[TARGET].to_numpy(),
        eval_X=val_df[feat_list].to_numpy(),
        eval_y=val_df[TARGET].to_numpy(),
        eval_metric="binary_logloss",
        feature_name=list(feat_list),
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model.predict_proba(val_df[feat_list].to_numpy(), num_iteration=model.best_iteration_)[:, 1]


def fit_xgb(train_df, val_df, params, feat_list=FEATURES):
    model = xgb.XGBClassifier(
        **params, n_estimators=2000, early_stopping_rounds=50, verbosity=0
    )
    model.fit(
        train_df[feat_list],
        train_df[TARGET],
        eval_set=[(val_df[feat_list], val_df[TARGET])],
        verbose=False,
    )
    return model.predict_proba(val_df[feat_list])[:, 1]


def fit_cat(train_df, val_df, params, feat_list=FEATURES):
    model = CatBoostClassifier(**params, iterations=2000, early_stopping_rounds=50)
    model.fit(
        train_df[feat_list], train_df[TARGET], eval_set=(val_df[feat_list], val_df[TARGET])
    )
    return model.predict_proba(val_df[feat_list])[:, 1]


def fit_dcn(train_df, val_df, params, batch_size, seed=0, max_epochs=100, patience=8):
    torch.manual_seed(seed)
    stats = fit_dcn_preprocessing(train_df)
    Xtr = torch.tensor(apply_dcn_preprocessing(train_df, stats))
    ytr = torch.tensor(train_df[TARGET].to_numpy(dtype=np.float32))
    Xval = torch.tensor(apply_dcn_preprocessing(val_df, stats))
    yval = torch.tensor(val_df[TARGET].to_numpy(dtype=np.float32))

    model = DCNv2(Xtr.shape[1], **params)
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n = Xtr.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval), yval).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xval)).numpy()


def fit_dcn_multiseed(train_df, val_df, params, batch_size, n_seeds):
    """시드마다 편차가 커서(common.py 근거) 독립적으로 n_seeds번 재학습해
    예측을 평균 낸다. F는 1시드 bss_F=33 -> 5시드 평균 366으로 안정화됨."""
    preds = [fit_dcn(train_df, val_df, params, batch_size, seed=s) for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def fit_embed(train_df, val_df, pitcher_vocab, batter_vocab, embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN,
              dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE, seed=0, max_epochs=100, patience=8):
    """pitcher/batter 저차원 임베딩 + MLP (common.py EmbedMLP, 2026-08-23 R 채택,
    2026-08-24 F 채택). DCN과 동일한 전처리(DCN_FEATURES, fit_dcn_preprocessing/
    apply_dcn_preprocessing)를 재사용한다 - stats도 DCN과 공유 가능(같은 피처셋).
    embed_dim/hidden/dropout/batch_size를 명시적 인자로 받는다 - R/F가 서로 다른
    설정을 쓰므로(EMBED_R_*/EMBED_F_*, common.py 근거) 기본값에 암묵적으로 의존하지
    않는다."""
    torch.manual_seed(seed)
    stats = fit_dcn_preprocessing(train_df)
    Xtr_raw = torch.tensor(apply_dcn_preprocessing(train_df, stats))
    Xval_raw = torch.tensor(apply_dcn_preprocessing(val_df, stats))
    pid_tr = torch.tensor(encode_ids(train_df, "pitcher_id", pitcher_vocab))
    bid_tr = torch.tensor(encode_ids(train_df, "batter_id", batter_vocab))
    pid_val = torch.tensor(encode_ids(val_df, "pitcher_id", pitcher_vocab))
    bid_val = torch.tensor(encode_ids(val_df, "batter_id", batter_vocab))
    ytr = torch.tensor(train_df[TARGET].to_numpy(dtype=np.float32))
    yval = torch.tensor(val_df[TARGET].to_numpy(dtype=np.float32))

    model = EmbedMLP(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab), embed_dim=embed_dim, hidden=hidden, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n = Xtr_raw.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_raw[idx], pid_tr[idx], bid_tr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_raw, pid_val, bid_val), yval).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, bad_epochs = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xval_raw, pid_val, bid_val)).numpy()


def fit_embed_multiseed(train_df, val_df, pitcher_vocab, batter_vocab, n_seeds=EMBED_R_N_SEEDS,
                         embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE):
    """R/F DCN과 동일 관례 - 시드 편차가 커서 다중 시드 평균으로 안정화."""
    preds = [fit_embed(train_df, val_df, pitcher_vocab, batter_vocab, embed_dim, hidden, dropout, batch_size, seed=s) for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def fit_ensemble_predictions(train_df, val_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, dcn_n_seeds, feat_list=FEATURES, embed_config=None):
    """베이스 모델들을 각각 train_df로 학습해 val_df에 대한 예측을 (n_val, 4 또는 5)
    배열로 반환한다(합치지 않음) - 균등 평균과 스태킹 블렌드가 같은 학습 결과를
    공유해서 쓰도록 분리했다(중복 학습 방지). embed_config=dict(pitcher_vocab=...,
    batter_vocab=..., n_seeds=..., embed_dim=..., hidden=..., dropout=...,
    batch_size=...)가 주어지면 5번째 멤버(pitcher/batter 임베딩, 2026-08-23 R
    채택/2026-08-24 F 채택)를 추가한다 - dict 키로 R/F 각자의 설정을 명시적으로
    전달해야 하므로(EMBED_R_*/EMBED_F_*가 암묵적으로 섞일 위험이 없다)."""
    p_lgb = fit_lgb(train_df, val_df, lgb_params, feat_list)
    p_xgb = fit_xgb(train_df, val_df, xgb_params, feat_list)
    p_cat = fit_cat(train_df, val_df, cat_params, feat_list)
    p_dcn = fit_dcn_multiseed(train_df, val_df, dcn_params, dcn_batch_size, dcn_n_seeds)
    cols = [p_lgb, p_xgb, p_cat, p_dcn]
    if embed_config is not None:
        cols.append(fit_embed_multiseed(
            train_df, val_df, embed_config["pitcher_vocab"], embed_config["batter_vocab"],
            embed_config["n_seeds"], embed_config["embed_dim"], embed_config["hidden"],
            embed_config["dropout"], embed_config["batch_size"],
        ))
    return np.column_stack(cols)


def fit_ensemble(train_df, val_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, dcn_n_seeds, feat_list=FEATURES, embed_config=None):
    preds = fit_ensemble_predictions(train_df, val_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, dcn_n_seeds, feat_list, embed_config)
    return preds.mean(axis=1)


"""OOF 스태킹(2026-08-21~22) - 기각됨, 진단 도구로만 남겨둠 (Platt scaling과 같은
취급, fit_final/script.py에는 연결한 적 없음).

균등(1/4) 평균 대신 데이터가 스스로 4개 베이스 모델의 블렌딩 가중치를 정하게
하려는 시도. 세 가지 메타러너를 R 3구간 walk-forward + F 4구간(월별)으로
검증했다.
1) 무정규화 로지스틱 회귀(LogisticRegression()): R 3구간 전부 avg보다 나빠짐
   (-27/-45/-39), F도 마이너스(-24, 4구간 중 3구간 마이너스). 계수 부호가
   구간마다 뒤집힘(1구간 cat 음수, 2·3구간 dcn 음수) - 4개 베이스 예측이
   서로 강하게 상관돼(다중공선성) 무정규화 회귀가 불안정한 것으로 진단.
2) L2 정규화 로지스틱(C 스윕 1.0~0.001): 격차는 줄었지만(R 최선 -9~-18) 3구간
   전부 여전히 avg보다 나빴다. 결정적 결함 발견: **같은 C를 R(~109만행)과
   F(~2.6만행)에 동일하게 적용하면 표본이 40배 작은 F에 정규화가 훨씬 강하게
   걸려**, C가 작아질수록 F가 오히려 절편만 남고 붕괴했다(C=1.0 bss_F=507 ->
   C=0.003 bss_F=51, 4구간 전부 거의 0으로 무너짐). R/F가 이미 완전히 분리된
   모델이라는 걸 감안하면 애초에 공유 C를 쓴 것 자체가 설계 결함이었다.
3) 볼록결합(가중치>=0, 합=1, Brier 직접 최소화, fit_convex_stack_from_oof):
   **모든 구간, R/F 둘 다에서 최적해가 정확히 균등(0.25,0.25,0.25,0.25)으로
   수렴했다** - 즉 지금 프로덕션의 1/4 균등 평균이 이미 이 제약 공간 안에서
   Brier 최적점이다. 4개 모델의 오차 패턴이 균등 가중치로 이미 최적으로
   상쇄되고 있다는 뜻으로, "아이디어가 틀림"이 아니라 "개선 여지 자체가
   없음"이 명확한 결론이다.

결론: 3가지 원칙적 변형 전부 기각. 균등 평균을 그대로 유지한다. 재시도 전 이
rationale부터 확인할 것 - 특히 R/F에 공유 정규화 강도를 쓰는 실수는 반복하지
말 것.
"""


def generate_oof_preds(train_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, feat_list):
    """train_df 내부에서 K-fold(기본 5) OOF 예측을 만든다 - 메타러너 학습 전용이고
    성능 추정치로 쓰지 않는다(성능 판정은 항상 walk-forward val_df 기준). DCN은
    폴드마다 고정 시드 하나만 쓴다(프로덕션의 다중시드 평균이 아님) - 메타러너는
    모델 간 상대적 가중치 신호만 필요하고, 폴드 수(5) x 시드 수만큼 재학습 비용이
    커지는 걸 피하기 위한 절충이다."""
    train_df = train_df.reset_index(drop=True)
    y = train_df[TARGET].to_numpy()
    oof = np.zeros((len(train_df), 4), dtype=np.float64)
    skf = StratifiedKFold(n_splits=STACK_N_FOLDS, shuffle=True, random_state=STACK_SEED)
    for fold_tr_idx, fold_val_idx in skf.split(train_df, y):
        fold_tr, fold_val = train_df.iloc[fold_tr_idx], train_df.iloc[fold_val_idx]
        oof[fold_val_idx, 0] = fit_lgb(fold_tr, fold_val, lgb_params, feat_list)
        oof[fold_val_idx, 1] = fit_xgb(fold_tr, fold_val, xgb_params, feat_list)
        oof[fold_val_idx, 2] = fit_cat(fold_tr, fold_val, cat_params, feat_list)
        oof[fold_val_idx, 3] = fit_dcn(fold_tr, fold_val, dcn_params, dcn_batch_size, seed=STACK_SEED)
    return oof, y


def fit_stack_meta_from_oof(oof, y_oof, C=1.0):
    """이미 계산된 OOF 예측(generate_oof_preds)에 로지스틱회귀를 적합한다 - 이
    단계 자체는 가볍다(4개 컬럼짜리 회귀). 4개 베이스 모델 예측이 서로 강하게
    상관돼 있어(다중공선성) 정규화가 약하면(C 큼) 계수가 구간마다 부호까지
    뒤집히며 불안정해지는 걸 1차 검증에서 확인했다(2026-08-21) - C를 작게 줘서
    L2 정규화를 강하게 건다(sklearn 기본 penalty='l2', C=1.0은 이 스케일의
    피처+대량 표본에서는 사실상 무정규화나 마찬가지였다)."""
    meta = LogisticRegression(C=C)
    meta.fit(oof, y_oof)
    return meta


def fit_stack_meta(train_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, feat_list, C=1.0):
    """train_df의 OOF 예측(generate_oof_preds)에 로지스틱회귀를 적합해 4개 베이스
    모델의 블렌딩 가중치를 학습한다. val_df를 전혀 보지 않으므로 이 메타러너를
    val_df 예측에 적용해도 val_df 정보가 학습에 새지 않는다."""
    oof, y_oof = generate_oof_preds(train_df, lgb_params, xgb_params, cat_params, dcn_params, dcn_batch_size, feat_list)
    return fit_stack_meta_from_oof(oof, y_oof, C)


def fit_convex_stack_from_oof(oof, y_oof):
    """4개 베이스 모델 OOF 예측의 볼록결합(가중치 >=0, 합=1)을 Brier Score 직접
    최소화로 구한다 - 로지스틱 메타러너(L2 정규화를 걸어도 계수 부호 자체는
    막지 못함)와 달리 확률 예측에 음의 가중치를 주는 걸 원천적으로 배제하고,
    초기값이 균등 가중치(현재 프로덕션과 동일)라 최적화가 잘 안 풀려도 최소한
    균등 평균 근방에서 시작한다. Brier은 가중치 w에 대해 이차형식(볼록)이라
    SLSQP가 전역해를 안정적으로 찾는다."""
    n_models = oof.shape[1]

    def objective(w):
        p = oof @ w
        return np.mean((p - y_oof) ** 2)

    result = minimize(
        objective, x0=np.full(n_models, 1.0 / n_models),
        bounds=[(0, 1)] * n_models, method="SLSQP",
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    return result.x


def apply_convex_stack(preds, weights):
    return preds @ weights


def segment_bss(val_df, p, months):
    """val_df를 game_month in months로 필터링한 부분집합의 BSS. 표본이 없으면 None."""
    mask = val_df["game_month"].isin(months).to_numpy()
    if mask.sum() == 0:
        return None, 0
    _, bss = brier_skill_score(val_df[TARGET].to_numpy()[mask], np.asarray(p)[mask])
    return bss, int(mask.sum())


def walk_forward_check(df, run_stack_diag=False):
    """run_stack_diag=True를 주면 OOF 스태킹 진단(2026-08-21/22, 기각됨 -
    generate_oof_preds 위 rationale 참고)까지 다시 돌린다 - K-fold OOF 재생성이
    포함돼 몇 배 더 오래 걸린다. 이미 결론 난 진단이라 기본은 False(균등 평균
    앙상블 + Platt 진단만) - 향후 정말 재검증이 필요할 때만 True로 켤 것."""
    bss_all_avg_list = []
    bss_all_stack_lists = {C: [] for C in STACK_C_CANDIDATES}
    for cutoff, val_season in WALK_FORWARD_SPLITS:
        train_df = df[df["season"] <= cutoff]
        val_df = df[df["season"] == val_season]

        season_means = compute_season_means(train_df)  # train만으로 계산 - 누출 방지
        train_df = add_shrunk_pitcher_feature(train_df, season_means)
        val_df = add_shrunk_pitcher_feature(val_df, season_means)

        tr_R, tr_F = split_regime(train_df)
        val_R = val_df[val_df["game_type"] == "R"]
        val_F = val_df[val_df["game_type"] == "F"]

        # pitcher/batter 임베딩(R 2026-08-23 채택, F 2026-08-24 채택) - 어휘는
        # 각 리그 자신의 train에서만 만든다(누출 방지), val의 미확인 ID는
        # encode_ids가 자동으로 unknown(0) 처리한다. R/F 설정은 EMBED_R_*/EMBED_F_*로
        # 완전히 독립적이다.
        pitcher_vocab_R = build_id_vocab(tr_R, "pitcher_id")
        batter_vocab_R = build_id_vocab(tr_R, "batter_id")
        embed_config_R = dict(
            pitcher_vocab=pitcher_vocab_R, batter_vocab=batter_vocab_R, n_seeds=EMBED_R_N_SEEDS,
            embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE,
        )
        pitcher_vocab_F = build_id_vocab(tr_F, "pitcher_id")
        batter_vocab_F = build_id_vocab(tr_F, "batter_id")
        embed_config_F = dict(
            pitcher_vocab=pitcher_vocab_F, batter_vocab=batter_vocab_F, n_seeds=EMBED_F_N_SEEDS,
            embed_dim=EMBED_F_DIM, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT, batch_size=EMBED_F_BATCH_SIZE,
        )

        preds_R = fit_ensemble_predictions(tr_R, val_R, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_N_SEEDS, FEATURES, embed_config_R)
        preds_F = fit_ensemble_predictions(tr_F, val_F, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS, FEATURES_F, embed_config_F)
        p_R_avg, p_F_avg = preds_R.mean(axis=1), preds_F.mean(axis=1)

        y_R, y_F = val_R[TARGET].to_numpy(), val_F[TARGET].to_numpy()
        y_all = np.concatenate([y_R, y_F])

        if run_stack_diag:
            # 메타러너는 val_df를 전혀 보지 않고 tr_R/tr_F 내부 OOF로만 학습한다.
            # OOF 생성(재학습 비용이 큰 부분)은 여기서 한 번만 하고 캐싱한다 -
            # 메타러너의 C 등 다운스트림 하이퍼파라미터를 바꿔가며 재검증할 때
            # 이 캐시에서 즉시 재현할 수 있다(전체 재학습 불필요).
            oof_R, y_oof_R = generate_oof_preds(tr_R, LGB_BASE_PARAMS, XGB_BASE_PARAMS, CAT_BASE_PARAMS, DCN_R_PARAMS, DCN_R_BATCH_SIZE, FEATURES)
            oof_F, y_oof_F = generate_oof_preds(tr_F, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS, DCN_F_PARAMS, DCN_F_BATCH_SIZE, FEATURES_F)

            CACHE_DIR.mkdir(exist_ok=True)
            np.savez(
                str(CACHE_DIR / f"split_{cutoff}_{val_season}.npz"),
                oof_R=oof_R, y_oof_R=y_oof_R, oof_F=oof_F, y_oof_F=y_oof_F,
                preds_R=preds_R, y_R=y_R, preds_F=preds_F, y_F=y_F,
                game_month_F=val_F["game_month"].to_numpy(),
            )

        _, bss_R_avg = brier_skill_score(y_R, p_R_avg)
        _, bss_F_avg = brier_skill_score(y_F, p_F_avg) if len(y_F) else (None, float("nan"))
        brier_all_avg, bss_all_avg = brier_skill_score(y_all, np.concatenate([p_R_avg, p_F_avg]))
        bss_all_avg_list.append(bss_all_avg)
        print(
            f"[walk-forward:avg  ] train<={cutoff} val={val_season}  "
            f"n_train_R={len(tr_R):,} n_train_F={len(tr_F):,}  "
            f"bss_R={bss_R_avg:,.0f}  bss_F={bss_F_avg:,.0f}  "
            f"ENSEMBLE Brier={brier_all_avg:.5f} BSS={bss_all_avg:,.0f}"
        )

        if run_stack_diag:
            # C를 스윕한다 - 각 후보마다 메타러너 재적합만 하면 되므로(OOF/프로덕션
            # 예측은 위에서 이미 한 번 계산) 후보 수만큼 늘어도 비용이 크지 않다.
            stack_results = {}
            for C in STACK_C_CANDIDATES:
                meta_R = fit_stack_meta_from_oof(oof_R, y_oof_R, C)
                meta_F = fit_stack_meta_from_oof(oof_F, y_oof_F, C)
                p_R_stack = meta_R.predict_proba(preds_R)[:, 1]
                p_F_stack = meta_F.predict_proba(preds_F)[:, 1]

                _, bss_R = brier_skill_score(y_R, p_R_stack)
                _, bss_F = brier_skill_score(y_F, p_F_stack) if len(y_F) else (None, float("nan"))
                brier_all, bss_all = brier_skill_score(y_all, np.concatenate([p_R_stack, p_F_stack]))
                bss_all_stack_lists[C].append(bss_all)
                stack_results[C] = (p_R_stack, p_F_stack, meta_R, meta_F)
                print(
                    f"[walk-forward:stack] train<={cutoff} val={val_season}  C={C:<6}  "
                    f"bss_R={bss_R:,.0f}  bss_F={bss_F:,.0f}  "
                    f"ENSEMBLE Brier={brier_all:.5f} BSS={bss_all:,.0f}"
                )
                print(
                    f"  [meta coef C={C}] R: lgb={meta_R.coef_[0][0]:+.3f} xgb={meta_R.coef_[0][1]:+.3f} "
                    f"cat={meta_R.coef_[0][2]:+.3f} dcn={meta_R.coef_[0][3]:+.3f} intercept={meta_R.intercept_[0]:+.3f}  |  "
                    f"F: lgb={meta_F.coef_[0][0]:+.3f} xgb={meta_F.coef_[0][1]:+.3f} "
                    f"cat={meta_F.coef_[0][2]:+.3f} dcn={meta_F.coef_[0][3]:+.3f} intercept={meta_F.intercept_[0]:+.3f}"
                )

            # F는 신체제 데이터가 1년(2024)뿐이라 walk-forward 3구간 중 이 마지막
            # 구간에서만 F 검증이 가능하다 - 전체 평균 하나만 보면 구간별 상쇄를
            # 놓칠 수 있어(HANDOFF.md 5-2절) 월 단위 4구간으로, 가장 강하게 정규화된
            # 후보(C 최솟값)에 한해 avg/stack을 확인한다(전체 후보를 다 찍으면 너무
            # 길어져서 - 나머지 후보는 필요하면 _stack_cache의 캐시로 사후 분석).
            if cutoff == WALK_FORWARD_SPLITS[-1][0] and len(val_F) > 0:
                probe_C = min(STACK_C_CANDIDATES)
                p_F_stack_probe = stack_results[probe_C][1]
                print(f"[F multi-segment check: avg vs stack(C={probe_C}), by game_month]")
                for months in F_SEGMENTS:
                    bss_avg, n_seg = segment_bss(val_F, p_F_avg, months)
                    if bss_avg is None:
                        continue
                    bss_stk, _ = segment_bss(val_F, p_F_stack_probe, months)
                    print(f"  game_month in {months}: n={n_seg:,}  avg={bss_avg:,.0f}  stack={bss_stk:,.0f}  delta={bss_stk-bss_avg:+.0f}")

        # F는 신체제 데이터가 1년(2024)뿐이라 이 마지막 구간에서만 검증 가능하다
        # (HANDOFF.md 5-2절) - 프로덕션 균등 평균 앙상블 자체도 월 단위 4구간으로
        # 방향 일관성을 항상 확인한다(F DCN 아키텍처 변경 등을 검증할 때 유용).
        if cutoff == WALK_FORWARD_SPLITS[-1][0] and len(val_F) > 0:
            print("[F multi-segment check: avg ensemble, by game_month]")
            for months in F_SEGMENTS:
                bss_avg, n_seg = segment_bss(val_F, p_F_avg, months)
                if bss_avg is None:
                    continue
                print(f"  game_month in {months}: n={n_seg:,}  avg={bss_avg:,.0f}")

        # Platt scaling은 시도했다가 완전히 제거했다 - 연도별로 개별 적합한
        # 계수의 부호 자체가 뒤집혔고(R: 2023 intercept=+0.0171, 2024
        # intercept=-0.0281), 2023에 적합한 Platt을 2024에 적용하면
        # bss_R이 703->676으로 오히려 악화됐다(그 반대인 2022->2023은 소폭
        # 개선 +4에 그침). "같은 해 안에서" 월별 분할 검증은 통과했지만
        # (R +24/F +14), 그건 "이 모델이 이 해에 얼마나 치우쳤는지"를 보정한
        # 것이지 안정적인 모델 편향이 아니었다 - R은 매년 드리프트량 자체가
        # 달라서 그 해 고유의 잔차를 다음 해로 그대로 못 옮긴다. 실제 배포는
        # "2024에 적합 -> 2025에 적용"과 정확히 같은 구조라 신뢰할 수 없다고
        # 판단해 common.py의 fit_platt/apply_platt는 남겨두되(재사용 가능한
        # 진단 도구로) 파이프라인에서는 빼기로 했다. LB 772.38 -> 763 회귀의
        # 유력한 원인 중 하나로 의심된다.
        if cutoff == WALK_FORWARD_SPLITS[-1][0]:
            coef_R, intercept_R = fit_platt(p_R_avg, y_R)
            coef_F, intercept_F = fit_platt(p_F_avg, y_F)
            print(f"[platt 진단용, 미적용] R coef={coef_R:.4f} intercept={intercept_R:.4f}  "
                  f"F coef={coef_F:.4f} intercept={intercept_F:.4f}")

    print(f"[walk-forward] ENSEMBLE AVG   BSS = {np.mean(bss_all_avg_list):,.0f}  per-split={[round(v) for v in bss_all_avg_list]}")
    if run_stack_diag:
        for C in STACK_C_CANDIDATES:
            print(f"[walk-forward] ENSEMBLE STACK BSS (C={C:<6}) = {np.mean(bss_all_stack_lists[C]):,.0f}  per-split={[round(v) for v in bss_all_stack_lists[C]]}")


def save_via_temp(save_callable, final_path):
    """final_path에 바로 저장하는 대신 동기화 안 되는 임시 디렉터리에 먼저
    저장한 뒤 옮긴다 - 프로젝트 경로가 OneDrive 동기화 대상(Documents 하위)이라
    LightGBM의 저수준 save_model이 간헐적으로 "not available for writes"로
    실패하는 걸 확인했다(2026-08-22). shutil.move는 OS 레벨 파일 이동이라
    OneDrive의 파일 잠금과 충돌하지 않는다. save_callable(path)는 그 경로에
    파일을 쓰는 콜백(예: lambda p: model.save_model(p)).

    확장자를 반드시 원래 그대로 유지해야 한다 - XGBoost의 save_model은 확장자로
    저장 포맷을 결정한다(.json/.ubj면 JSON/UBJSON, 아니면 레거시 바이너리).
    처음엔 임시 파일명을 `원본이름.pid.tmp`로 만들었다가(확장자가 .tmp가 됨)
    xgb가 legacy 바이너리로 저장해버려서, 최종적으로 .json 확장자로 옮긴 뒤
    script.py가 JSON으로 파싱하려다 깨지는 걸 스모크 테스트로 잡았다(2026-08-22)
    - 확장자는 그대로 두고 파일명(stem)에만 pid를 끼워 넣는다."""
    final_path = Path(final_path)
    tmp_path = Path(tempfile.gettempdir()) / f"{final_path.stem}_{os.getpid()}{final_path.suffix}"
    save_callable(str(tmp_path))
    shutil.move(str(tmp_path), str(final_path))


def fit_final(
    train_df, lgb_params, lgb_n, xgb_params, xgb_n, cat_params, cat_n,
    dcn_params, dcn_batch_size, dcn_epochs, dcn_n_seeds, suffix, feat_list=FEATURES,
    embed_config=None,
):
    X, y = train_df[feat_list], train_df[TARGET]

    # fit_lgb와 동일한 이유로 lgb만 numpy 변환 필요 (이 환경의 pandas 입력 크래시,
    # 위 fit_lgb 주석 참고) - feature_name을 명시해 이름을 보존한다.
    lgb_model = lgb.LGBMClassifier(**lgb_params, n_estimators=lgb_n, verbose=-1)
    lgb_model.fit(X.to_numpy(), y.to_numpy(), feature_name=list(feat_list))
    save_via_temp(lgb_model.booster_.save_model, MODEL_DIR / f"lgb_model_{suffix}.txt")

    xgb_model = xgb.XGBClassifier(**xgb_params, n_estimators=xgb_n, verbosity=0)
    xgb_model.fit(X, y)
    save_via_temp(xgb_model.save_model, MODEL_DIR / f"xgb_model_{suffix}.json")

    cat_model = CatBoostClassifier(**cat_params, iterations=cat_n)
    cat_model.fit(X, y)
    save_via_temp(cat_model.save_model, MODEL_DIR / f"cat_model_{suffix}.cbm")

    # DCNv2: 전처리 통계는 한 번만 계산(시드와 무관), 시드별로 독립 재학습해서
    # dcn_n_seeds개의 state_dict를 각각 저장한다 - 시드 편차가 커서(common.py
    # 근거) 추론 시 예측을 평균 낸다.
    stats = fit_dcn_preprocessing(train_df)
    Xtr_np = apply_dcn_preprocessing(train_df, stats)
    ytr = torch.tensor(train_df[TARGET].to_numpy(dtype=np.float32))
    n = Xtr_np.shape[0]
    for seed in range(dcn_n_seeds):
        torch.manual_seed(seed)
        Xtr = torch.tensor(Xtr_np)
        dcn_model = DCNv2(Xtr.shape[1], **dcn_params)
        opt = torch.optim.Adam(dcn_model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(dcn_epochs):
            dcn_model.train()
            perm = torch.randperm(n)
            for i in range(0, n, dcn_batch_size):
                idx = perm[i:i + dcn_batch_size]
                opt.zero_grad()
                loss = loss_fn(dcn_model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
        save_via_temp(lambda p, m=dcn_model: torch.save(m.state_dict(), p), MODEL_DIR / f"dcn_model_{suffix}_seed{seed}.pt")

    np.savez(
        str(MODEL_DIR / f"dcn_stats_{suffix}.npz"),
        mean=stats["mean"], std=stats["std"], miss_cols=np.array(stats["miss_cols"]),
    )

    # pitcher/batter 임베딩(R 2026-08-23 채택, F 2026-08-24 채택) - DCN과 동일한
    # 전처리 통계를 재사용하므로(dcn_stats_{suffix}.npz, 위에서 이미 저장) 별도
    # stats 파일이 필요 없다. 어휘(pitcher_id/batter_id -> index, 0=unknown)는
    # script.py가 학습 데이터 없이도 동일하게 재현해야 하므로 CSV로 저장한다.
    # embed_config는 dict(pitcher_vocab=..., batter_vocab=..., n_seeds=..., epochs=...,
    # embed_dim=..., hidden=..., dropout=..., batch_size=...) - R/F 설정을 명시적으로
    # 전달해 EMBED_R_*/EMBED_F_*가 암묵적으로 섞이지 않게 한다.
    if embed_config is not None:
        pitcher_vocab = embed_config["pitcher_vocab"]
        batter_vocab = embed_config["batter_vocab"]
        embed_n_seeds = embed_config["n_seeds"]
        embed_epochs = embed_config["epochs"]
        pd.DataFrame({"pitcher_id": list(pitcher_vocab.keys()), "idx": list(pitcher_vocab.values())}).to_csv(
            MODEL_DIR / f"embed_pitcher_vocab_{suffix}.csv", index=False
        )
        pd.DataFrame({"batter_id": list(batter_vocab.keys()), "idx": list(batter_vocab.values())}).to_csv(
            MODEL_DIR / f"embed_batter_vocab_{suffix}.csv", index=False
        )
        pid_tr = torch.tensor(encode_ids(train_df, "pitcher_id", pitcher_vocab))
        bid_tr = torch.tensor(encode_ids(train_df, "batter_id", batter_vocab))
        embed_batch_size = embed_config["batch_size"]
        for seed in range(embed_n_seeds):
            torch.manual_seed(seed)
            Xtr = torch.tensor(Xtr_np)
            embed_model = EmbedMLP(
                Xtr.shape[1], len(pitcher_vocab), len(batter_vocab),
                embed_dim=embed_config["embed_dim"], hidden=embed_config["hidden"], dropout=embed_config["dropout"],
            )
            opt = torch.optim.Adam(embed_model.parameters(), lr=DCN_LR, weight_decay=DCN_WEIGHT_DECAY)
            loss_fn = nn.BCEWithLogitsLoss()
            for _ in range(embed_epochs):
                embed_model.train()
                perm = torch.randperm(n)
                for i in range(0, n, embed_batch_size):
                    idx = perm[i:i + embed_batch_size]
                    opt.zero_grad()
                    loss = loss_fn(embed_model(Xtr[idx], pid_tr[idx], bid_tr[idx]), ytr[idx])
                    loss.backward()
                    opt.step()
            save_via_temp(lambda p, m=embed_model: torch.save(m.state_dict(), p), MODEL_DIR / f"embed_model_{suffix}_seed{seed}.pt")
        print(f"saved: model/embed_{{pitcher,batter}}_vocab_{suffix}.csv, embed_model_{suffix}_seed[0-{embed_n_seeds-1}].pt")

    print(f"saved: model/{{lgb,xgb,cat}}_model_{suffix}.*, dcn_model_{suffix}_seed[0-{dcn_n_seeds-1}].pt  "
          f"(n_train={len(train_df):,})")


def main(check_only=False):
    df = load_data()
    walk_forward_check(df)
    if check_only:
        return

    season_means = compute_season_means(df)
    df = add_shrunk_pitcher_feature(df, season_means)
    r_df, f_df = split_regime(df)
    MODEL_DIR.mkdir(exist_ok=True)

    season_means.rename("mean").reset_index().to_csv(
        MODEL_DIR / "shrink_season_means.csv", index=False
    )

    # pitcher/batter 임베딩(R 2026-08-23 채택, F 2026-08-24 채택) - 어휘는 각
    # 리그의 프로덕션 학습셋 전체(2019~2024)에서 만든다. R/F 설정은 EMBED_R_*/
    # EMBED_F_*로 완전히 독립적이다.
    pitcher_vocab_R = build_id_vocab(r_df, "pitcher_id")
    batter_vocab_R = build_id_vocab(r_df, "batter_id")
    embed_config_R = dict(
        pitcher_vocab=pitcher_vocab_R, batter_vocab=batter_vocab_R,
        n_seeds=EMBED_R_N_SEEDS, epochs=EMBED_R_FINAL_EPOCHS,
        embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT, batch_size=EMBED_R_BATCH_SIZE,
    )
    pitcher_vocab_F = build_id_vocab(f_df, "pitcher_id")
    batter_vocab_F = build_id_vocab(f_df, "batter_id")
    embed_config_F = dict(
        pitcher_vocab=pitcher_vocab_F, batter_vocab=batter_vocab_F,
        n_seeds=EMBED_F_N_SEEDS, epochs=EMBED_F_FINAL_EPOCHS,
        embed_dim=EMBED_F_DIM, hidden=EMBED_F_HIDDEN, dropout=EMBED_F_DROPOUT, batch_size=EMBED_F_BATCH_SIZE,
    )

    fit_final(
        r_df, LGB_BASE_PARAMS, LGB_FINAL_N_ESTIMATORS,
        XGB_BASE_PARAMS, XGB_FINAL_N_ESTIMATORS,
        CAT_BASE_PARAMS, CAT_FINAL_N_ESTIMATORS,
        DCN_R_PARAMS, DCN_R_BATCH_SIZE, DCN_R_FINAL_EPOCHS, DCN_R_N_SEEDS, "R",
        embed_config=embed_config_R,
    )
    fit_final(
        f_df, LGB_F_PARAMS, LGB_F_FINAL_N_ESTIMATORS,
        XGB_F_PARAMS, XGB_F_FINAL_N_ESTIMATORS,
        CAT_F_PARAMS, CAT_F_FINAL_N_ESTIMATORS,
        DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_FINAL_EPOCHS, DCN_F_N_SEEDS, "F", FEATURES_F,
        embed_config=embed_config_F,
    )


if __name__ == "__main__":
    import sys
    main(check_only="--check-only" in sys.argv)
