"""24단계 — NODE(Neural Oblivious Decision Ensembles) 단독 성능 스크리닝.

배경: FT-Transformer(23단계)는 두 번 시도(기본 설정, warmup+gradient accumulation)
모두 CatBoost(776.56, 이 환경 재현치) 대비 확실히 낮았다(-29~-74) — 사전에 정한
"오늘 신호 안 보이면 접는다" 기준에 따라 기각.

NODE를 다음 후보로 고른 구체적 근거: 19단계(grow_policy 스윕)에서 이 데이터가
CatBoost의 대칭/oblivious tree 구조(SymmetricTree)에 유독 잘 맞는다는 게 확실히
증명됐다(Depthwise/Lossguide 둘 다 -14~-35로 명확히 나빴음, 두 fold 모두). NODE는
정확히 그 oblivious tree 구조를 미분 가능하게(soft, entmax/엔트모이드 기반) 재구현한
신경망 앙상블이다 — CatBoost가 이기는 구조적 이유는 유지하면서, 학습 방식만
부스팅(순차적 잔차 학습)에서 경사하강법(전체 앙상블 동시 학습)으로 바꾸는 셈이라
FT-Transformer보다 훨씬 구체적인 성공 근거를 갖는다.

라이브러리는 처음부터 구현하지 않고 `pytorch_tabular`(NodeConfig, NODE 논문 저자들의
전신 프레임워크가 아니라 유지되는 3rd-party 구현체)를 사용한다.

pitcher_id/batter_id는 이번엔 ayeon 스타일 수동 임베딩이 아니라 pytorch_tabular
자체의 categorical_cols 파이프라인에 맡긴다(embedding_dims로 차원 직접 지정,
handle_unknown_categories=True로 val에서 미확인 ID 자동 처리) — NODE 프레임워크
자체의 카테고리 처리 방식을 그대로 신뢰하는 것이 이 실험의 "기존 라이브러리를
있는 그대로 써본다"는 취지에 더 맞는다.

1단계: val=2024 fold 하나로 빠른 스크리닝. CatBoost 902점 구성의 이 환경
재현치(776.56)와 비교.
"""

import time

import numpy as np
import pandas as pd
import torch

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import NodeConfig

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
K = 50

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS
ID_COLS = ["pitcher_id", "batter_id"]

BATCH_SIZE = 512  # NODE 기본 용량(2048 트리 x depth 6)이 batch=8192에서 OOM(25.91GiB
                   # 할당) - 트리당 2^depth개 리프 조합 텐서가 배치 크기에 비례해 커짐
MAX_EPOCHS = 60
PATIENCE = 6
EMBED_DIM_LOW_CARD = 4
EMBED_DIM_ID = 16
NODE_NUM_TREES = 512   # 기본값 2048 -> 4배 축소 (스크리닝 단계, OOM 회피)
NODE_DEPTH = 4          # 기본값 6 -> 리프 조합 수(2^depth) 4배 축소


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def prep_fold(raw_train, prior_table, val_season):
    train = add_shrinkage_features(raw_train, prior_table, k=K)
    train = add_combo_features(train)

    is_train = train["season"] < val_season
    is_val = train["season"] == val_season

    trend = fit_season_trend(train.loc[is_train])
    train = add_season_trend_feature(train, trend)

    BASE_FEATURES = [c for c in raw_train.columns if c not in (ID, TARGET)]
    FEATURES = BASE_FEATURES + SHRINKAGE_FEATURE_NAMES + COMBO_FEATURE_NAMES + TREND_FEATURE_NAMES
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS and c not in ID_COLS]
    CAT_ALL = CAT_COLS + ID_COLS

    train_df = train.loc[is_train, FEATURES + [TARGET]].reset_index(drop=True).copy()
    val_df = train.loc[is_val, FEATURES + [TARGET]].reset_index(drop=True).copy()

    for c in NUM_COLS:
        train_df[c] = pd.to_numeric(train_df[c], errors="coerce")
        val_df[c] = pd.to_numeric(val_df[c], errors="coerce")
    for c in CAT_ALL:
        train_df[c] = train_df[c].astype(str)
        val_df[c] = val_df[c].astype(str)

    # ---- pitcher_id/batter_id: pytorch_tabular의 handle_unknown_categories가
    # NODE 경로에서 실제로는 동작하지 않아(val의 미확인 ID -> CUDA device-side
    # assert, embedding_dims를 자동으로 맡겨도 재현됨) ayeon 방식대로 직접
    # "UNK" 카테고리를 사전 매핑한다. "UNK"가 train에도 실제로 존재해야
    # 라이브러리가 학습된 카테고리로 인식하므로, train의 일부 행(1%)을 무작위로
    # "UNK"로 치환해 실제 학습 신호를 갖는 unknown 임베딩을 만든다(가짜 행을
    # 추가하는 대신 실제 데이터를 재사용 - ayeon의 cold-start 처리와 동일 원리).
    rng = np.random.RandomState(42)
    for c in ID_COLS:
        seen_ids = set(train_df[c].unique())
        unk_mask = rng.rand(len(train_df)) < 0.01
        train_df.loc[unk_mask, c] = "UNK"
        seen_ids.discard("UNK")  # UNK로 치환된 행을 뺀 "진짜" 학습 ID 집합
        val_df[c] = val_df[c].where(val_df[c].isin(seen_ids), "UNK")

    train_df[TARGET] = train_df[TARGET].astype(int)
    val_df[TARGET] = val_df[TARGET].astype(int)

    return train_df, val_df, NUM_COLS, CAT_ALL


import os


def main():
    force_cpu = os.environ.get("FORCE_CPU") == "1"
    subsample = int(os.environ.get("SUBSAMPLE", "0"))
    device_str = "cpu" if force_cpu else ("gpu" if torch.cuda.is_available() else "cpu")
    print(f"accelerator={device_str} | subsample={subsample or 'off'}")

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    val_season = 2024
    train_df, val_df, NUM_COLS, CAT_ALL = prep_fold(raw_train, prior_table, val_season)
    if subsample:
        train_df = train_df.sample(n=min(subsample, len(train_df)), random_state=42).reset_index(drop=True)
        val_df = val_df.sample(n=min(subsample // 4, len(val_df)), random_state=42).reset_index(drop=True)
    print(f"train={len(train_df)} | val={len(val_df)} | 수치형={len(NUM_COLS)} | 범주형={len(CAT_ALL)}")

    # embedding_dims를 직접 지정했더니 pytorch_tabular 내부 카디널리티 계산과
    # 어긋나 CUDA device-side assert(임베딩 인덱스 OOB)가 발생 - 라이브러리 자체
    # 휴리스틱에 맡긴다(embedding_dims=None, NodeConfig 기본 동작)

    data_config = DataConfig(
        target=[TARGET],
        continuous_cols=NUM_COLS,
        categorical_cols=CAT_ALL,
        normalize_continuous_features=True,
        handle_unknown_categories=True,
        handle_missing_values=True,
        num_workers=0,
    )
    trainer_config = TrainerConfig(
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        early_stopping="valid_loss",
        early_stopping_patience=PATIENCE,
        early_stopping_mode="min",
        checkpoints=None,
        load_best=True,
        accelerator=device_str,
        devices=1,
        progress_bar="none",
        seed=42,
    )
    optimizer_config = OptimizerConfig()
    model_config = NodeConfig(
        task="classification",
        learning_rate=1e-3,
        num_trees=NODE_NUM_TREES,
        depth=NODE_DEPTH,
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    t = time.time()
    tabular_model.fit(train=train_df, validation=val_df)
    print(f"학습 완료 :: {time.time()-t:.1f}s")

    pred_df = tabular_model.predict(val_df)
    prob_col = [c for c in pred_df.columns if c.endswith("_1_probability")]
    assert prob_col, f"확률 컬럼을 못 찾음: {list(pred_df.columns)}"
    val_pred = pred_df[prob_col[0]].to_numpy()
    val_score = score(val_pred, val_df[TARGET].to_numpy())

    print(f"\n=== 스크리닝 결과 (val={val_season}) ===")
    print(f"  NODE : {val_score:8.2f}")
    print(f"  CatBoost 902점 구성(이 환경 재현치, 17단계) : 776.56")
    print(f"  차이: {val_score - 776.56:+.2f}")


if __name__ == "__main__":
    main()
