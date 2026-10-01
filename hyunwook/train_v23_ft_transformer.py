"""23단계 — FT-Transformer(rtdl_revisiting_models) 단독 성능 스크리닝.

배경: CatBoost 하이퍼파라미터 튜닝 계열은 5번(k값/정규화/선수ID 범주형/
grow_policy/depth+lr+l2 조합) 시도해서 5번 다 실제 LB에서 뒤집혔다 — 이 방향은
소진됐다고 판단, 완전히 다른 모델 구조로 전환한다.

이 스크립트가 CatBoost와 다르게 성공할 수 있다고 보는 구체적 근거: 두 팀 모두
pitcher_id/batter_id를 CatBoost의 원시 범주형(ordered target statistics)으로
넣었다가 실패했다(-29~-174 / 실제 LB 755.91<758.37). 그런데 ayeon 파이프라인의
저차원 학습 임베딩(nn.Embedding, 고정 차원, 경사하강법으로 학습)은 R에서 실제로
검증됐다([[lgaimers-model-state]] 항목 5/6) — 완전히 다른 인코딩 메커니즘이라
CatBoost가 걸린 함정에 안 걸린다. 이 스크립트는 그 검증된 임베딩 메커니즘을
FT-Transformer와 결합해, CatBoost가 못 살린 pitcher_id/batter_id 신호를 다른
구조로 살릴 수 있는지 확인한다.

구성:
  - 수치형(count/situational/shrinkage/trend 등, pitcher_id/batter_id 제외)은
    FT-Transformer의 LinearEmbeddings(연속값 토크나이저)로.
  - 저카디널리티 범주형 8개(top_bottom/game_type/base_state/team_id 2개/
    count_state/hand_matchup/outs_base_state)는 FT-Transformer의
    CategoricalEmbeddings로.
  - pitcher_id/batter_id는 FT-Transformer 토큰이 아니라 별도
    nn.Embedding(dim=16, 0=unknown, ayeon의 build_id_vocab/encode_ids 그대로
    재사용)으로 인코딩해, FT-Transformer의 pooled 표현(d_out=None, CLS 토큰)과
    concat한 뒤 작은 MLP 헤드로 최종 로짓을 낸다.
  - hyunwook님 902.04 구성과 동일한 피처 엔지니어링(shrinkage k=50, combo 18개,
    season-trend)을 그대로 재사용, R/F 분리하지 않음(hyunwook님 구조적 선택
    유지 — 공정 비교).

1단계: val=2024 fold 하나로 빠른 스크리닝(전체 다중fold 전 신호 유무만 확인).
902점 구성의 이 환경 재현치(776.56, 17단계 기준)와 비교한다.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import rtdl_revisiting_models as rtdl
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
K = 50

BASE_CAT_COLS = ["top_bottom", "game_type", "base_state", "pitcher_team_id", "batter_team_id"]
CAT_COLS = BASE_CAT_COLS + CATEGORICAL_COMBO_COLS
ID_COLS = ["pitcher_id", "batter_id"]

EMBED_DIM = 16
BATCH_SIZE = 1024          # GPU 메모리 한계상 물리 배치는 이만큼
ACCUM_STEPS = 8             # 유효 배치 1024*8=8192로 복원 (gradient accumulation)
LR = 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 500           # optimizer step(=accumulation 완료 시점) 기준 linear warmup
MAX_EPOCHS = 60
PATIENCE = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def score(pred, y):
    r = y.mean()
    brier = ((pred - y) ** 2).mean()
    base = r * (1 - r)
    return max(0, 100000 * (1 - brier / base))


def build_id_vocab(train_df, col):
    """train에서만 어휘를 만든다(val/test 절대 미포함 - 누출 방지). 0=unknown."""
    return {v: i + 1 for i, v in enumerate(sorted(train_df[col].unique()))}


def encode_ids(df, col, vocab):
    return df[col].map(vocab).fillna(0).to_numpy(dtype=np.int64)


class FTTWithIdEmbed(nn.Module):
    def __init__(self, n_cont, cat_cardinalities, n_pitchers, n_batters, embed_dim=EMBED_DIM):
        super().__init__()
        ft_kwargs = rtdl.FTTransformer.get_default_kwargs()
        self.backbone = rtdl.FTTransformer(
            n_cont_features=n_cont, cat_cardinalities=cat_cardinalities, d_out=None, **ft_kwargs
        )
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, embed_dim)
        self.batter_emb = nn.Embedding(n_batters + 1, embed_dim)
        d_block = ft_kwargs["d_block"]
        self.head = nn.Sequential(
            nn.Linear(d_block + 2 * embed_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x_cont, x_cat, pid, bid):
        h = self.backbone(x_cont, x_cat)
        h = torch.cat([h, self.pitcher_emb(pid), self.batter_emb(bid)], dim=1)
        return self.head(h).squeeze(-1)


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

    train_df = train.loc[is_train].reset_index(drop=True)
    val_df = train.loc[is_val].reset_index(drop=True)

    # ---- 수치형: train 기준 median 대치 + 표준화 ----
    num_train = train_df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    num_val = val_df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    medians = num_train.median()
    num_train = num_train.fillna(medians)
    num_val = num_val.fillna(medians)
    means = num_train.mean()
    stds = num_train.std().replace(0, 1.0)
    x_cont_train = ((num_train - means) / stds).to_numpy(dtype=np.float32)
    x_cont_val = ((num_val - means) / stds).to_numpy(dtype=np.float32)

    # ---- 저카디널리티 범주형: train 기준 OrdinalEncoder, 미확인 값은 별도 인덱스 ----
    cat_train_raw = train_df[CAT_COLS].astype(str)
    cat_val_raw = val_df[CAT_COLS].astype(str)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    x_cat_train = enc.fit_transform(cat_train_raw)
    cat_cardinalities = [len(cats) + 1 for cats in enc.categories_]  # +1 = unknown 슬롯
    for j, card in enumerate(cat_cardinalities):
        x_cat_train[:, j] = np.where(x_cat_train[:, j] < 0, card - 1, x_cat_train[:, j])
    x_cat_val = enc.transform(cat_val_raw)
    for j, card in enumerate(cat_cardinalities):
        x_cat_val[:, j] = np.where(x_cat_val[:, j] < 0, card - 1, x_cat_val[:, j])

    # ---- pitcher/batter ID: ayeon 방식 임베딩 어휘 (train만, 0=unknown) ----
    pitcher_vocab = build_id_vocab(train_df, "pitcher_id")
    batter_vocab = build_id_vocab(train_df, "batter_id")
    pid_train = encode_ids(train_df, "pitcher_id", pitcher_vocab)
    bid_train = encode_ids(train_df, "batter_id", batter_vocab)
    pid_val = encode_ids(val_df, "pitcher_id", pitcher_vocab)
    bid_val = encode_ids(val_df, "batter_id", batter_vocab)

    y_train = train_df[TARGET].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET].to_numpy(dtype=np.float32)

    data = dict(
        x_cont_train=x_cont_train, x_cat_train=x_cat_train.astype(np.int64),
        pid_train=pid_train, bid_train=bid_train, y_train=y_train,
        x_cont_val=x_cont_val, x_cat_val=x_cat_val.astype(np.int64),
        pid_val=pid_val, bid_val=bid_val, y_val=y_val,
        cat_cardinalities=cat_cardinalities,
        n_pitchers=len(pitcher_vocab), n_batters=len(batter_vocab),
        n_cont=len(NUM_COLS),
    )
    return data


def to_tensor(x, dtype, device):
    return torch.as_tensor(x, dtype=dtype, device=device)


def run_fold(raw_train, prior_table, val_season):
    print(f"\n=== val={val_season} : 피처 준비 ===")
    d = prep_fold(raw_train, prior_table, val_season)
    print(f"  n_cont={d['n_cont']} | cat_cardinalities={d['cat_cardinalities']} | "
          f"n_pitchers={d['n_pitchers']} | n_batters={d['n_batters']}")
    print(f"  train={len(d['y_train'])} | val={len(d['y_val'])}")

    x_cont_train = to_tensor(d["x_cont_train"], torch.float32, DEVICE)
    x_cat_train = to_tensor(d["x_cat_train"], torch.long, DEVICE)
    pid_train = to_tensor(d["pid_train"], torch.long, DEVICE)
    bid_train = to_tensor(d["bid_train"], torch.long, DEVICE)
    y_train = to_tensor(d["y_train"], torch.float32, DEVICE)

    x_cont_val = to_tensor(d["x_cont_val"], torch.float32, DEVICE)
    x_cat_val = to_tensor(d["x_cat_val"], torch.long, DEVICE)
    pid_val = to_tensor(d["pid_val"], torch.long, DEVICE)
    bid_val = to_tensor(d["bid_val"], torch.long, DEVICE)
    y_val_np = d["y_val"]

    torch.manual_seed(42)
    model = FTTWithIdEmbed(d["n_cont"], d["cat_cardinalities"], d["n_pitchers"], d["n_batters"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss()

    n = x_cont_train.shape[0]
    best_score, best_state, no_improve = -1, None, 0
    global_step = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t = time.time()
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        opt.zero_grad()
        n_micro = (n + BATCH_SIZE - 1) // BATCH_SIZE
        for micro_i, start in enumerate(range(0, n, BATCH_SIZE)):
            idx = perm[start:start + BATCH_SIZE]
            logit = model(x_cont_train[idx], x_cat_train[idx], pid_train[idx], bid_train[idx])
            loss = loss_fn(logit, y_train[idx])
            (loss / ACCUM_STEPS).backward()
            total_loss += loss.item() * len(idx)

            is_last_micro = (micro_i == n_micro - 1)
            if (micro_i + 1) % ACCUM_STEPS == 0 or is_last_micro:
                global_step += 1
                lr_scale = min(1.0, global_step / WARMUP_STEPS)
                for g in opt.param_groups:
                    g["lr"] = LR * lr_scale
                opt.step()
                opt.zero_grad()
        train_loss = total_loss / n

        model.eval()
        with torch.no_grad():
            val_logit = []
            for start in range(0, x_cont_val.shape[0], BATCH_SIZE):
                sl = slice(start, start + BATCH_SIZE)
                val_logit.append(model(x_cont_val[sl], x_cat_val[sl], pid_val[sl], bid_val[sl]))
            val_pred = torch.sigmoid(torch.cat(val_logit)).cpu().numpy()
        val_score = score(val_pred, y_val_np)

        improved = val_score > best_score
        if improved:
            best_score = val_score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        print(f"  epoch={epoch:3d} | lr={LR*min(1.0, global_step/WARMUP_STEPS):.2e} | "
              f"train_loss={train_loss:.5f} | val Score={val_score:8.2f} "
              f"{'*' if improved else '':1s} | {time.time()-t:.1f}s")

        if no_improve >= PATIENCE:
            print(f"  조기종료 (patience={PATIENCE})")
            break

    print(f"  최고 val Score={best_score:.2f}")
    return best_score


def main():
    print(f"device={DEVICE}")
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    BASE_FEATURES = [c for c in test_cols if c != ID]
    raw_train = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                             usecols=BASE_FEATURES + [TARGET])
    prior_table = load_prior_table()

    val_season = 2024
    best = run_fold(raw_train, prior_table, val_season)

    print(f"\n=== 스크리닝 결과 (val={val_season}) ===")
    print(f"  FT-Transformer+ID임베딩 : {best:8.2f}")
    print(f"  CatBoost 902점 구성(이 환경 재현치, 17단계) : 776.56")
    print(f"  차이: {best - 776.56:+.2f}")


if __name__ == "__main__":
    main()
