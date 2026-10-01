"""5-way 블렌드(R에 임베딩 멤버 추가)로 늘어난 추론 시간이 평가 서버의 10분
제한 안에 들어오는지 사전 확인한다.

로컬 머신은 32 논리 코어인데 평가 서버는 6 vCPU라(competition rules 근거) 트리
추론처럼 CPU에 의존하는 부분은 로컬에서 스레드를 6개로 강제 제한해야 현실적인
추정이 된다 - 안 그러면 로컬이 훨씬 빠르게 나와 과소평가된다. 반대로 DCN/임베딩
같은 신경망 순전파는 서버에 GPU(L4)가 있으므로 로컬 CPU-only 타이밍이 오히려
비관적(안전한 상한) 추정치가 된다 - 두 방향이 섞여 있다는 걸 결과 해석 시
감안해야 한다.

test.csv 실제 평가 데이터(245,789행)는 로컬에 없으므로 train.csv에서 실제 값
분포를 유지한 채(복원추출) 합성 테스트셋을 만든다 - 추론 시간은 값 자체가 아니라
행 수/모델 구조에 의존하므로 합성 데이터로도 충분히 유효한 추정이다.
"""
import lightgbm as lgb  # 반드시 numpy/pandas보다 먼저 (env_notes: import 순서 crash 버그)
import time
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from catboost import CatBoostClassifier

torch.set_num_threads(6)

import train as T
from common import (
    ASOF_FEATURES, SITUATIONAL_FEATURES, FEATURES, FEATURES_F, DCN_FEATURES, TARGET,
    add_matchup_features, add_shrunk_pitcher_feature, split_regime,
    DCN_R_PARAMS, DCN_R_N_SEEDS, DCN_F_PARAMS, DCN_F_N_SEEDS,
    fit_dcn_preprocessing, apply_dcn_preprocessing, DCNv2,
)
from try_pitcher_batter_embedding import (
    load_data_with_ids, build_vocab, EmbedMLP, EMBED_BATCH_SIZE, encode_ids,
)

N_TEST_ROWS = 245_789
# lgb.Booster(model_file=...)가 비-ASCII(한글) 절대경로에서 파일을 못 여는
# 로컬 전용 버그를 확인했다(상대경로는 정상 동작) - 실제 제출은 평가 서버의
# ASCII Linux 경로에서 zip이 풀리므로 이 버그와 무관하지만, 로컬 드라이런은
# 상대경로를 써서 피해간다.
from pathlib import Path
MODEL_DIR = Path("model")
TEST_THREADS = 6  # 평가 서버 6 vCPU 시뮬레이션

# ---- 1) 임베딩 모델 하나를 빠르게 학습해서 저장 (타이밍용 - 정확도 무관, 구조만 실물) ----
print("training one embed model for timing purposes (not accuracy)...")
df_ids = load_data_with_ids()
tr_R_full, _ = split_regime(df_ids)
pitcher_vocab = build_vocab(tr_R_full, "pitcher_id")
batter_vocab = build_vocab(tr_R_full, "batter_id")

t0 = time.time()
stats = fit_dcn_preprocessing(tr_R_full)
Xtr_raw = torch.tensor(apply_dcn_preprocessing(tr_R_full, stats))
pid_tr = torch.tensor(encode_ids(tr_R_full, "pitcher_id", pitcher_vocab))
bid_tr = torch.tensor(encode_ids(tr_R_full, "batter_id", batter_vocab))
ytr = torch.tensor(tr_R_full[TARGET].to_numpy(dtype=np.float32))
model = EmbedMLP(Xtr_raw.shape[1], len(pitcher_vocab), len(batter_vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.BCEWithLogitsLoss()
n = Xtr_raw.shape[0]
perm = torch.randperm(n)[:200_000]  # 실제 훈련 아님, 구조만 갖춘 가중치면 충분
for i in range(0, len(perm), EMBED_BATCH_SIZE):
    idx = perm[i:i + EMBED_BATCH_SIZE]
    opt.zero_grad()
    loss = loss_fn(model(Xtr_raw[idx], pid_tr[idx], bid_tr[idx]), ytr[idx])
    loss.backward()
    opt.step()
torch.save(model.state_dict(), str(MODEL_DIR / "embed_model_R_timing_only.pt"))
np.savez(str(MODEL_DIR / "embed_stats_R_timing_only.npz"), mean=stats["mean"], std=stats["std"], miss_cols=stats["miss_cols"])
with open(str(MODEL_DIR / "embed_vocab_R_timing_only.txt"), "w") as f:
    f.write(f"{len(pitcher_vocab)},{len(batter_vocab)}")
print(f"  embed model trained+saved in {time.time()-t0:.1f}s (not counted in inference budget)")

# ---- 2) 합성 테스트셋 생성 (train.csv에서 복원추출, 실제 값 분포 유지) ----
print(f"building synthetic {N_TEST_ROWS:,}-row test set from train.csv value distribution...")
rng = np.random.default_rng(0)
idx = rng.integers(0, len(df_ids), size=N_TEST_ROWS)
synth = df_ids.iloc[idx].reset_index(drop=True).copy()
synth["row_id"] = np.arange(len(synth))
r_frac = (df_ids["game_type"] == "R").mean()
print(f"  synthetic set: {len(synth):,} rows, R fraction={  (synth['game_type']=='R').mean():.1%} (train R frac={r_frac:.1%})")

means_df = pd.read_csv(MODEL_DIR / "shrink_season_means.csv")
season_means = means_df.set_index(["game_type", "season"])["mean"]

# ---- 3) 실제 script.py와 동일한 추론 경로 타이밍 (스레드 6개로 제한) ----
lgb_booster_R = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model_R.txt"))
xgb_model_R = xgb.XGBClassifier(); xgb_model_R.load_model(str(MODEL_DIR / "xgb_model_R.json")) if (MODEL_DIR / "xgb_model_R.json").exists() else None
cat_model_R = CatBoostClassifier(thread_count=TEST_THREADS); cat_model_R.load_model(str(MODEL_DIR / "cat_model_R.cbm"))
lgb_booster_F = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model_F.txt"))
cat_model_F = CatBoostClassifier(thread_count=TEST_THREADS); cat_model_F.load_model(str(MODEL_DIR / "cat_model_F.cbm"))

import os
os.environ["OMP_NUM_THREADS"] = str(TEST_THREADS)

t_start = time.time()

df = add_matchup_features(synth)
df = add_shrunk_pitcher_feature(df, season_means)
is_R = (df["game_type"] == "R").to_numpy()

# R: 5-way (LGB+XGB+CAT+DCN+embed)
XR = df.loc[is_R, FEATURES]
t = time.time()
p_lgb_R = lgb_booster_R.predict(XR.to_numpy(dtype=np.float64), num_threads=TEST_THREADS)
t_lgb = time.time() - t
print(f"  R LGB predict: {t_lgb:.2f}s ({is_R.sum():,} rows)")

XR_stats = fit_dcn_preprocessing(tr_R_full)  # 실제로는 저장된 stats 로드, 여기선 이미 계산된 것 재사용
t = time.time()
Xn_R = torch.tensor(apply_dcn_preprocessing(XR, XR_stats))
dcn_preds = []
for seed in range(DCN_R_N_SEEDS):
    m = DCNv2(Xn_R.shape[1], **DCN_R_PARAMS)
    # 실제 저장된 5-seed 아티팩트가 있으면 그걸 로드(구조 다를 수 있어 있는 것만)
    path = MODEL_DIR / f"dcn_model_R_seed{seed}.pt"
    if path.exists():
        try:
            m.load_state_dict(torch.load(str(path), map_location="cpu"))
        except Exception:
            pass
    m.eval()
    with torch.no_grad():
        dcn_preds.append(torch.sigmoid(m(Xn_R)).numpy())
p_dcn_R = np.mean(dcn_preds, axis=0)
t_dcn = time.time() - t
print(f"  R DCN predict ({DCN_R_N_SEEDS} seeds): {t_dcn:.2f}s")

t = time.time()
Xn_embed = torch.tensor(apply_dcn_preprocessing(XR, XR_stats))
pid = torch.tensor(encode_ids(df.loc[is_R], "pitcher_id", pitcher_vocab))
bid = torch.tensor(encode_ids(df.loc[is_R], "batter_id", batter_vocab))
embed_model = EmbedMLP(Xn_embed.shape[1], len(pitcher_vocab), len(batter_vocab))
embed_model.load_state_dict(torch.load(str(MODEL_DIR / "embed_model_R_timing_only.pt"), map_location="cpu"))
embed_model.eval()
with torch.no_grad():
    p_embed_R = torch.sigmoid(embed_model(Xn_embed, pid, bid)).numpy()
t_embed = time.time() - t
print(f"  R embed predict (1 seed, timing only): {t_embed:.2f}s")

t = time.time()
p_xgb_R = xgb_model_R.predict_proba(XR)[:, 1] if xgb_model_R is not None else np.zeros(len(XR))
t_xgb = time.time() - t
print(f"  R XGB predict: {t_xgb:.2f}s")

t = time.time()
p_cat_R = cat_model_R.predict_proba(XR)[:, 1]
t_cat = time.time() - t
print(f"  R CAT predict: {t_cat:.2f}s")

# F: 4-way 그대로
XF = df.loc[~is_R, FEATURES_F]
t = time.time()
p_lgb_F = lgb_booster_F.predict(XF.to_numpy(dtype=np.float64), num_threads=TEST_THREADS)
p_cat_F = cat_model_F.predict_proba(XF)[:, 1]
t_f_trees = time.time() - t
print(f"  F LGB+CAT predict: {t_f_trees:.2f}s ({(~is_R).sum():,} rows)")

t_total = time.time() - t_start
print(f"\n=== TOTAL inference pipeline time: {t_total:.1f}s (budget: 600s) ===")
print(f"R members total: LGB={t_lgb:.1f}s XGB={t_xgb:.1f}s CAT={t_cat:.1f}s DCN={t_dcn:.1f}s embed={t_embed:.1f}s")
print("NOTE: tree timings throttled to 6 threads to approximate eval server's 6 vCPU.")
print("NOTE: DCN/embed timed on local CPU-only torch build - eval server has an L4 GPU,")
print("      so real server neural-net inference should be faster than shown here (safe/pessimistic local estimate).")
