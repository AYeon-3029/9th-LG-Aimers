"""search_f_dcn_arch.py의 '366->462'가 이미 whole-split(월 미분할) 계산인지,
그리고 isolate_r_vs_f_change.py의 531/526(4모델 블렌드)과 dcn 단독 수치가
실제로 어떻게 다른지 직접 대조 확인.

_stack_cache의 preds_F_old[:,3]이 정확히 구아키텍처 DCN(5시드) 예측이라는 전제 하에,
이를 brier_skill_score로 whole-split 단일 계산하면 search_f_dcn_arch.py가 보고한
366과 일치해야 한다(둘 다 DCN 단독, 둘 다 월 미분할 - 같은 계산이어야 함).
"""
import train as T
import numpy as np
from common import (
    TARGET, add_shrunk_pitcher_feature, compute_season_means, split_regime,
    brier_skill_score, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
)

OLD_DCN_F_PARAMS = dict(cross_layers=2, deep_dims=(32, 16), dropout=0.2)

df = T.load_data()
cutoff, val_season = 2023, 2024
d = np.load(str(T.CACHE_DIR / f"split_{cutoff}_{val_season}.npz"))
preds_F_old, y_F = d["preds_F"], d["y_F"]

train_df = df[df["season"] <= cutoff]
val_df = df[df["season"] == val_season]
season_means = compute_season_means(train_df)
train_df = add_shrunk_pitcher_feature(train_df, season_means)
val_df = add_shrunk_pitcher_feature(val_df, season_means)
_, tr_F = split_regime(train_df)
val_F = val_df[val_df["game_type"] == "F"]
y_F_fresh = val_F[TARGET].to_numpy()
print(f"n_val_F={len(val_F)}  y_F cache matches fresh: {np.array_equal(y_F, y_F_fresh)}")

# DCN 단독, 구 아키텍처 (캐시 재사용, whole-split)
_, bss_dcn_old_whole = brier_skill_score(y_F, preds_F_old[:, 3])
print(f"[DCN alone, OLD arch] whole-split bss_F = {bss_dcn_old_whole:,.0f}  (search_f_dcn_arch.py reported 366)")

# DCN 단독, 신 아키텍처 (재학습 필요, whole-split)
p_dcn_new = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)
_, bss_dcn_new_whole = brier_skill_score(y_F, p_dcn_new)
print(f"[DCN alone, NEW arch] whole-split bss_F = {bss_dcn_new_whole:,.0f}  (search_f_dcn_arch.py reported 462)")

# 참고: 4모델 블렌드 (isolate_r_vs_f_change.py 결과 재확인용)
p_blend_old = (preds_F_old[:, 0] + preds_F_old[:, 1] + preds_F_old[:, 2] + preds_F_old[:, 3]) / 4
p_blend_new = (preds_F_old[:, 0] + preds_F_old[:, 1] + preds_F_old[:, 2] + p_dcn_new) / 4
_, bss_blend_old = brier_skill_score(y_F, p_blend_old)
_, bss_blend_new = brier_skill_score(y_F, p_blend_new)
print(f"[4-model BLEND, OLD F-DCN] whole-split bss_F = {bss_blend_old:,.0f}  (isolate script reported 531)")
print(f"[4-model BLEND, NEW F-DCN] whole-split bss_F = {bss_blend_new:,.0f}  (isolate script reported 526)")
