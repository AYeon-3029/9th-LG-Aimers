"""F 신규 DCN 아키텍처를 폐기하기 전 저비용 확인: 균등가중치(1/4) 대신 볼록결합
가중치를 신규 F DCN에 맞게 재탐색하면 531(구성 기준선)을 넘을 수 있는지 확인한다.

가중치는 이전 스태킹 실험과 동일한 방법론으로 - val2024를 직접 보고 최적화하면
테스트셋에 가중치를 맞추는 leakage이므로, tr_F의 OOF 예측(generate_oof_preds)에서
fit_convex_stack_from_oof로 가중치를 구하고, 그 가중치를 val2024 F에 "적용만" 해서
평가한다(val_df는 가중치 학습에 전혀 사용되지 않음 - fit_stack_meta의 독스트링과
동일한 논리)."""
import train as T
import numpy as np
from common import (
    TARGET, FEATURES_F, add_shrunk_pitcher_feature, compute_season_means,
    split_regime, brier_skill_score, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS,
    LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS,
)

df = T.load_data()
cutoff, val_season = 2023, 2024
d = np.load(str(T.CACHE_DIR / f"split_{cutoff}_{val_season}.npz"))
preds_F_old, y_F = d["preds_F"], d["y_F"]  # (n,4): lgb,xgb,cat,dcn(OLD arch) on val2024

train_df = df[df["season"] <= cutoff]
val_df = df[df["season"] == val_season]
season_means = compute_season_means(train_df)
train_df = add_shrunk_pitcher_feature(train_df, season_means)
val_df = add_shrunk_pitcher_feature(val_df, season_means)
_, tr_F = split_regime(train_df)
val_F = val_df[val_df["game_type"] == "F"]
print(f"n_train_F(OOF fit)={len(tr_F):,}  n_val_F={len(val_F):,}")

# 1) 신규 F DCN 아키텍처로 val2024 예측 재생성 (동일하게 5시드 평균, 프로덕션과 동일 절차)
p_dcn_F_new = T.fit_dcn_multiseed(tr_F, val_F, DCN_F_PARAMS, DCN_F_BATCH_SIZE, DCN_F_N_SEEDS)

# 2) 신규 F DCN 아키텍처로 OOF 예측 생성 후 볼록결합 가중치 탐색 (val을 보지 않음)
oof, y_oof = T.generate_oof_preds(tr_F, LGB_F_PARAMS, XGB_F_PARAMS, CAT_F_PARAMS, DCN_F_PARAMS, DCN_F_BATCH_SIZE, FEATURES_F)
w = T.fit_convex_stack_from_oof(oof, y_oof)
print(f"convex weights (lgb,xgb,cat,dcn) = {np.round(w, 4)}")

# 3) 그 가중치를 val2024 F에 적용
preds_val_new = np.column_stack([preds_F_old[:, 0], preds_F_old[:, 1], preds_F_old[:, 2], p_dcn_F_new])
p_reweighted = T.apply_convex_stack(preds_val_new, w)
_, bss_reweighted = brier_skill_score(y_F, p_reweighted)

_, bss_equal_old_dcn = brier_skill_score(y_F, preds_F_old.mean(axis=1))
p_equal_new_dcn = (preds_F_old[:, 0] + preds_F_old[:, 1] + preds_F_old[:, 2] + p_dcn_F_new) / 4
_, bss_equal_new_dcn = brier_skill_score(y_F, p_equal_new_dcn)

print(f"\n[baseline]        equal 1/4, OLD F-DCN   bss_F = {bss_equal_old_dcn:,.0f}")
print(f"[current adopted]  equal 1/4, NEW F-DCN   bss_F = {bss_equal_new_dcn:,.0f}")
print(f"[reweighted]       convex w, NEW F-DCN   bss_F = {bss_reweighted:,.0f}")
print(f"\n=> reweighted beats baseline(531)? {'YES' if bss_reweighted > bss_equal_old_dcn else 'NO'}")
