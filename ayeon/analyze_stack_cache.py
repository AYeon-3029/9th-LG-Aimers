"""_stack_cache/split_*.npz(walk_forward_check가 저장)를 읽어, 재학습 없이 여러
메타러너 후보(균등 평균 / L2 로지스틱 스윕 / 볼록결합)를 R 3구간 + F 다중구간
기준으로 한 번에 비교한다. train.py를 다시 돌리지 않아도 되므로 몇 초~몇 분
안에 끝난다 - 2026-08-21 OOF 스태킹 검증(1차: 무정규화 로지스틱 기각, 2차:
정규화/볼록결합 재검증)의 후속 분석용.
"""
import train as T  # lightgbm을 가장 먼저 import해야 하는 이 환경의 이슈 회피(memory: lgaimers-env-notes)
import numpy as np
from common import brier_skill_score, TARGET

CACHE_DIR = T.CACHE_DIR
SPLITS = [(2021, 2022), (2022, 2023), (2023, 2024)]
C_CANDIDATES = [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def load_split(cutoff, val_season):
    d = np.load(str(CACHE_DIR / f"split_{cutoff}_{val_season}.npz"))
    return {k: d[k] for k in d.files}


def main():
    avg_bss_list = []
    l2_bss_lists = {C: [] for C in C_CANDIDATES}
    convex_bss_list = []
    convex_weights_log = []

    cached_splits = {}
    for cutoff, val_season in SPLITS:
        path = CACHE_DIR / f"split_{cutoff}_{val_season}.npz"
        if not path.exists():
            print(f"[skip] {path} not found yet")
            continue
        d = load_split(cutoff, val_season)
        cached_splits[(cutoff, val_season)] = d

        preds_R, preds_F = d["preds_R"], d["preds_F"]
        y_R, y_F = d["y_R"], d["y_F"]
        y_all = np.concatenate([y_R, y_F])

        p_R_avg, p_F_avg = preds_R.mean(axis=1), preds_F.mean(axis=1)
        _, bss_R_avg = brier_skill_score(y_R, p_R_avg)
        _, bss_F_avg = brier_skill_score(y_F, p_F_avg) if len(y_F) else (None, float("nan"))
        _, bss_all_avg = brier_skill_score(y_all, np.concatenate([p_R_avg, p_F_avg]))
        avg_bss_list.append(bss_all_avg)
        print(f"[avg  ] train<={cutoff} val={val_season}  bss_R={bss_R_avg:,.0f}  bss_F={bss_F_avg:,.0f}  ENSEMBLE={bss_all_avg:,.0f}")

        for C in C_CANDIDATES:
            meta_R = T.fit_stack_meta_from_oof(d["oof_R"], d["y_oof_R"], C)
            meta_F = T.fit_stack_meta_from_oof(d["oof_F"], d["y_oof_F"], C)
            p_R = meta_R.predict_proba(preds_R)[:, 1]
            p_F = meta_F.predict_proba(preds_F)[:, 1]
            _, bss_R = brier_skill_score(y_R, p_R)
            _, bss_F = brier_skill_score(y_F, p_F) if len(y_F) else (None, float("nan"))
            _, bss_all = brier_skill_score(y_all, np.concatenate([p_R, p_F]))
            l2_bss_lists[C].append(bss_all)
            print(f"[L2 C={C:<6}] train<={cutoff} val={val_season}  bss_R={bss_R:,.0f}  bss_F={bss_F:,.0f}  ENSEMBLE={bss_all:,.0f}")

        w_R = T.fit_convex_stack_from_oof(d["oof_R"], d["y_oof_R"])
        w_F = T.fit_convex_stack_from_oof(d["oof_F"], d["y_oof_F"])
        p_R = T.apply_convex_stack(preds_R, w_R)
        p_F = T.apply_convex_stack(preds_F, w_F)
        _, bss_R = brier_skill_score(y_R, p_R)
        _, bss_F = brier_skill_score(y_F, p_F) if len(y_F) else (None, float("nan"))
        _, bss_all = brier_skill_score(y_all, np.concatenate([p_R, p_F]))
        convex_bss_list.append(bss_all)
        convex_weights_log.append((cutoff, val_season, w_R, w_F))
        print(f"[convex] train<={cutoff} val={val_season}  bss_R={bss_R:,.0f}  bss_F={bss_F:,.0f}  ENSEMBLE={bss_all:,.0f}")
        print(f"  weights R: lgb={w_R[0]:.3f} xgb={w_R[1]:.3f} cat={w_R[2]:.3f} dcn={w_R[3]:.3f}  |  "
              f"F: lgb={w_F[0]:.3f} xgb={w_F[1]:.3f} cat={w_F[2]:.3f} dcn={w_F[3]:.3f}")

        if cutoff == SPLITS[-1][0] and len(y_F) > 0:
            game_month_F = d["game_month_F"]
            print("[F multi-segment: avg vs convex vs best-L2, by game_month]")
            best_C = max(C_CANDIDATES, key=lambda C: l2_bss_lists[C][-1] if l2_bss_lists[C] else -1)
            meta_F_best = T.fit_stack_meta_from_oof(d["oof_F"], d["y_oof_F"], best_C)
            p_F_bestL2 = meta_F_best.predict_proba(preds_F)[:, 1]
            for months in T.F_SEGMENTS:
                mask = np.isin(game_month_F, months)
                if mask.sum() == 0:
                    continue
                _, b_avg = brier_skill_score(y_F[mask], p_F_avg[mask])
                _, b_cvx = brier_skill_score(y_F[mask], p_F[mask])
                _, b_l2 = brier_skill_score(y_F[mask], p_F_bestL2[mask])
                print(f"  game_month in {months}: n={mask.sum():,}  avg={b_avg:,.0f}  convex={b_cvx:,.0f}  L2(C={best_C})={b_l2:,.0f}")

    print()
    print(f"[SUMMARY] ENSEMBLE AVG    BSS = {np.mean(avg_bss_list):,.0f}  per-split={[round(v) for v in avg_bss_list]}")
    for C in C_CANDIDATES:
        if l2_bss_lists[C]:
            print(f"[SUMMARY] ENSEMBLE L2(C={C:<6}) BSS = {np.mean(l2_bss_lists[C]):,.0f}  per-split={[round(v) for v in l2_bss_lists[C]]}")
    if convex_bss_list:
        print(f"[SUMMARY] ENSEMBLE CONVEX BSS = {np.mean(convex_bss_list):,.0f}  per-split={[round(v) for v in convex_bss_list]}")


if __name__ == "__main__":
    main()
