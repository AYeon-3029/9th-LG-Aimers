# 9th-LG-Aimers

LG Aimers Phase 2 (DACon) — `control_success` 예측 프로젝트 코드 모음.

## 구조

| 폴더 | 내용 |
|---|---|
| [`ayeon/`](ayeon) | 메인 모델. R(1군)/F(퓨처스) 리그별 분리 학습, LightGBM + XGBoost + CatBoost + DCNv2 (+ 임베딩) 앙상블. 학습 `train.py`, 제출 추론 `script.py`, 공용 로직/실험 기록 `common.py`, 실험 스크립트 `try_*.py`, Optuna 탐색, 로그 `logs/`, 학습된 모델 `model/` |
| [`hyunwook/`](hyunwook) | CatBoost 단일 모델 baseline 계열 (LB 902.04) 및 v2~v26 실험 스크립트, 실험 로그 `logs/`, 제출용 `submission/` |
| [`hyunmin/`](hyunmin) | 팀원 모델 및 convex blend / walk-forward 검증 스크립트 |
| [`docs/`](docs) | 대회 규칙, 평가 방식, 데이터 설명/분석, HANDOFF 문서 |

## 제외된 항목

용량·재현성 문제로 레포에 포함하지 않았습니다.

- 원본 데이터 (`train.csv`, `trackman_history.csv` 등) — 대회 사이트에서 내려받아 `ayeon/data/` 등에 배치
- 제출용 `*.zip`, 가상환경, `__pycache__`, `catboost_info`, `lightning_logs`
- `hyunmin`의 대용량 캐시 (`convex_blend_cache.npz`, `hyunmin_train_meta.csv`) 및 `hyunwook/model/`

## 실행

```bash
cd ayeon
pip install -r requirements.txt
python train.py      # 학습 (data/ 필요)
```

자세한 설계/실험 근거는 [`docs/HANDOFF.md`](docs/HANDOFF.md)와 `ayeon/common.py` docstring, 각 팀원 폴더의 README를 참고하세요.
