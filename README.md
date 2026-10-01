<div align="center">

# ⚾ 9th LG Aimers · Phase 2

**투구 직전 정보만으로 제구 성공 확률(`control_success`)을 예측하는 AI 모델**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-✓-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-✓-EB5E28)
![CatBoost](https://img.shields.io/badge/CatBoost-✓-FFCC00)
![PyTorch](https://img.shields.io/badge/PyTorch-DCNv2-EE4C2C?logo=pytorch&logoColor=white)

</div>

---

## 📌 대회 개요

| 항목 | 내용 |
|---|---|
| 과제 | 투구 단위 제구 성공 확률 예측 (이진 확률 예측) |
| 평가 지표 | Brier Skill Score — `100000 × (1 − BS / (r(1−r)))` , 높을수록 좋음 |
| 데이터 | 경기 상황 · 선수 · 주자 정보 + 트랙맨(2019~2024) 과거 투구 이력 |
| 제출 방식 | `submit.zip` (`model/` + `script.py`) 코드 제출 |

> 제구 실패 = ① 존 한가운데 ② 존에서 크게 벗어남 ③ 포수 요구와 반대 방향

## 🏆 리더보드 기록

| 모델 | 담당 | LB 점수 |
|---|---|---|
| R/F 분리 4-모델 앙상블 (`ayeon/`) | AYeon | **792.24** |
| CatBoost 단일 모델 (`hyunwook/`) | Hyunwook | **902.04** |

## 🗂 레포 구조

```
9th-LG-Aimers/
├── ayeon/        # 메인: R/F 리그 분리 앙상블
├── hyunwook/     # CatBoost 계열 실험 (v2 ~ v26)
├── hyunmin/      # 팀원 모델 · blend / walk-forward 검증
└── docs/         # 대회 규칙 · 평가 · 데이터 설명 · HANDOFF
```

| 폴더 | 설명 |
|---|---|
| [`ayeon/`](ayeon) | `train.py` 학습, `script.py` 추론(자체 완결형), `common.py` 공용 로직 및 실험 기록, `try_*.py` 실험, Optuna 탐색, `logs/`, `model/` |
| [`hyunwook/`](hyunwook) | `train_v*.py` 실험 스크립트, `features.py`, `submission/` 제출용 코드, `logs/` 실험 로그, 상세 [README](hyunwook/README.md) |
| [`hyunmin/`](hyunmin) | convex blend, walk-forward 스크립트, 상세 [README](hyunmin/README.md) |
| [`docs/`](docs) | [HANDOFF](docs/HANDOFF.md) · [규칙](docs/rule.md) · [평가](docs/evaluation.md) · [데이터 설명](docs/data_description.md) · [분석 리포트](docs/data_analysis_report.md) |

## 🧠 메인 모델 (`ayeon/`)

- **리그별 완전 분리**: `R`(1군, 약 130만 행) / `F`(퓨처스, 2023 시즌 이후만 사용 — 2022→2023 성공률 체제 변화 때문)
- **앙상블**: LightGBM + XGBoost + CatBoost + DCNv2 (+ 임베딩), 동일 가중 평균
- **피처**: as-of 집계(시점 누수 방지) · 투타 손 매치업 · 상황 피처
- **검증**: 시즌 기준 walk-forward (≤2023 학습 → 2024 검증)

## 🚀 실행

```bash
cd ayeon
pip install -r requirements.txt

# 데이터(train.csv, trackman_history.csv 등)를 ayeon/data/ 에 배치
python train.py          # 학습 → model/ 생성
python script.py         # 추론 (제출 환경과 동일한 진입점)
```

> ⚠️ `script.py`는 제출 형식상 `common.py`를 import하지 않습니다. `common.py`의 상수를 바꾸면 `script.py`에도 같이 반영해야 합니다.

## 📦 레포에서 제외된 항목

용량 · 재현성 문제로 포함하지 않았습니다.

- 원본 데이터 (`train.csv`, `trackman_history.csv` 등)
- 제출용 `*.zip`, 가상환경, `__pycache__`, `catboost_info`, `lightning_logs`
- `hyunmin/` 대용량 캐시 (`convex_blend_cache.npz`, `hyunmin_train_meta.csv`)
- `hyunwook/model/`

## 👥 팀

| | |
|---|---|
| **AYeon** | 메인 앙상블 (`ayeon/`) |
| **Hyunwook** | CatBoost 계열 (`hyunwook/`) |
| **Hyunmin** | blend / 검증 (`hyunmin/`) |
