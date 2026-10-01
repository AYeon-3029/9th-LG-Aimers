# 개발 환경 이전 가이드 (LG Aimers Phase2 — control_success 예측)

이 문서는 새 환경으로 옮길 때 빠르게 다시 궤도에 오르기 위한 압축 요약이다.
**실험 하나하나의 상세 근거(수치, 실패 이유)는 전부 `common.py` 맨 위
docstring에 시간순으로 기록돼 있다 — 뭔가 다시 시도하고 싶으면 먼저 거기서
검색해볼 것.** 이 문서는 그 로그를 항해할 수 있게 해주는 지도 역할만 한다.

## 0. 최우선으로 알아야 할 것

- **평가 서버 환경을 반드시 맞춰라.** `evaluation.md`에 서버 스펙이 있다
  (Ubuntu 22.04, `torch==2.7.1+cu128`, Python 3.11.15). 로컬 torch가 서버보다
  **최신** 버전이면 `torch.save`로 저장한 체크포인트를 서버가 못 읽을 위험이
  있다(직렬화 포맷은 하위호환은 되지만 상위호환은 보장 안 됨). 새 환경에서
  가장 먼저 할 일: `pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu`
  (또는 서버와 동일 버전).
- **`script.py`는 `common.py`를 import하지 않는 완전 자체완결 파일이다**
  (zip 제출 규격 때문 - `rule.md` 참고). `FEATURES`, `DCN_R_PARAMS`,
  `DCN_R_N_SEEDS` 등 여러 상수를 **손으로 복제**해서 갖고 있다. `common.py`를
  고칠 때마다 `script.py`의 대응 상수를 반드시 같이 고쳐야 한다 - 이걸
  두 번이나 깜빡해서 크래시나는 zip을 만들 뻔했다. 매번 아래 4번 스모크
  테스트를 거칠 것.

## 1. 대회 개요

- DACon 코드 제출 대회. `data/train.csv`의 각 투구(row)에 대해
  `control_success`(제구 성공 여부) 확률을 예측.
- 리더보드 산식: `100000 × (1 - Brier/베이스라인)` (BSS), `evaluation.md` 참고.
- 수료 기준선 LB 549.51. 하루 제출 5회 제한(`rule.md`).
- 리포: GitHub `duriadda/LGAimers`, 작업 브랜치 `ayeon`.

## 2. 데이터의 핵심 특성 (왜 이런 구조가 됐는지)

- `game_type`이 R(1군)/F(퓨처스) 두 값. **F가 2022→2023 시즌 사이 성공률이
  뒤집힌다**(0.71→0.47, 계단형 레짐 전환). R은 계단 없이 매년 완만하게
  우하향하는 연속 드리프트(-0.01~-0.02/시즌).
- R은 130만 행, F는 신체제(2023~)만 쓰면 5.6만 행뿐 — **표본 크기가
  완전히 다른 문제라서 검증 방법론도 R/F를 다르게 취급해야 한다** (5번 참고).
- `asof_*` 피처(19개)는 대회 주최측이 이미 as-of(그 시점까지 누적) 방식으로
  계산해 제공한 것 - 리키지 없음. 직접 추가 파생을 만들 때는 항상 leak-safe
  여부를 먼저 따질 것.

## 3. 현재 아키텍처 (`common.py`/`train.py`/`script.py`)

```
game_type == R  →  R 모델 (전체 데이터, 130만행)
game_type == F  →  F 모델 (신체제만, season >= 2023, 5.6만행)

각 리그마다: LightGBM + XGBoost + CatBoost + DCNv2, 균등 가중(1/4) 평균
```

- **피처**: `ASOF_FEATURES`(19) + `MATCHUP_FEATURES`(pitcher/batter 손 조합,
  5) + `SITUATIONAL_FEATURES`(볼카운트/이닝/승리기대값 등, 6) = 30개 공통.
  **`shrunk_pitcher_rate`(asof_pitcher_n 기반 shrinkage)는 R만 씀** -
  `FEATURES`(31, R용) vs `FEATURES_F`(30, F용, shrinkage 없음)로 나뉜다.
- **DCNv2**: R은 `cross_layers=2, deep_dims=(256,128)`, F는
  `cross_layers=2, deep_dims=(32,16)`(표본 작아 훨씬 작게). 둘 다 여러
  시드로 독립 재학습해 예측을 평균(`DCN_R_N_SEEDS=3`, `DCN_F_N_SEEDS=5`) -
  단일 시드는 특히 F에서 심하게 불안정하다(bss_F가 33~514까지 요동).
  `DCN_FEATURES`는 shrinkage를 뺀 30개(다중공선성으로 CrossLayer가
  붕괴하는 걸 발견해서 제외).
- **하이퍼파라미터**: `LGB_BASE_PARAMS`/`XGB_BASE_PARAMS`/`CAT_BASE_PARAMS`가
  R용, `LGB_F_PARAMS`/`XGB_F_PARAMS`/`CAT_F_PARAMS`가 F용(더 얕은 트리 -
  F는 표본이 작아서 오히려 더 보수적인 게 이긴다, 직관과 반대).
- **모델 파일**: `model/{lgb,xgb,cat}_model_{R,F}.*`,
  `model/dcn_model_{R,F}_seed{N}.pt`, `model/dcn_stats_{R,F}.npz`,
  `model/shrink_season_means.csv`. 전부 git에 안 잡힘(바이너리, .gitignore
  대상 아니지만 그냥 안 커밋해왔음) - 환경 옮기면 `python train.py`로
  처음부터 재학습해야 한다.

## 4. 재현·제출 절차 (매번 이 순서대로)

```bash
# 1) 학습 (walk-forward 진단 출력 + model/ 저장, 수 분~수십 분)
python train.py

# 2) submit.zip 빌드 (model/*, script.py, requirements.txt만 - 평탄 구조)
python -c "
import zipfile
from pathlib import Path
files = sorted(Path('model').glob('*')) + [Path('script.py'), Path('requirements.txt')]
with zipfile.ZipFile('submit.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files: z.write(f, arcname=f.as_posix())
"

# 3) 격리 스모크 테스트 - 반드시 이걸 거칠 것 (dead code/desync를 이렇게 잡았음)
#    - submit.zip을 완전히 별도 디렉터리에 풀고
#    - data/test.csv에 game_type='F'인 합성 행을 섞어 R/F 라우팅 둘 다 검증
#    - python script.py 실행해서 끝까지 에러 없이 도는지 + 값이 sane한지 확인

# 4) 피처 이름/순서까지 확인 (개수만 맞고 순서/내용이 다른 조용한 버그 방지)
python -c "
import lightgbm as lgb
from common import FEATURES, FEATURES_F
print(lgb.Booster(model_file='model/lgb_model_R.txt').feature_name() == FEATURES)
print(lgb.Booster(model_file='model/lgb_model_F.txt').feature_name() == FEATURES_F)
"
```

이 순서를 안 지켜서 "피처 개수 불일치로 서버에서 크래시"를 두 번 낼 뻔했다.

## 5. 검증 방법론 규칙 (가장 비싸게 배운 교훈들)

1. **R은 walk-forward 3구간(train≤2021→val2022, ≤2022→val2023,
   ≤2023→val2024)만으로 충분히 신뢰할 수 있다.** 데이터가 많아서.
2. **F는 신체제 데이터가 1년(2024)뿐이라 3구간 walk-forward 자체가 안 된다**
   (2021/2022 컷오프는 F 신체제 데이터가 아직 없음). 그래서 F는 **val=2024를
   최소 2~4개 구간(월별)으로 쪼개서 각 구간이 같은 방향인지** 반드시 확인.
   전체 평균 하나만 보면 한 구간의 우연(예: 7-8월 한 구간이 만든 +40)에
   속아서 "순효과 +10"처럼 보이는 착시가 반복해서 발생했다.
3. **"과거 레벨/추세로 미래를 명시적으로 보정"하는 접근은 이 데이터에서
   구조적으로 잘 안 통한다** - R 시즌 센터링, 레짐 가중치, Platt 재보정
   세 가지가 전부 이 패턴으로 실패했다(R의 드리프트 속도 자체가 매년
   달라서 "그 해 고유의 잔차"를 다음 해로 못 옮김). 반대로 shrinkage
   ("표본 작을 때 더 큰 표본 쪽으로 끌어당기기")는 레벨을 외삽하는 게
   아니라서 이 실패 패턴에 안 걸리고, R에서 실제로 채택됨.
4. **캘리브레이션(Platt 등)을 검증할 때는 "같은 해 안에서 held-out 분할"로는
   부족하다 - 반드시 다른 해로 전이되는지까지 확인해야 한다.** 월별 분할
   검증은 통과했는데 실제로는 그 해 고유의 잔차를 잡은 것이었고, 다른 해로
   옮기면(2023 적합→2024 적용) 오히려 악화됐다. 최종 배포(2024 적합→2025
   적용)도 똑같은 구조라 신뢰 불가 판정, 철회함.
5. **개별 멤버 성능과 앙상블 기여도는 반대로 갈 수 있다.** TabM이 DCNv2보다
   단독 성능은 훨씬 좋았지만(F에서), 트리와 오차 패턴이 덜 달라서 앙상블에
   블렌딩하면 오히려 DCNv2보다 나빴다.
6. **통제 실험은 변수를 하나만 바꿔라.** "772.38 완전 재현"을 하겠다고
   R shrinkage 제거 + F DCN 다중시드까지 한꺼번에 되돌렸다가, 이미 검증됐던
   F 다중시드 개선분까지 같이 꺼져서(F 단일시드가 하필 나쁜 값을 뽑음,
   bss_F=33) 엉뚱한 결론을 낼 뻔했다.
7. **새 피처를 R에 추가하는 것 자체가 구조적으로 -5~-8 정도의 비용을 물 수
   있다** (`feature_fraction=0.6` 서브샘플링이 늘어난 피처 수만큼 기존
   유용한 피처의 선택 확률을 희석하는 것으로 추정). 그보다 뚜렷이 큰
   개선이 아니면 채택하지 않는다.
8. LB 제출은 하루 5회 제한 - 로컬 다중구간 검증으로 최대한 걸러내고,
   진짜 애매한 것만 제출로 확인.

## 6. 지금까지 채택된 것 / 기각된 것 요약

**채택** (`common.py` 상단 rationale 참고):
- R/F 완전 분리 모델 + game_type 라우팅
- F는 신체제(season≥2023)만 학습
- LGB+XGB+CAT+DCNv2 4-way 균등 가중 앙상블
- R 전용 shrinkage (직전 시즌 game_type 평균 prior)
- DCNv2 다중시드 평균화 (R=3, F=5)
- R DCN 아키텍처 재탐색 (cross_layers=2, deep_dims=(256,128))
- F 전용 경량 트리 하이퍼파라미터
- `TREE_SEED=0` 고정 (재현성)

**기각** (전부 근거와 함께 `common.py`에 기록됨 - 재시도 전 꼭 확인):
- `season`/`game_type`을 원시 피처로 직접 추가
- R 시즌 드리프트 센터링, 레짐 가중치, Platt scaling (5번 규칙)
- F 전용 shrinkage 채택 + F 하이퍼파라미터 재탐색 (한때 채택했다가 다구간
  검증에서 상쇄 패턴으로 드러나 철회)
- `pitcher_id`/`batter_id`를 CatBoost 범주형으로
- 위기상황(li 상위 20%) 조건부 투수 성공률(shrinkage 버전 포함)
- TabM(BatchEnsemble) - F 앙상블 멤버로
- `pitcher_win_expectancy`, `score_diff_pitcher_team`, `base_state`,
  `game_month`, `game_dayofweek`, `top_bottom` (R 피처 확장 시도 6종 전부)
- 볼카운트 파생, trackman 구위 지표, OOF 위기상황 피처 (공유 모델 시절)

## 7. 현재 상태

- 로컬 walk-forward(`train≤2023→val=2024`): **bss_all≈731, bss_R≈715,
  bss_F≈525**
- LB 히스토리: 772.38(최고, 최초 R/F 분리+DCNv2 제출) → 763대(shrinkage+F
  하이퍼파라미터+Platt 번들 제출, 회귀) → Platt/F 원인 조사 및 롤백 진행 중.
  763대의 정확한 원인은 F 쪽(shrinkage+하이퍼파라미터, 다구간 검증에서
  상쇄 패턴 확인)일 가능성이 높지만 100% 특정되지는 않음.
- 마지막 커밋: `e99a675`(R shrinkage 롤백 + script.py 동기화 수정 + R DCN
  재탐색). 이 상태로 아직 LB 제출 전.

## 8. 다음으로 고려할 것

- 위 커밋 상태 LB 제출 → 763대 회귀가 F 쪽 요인이었는지 최종 확인
- OOF 스태킹(메타러너로 4-way 가중치를 데이터 기반으로 학습) - R 전용,
  아직 미시도
- F는 당분간 동결(표본 작아 실험할수록 상쇄 패턴만 반복 확인됨) - R 위주로
  계속 진행하기로 합의됨

## 9. 새 환경 셋업 체크리스트

```bash
git clone <repo> && cd open
pip install lightgbm==4.7.0 xgboost==3.2.0 catboost==1.2.10
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu  # 서버와 버전 맞추기
# data/train.csv, data/test.csv는 대회 데이터 - 별도로 준비 필요(레포에 없음, .gitignore 대상)
python train.py   # 모델 재학습, walk-forward 진단 출력 확인
# 4번 섹션대로 submit.zip 빌드 + 스모크 테스트 후 제출
```
