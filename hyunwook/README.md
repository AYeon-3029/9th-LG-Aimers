# LG Aimers — 투구 제구 성공 예측 프로젝트 노트

`data_description.md`(공식 데이터 명세)와 `Sample.md`(피처 엔지니어링 아이디어 노트)를 교차 검토하여
정리한 문서입니다. Sample.md에 이미 있는 아이디어를 다시 나열하기보다,
**data_description.md 기준으로 봤을 때 Sample.md에 빠져 있거나 더 보강이 필요한 지점**,
그리고 **베이스라인 설계 관점에서 반드시 고려해야 할 점**을 중심으로 정리했습니다.

---

## 1. 문제 개요

- **태스크**: 투구 직전 상황(경기 정보, 카운트, 주자, 선수/팀, 과거 이력)으로
  해당 투구의 **제구 성공 확률**(`control_success`, 0/1)을 예측하는 이진 분류
- **평가지표**: Brier Skill Score — `100000 * max(0, 1 - brier / (r(1-r)))`
- **데이터**: `train.csv`(2019~2024, 147만 행) / `test.csv`(2025, 서버에서 24.5만 행 제공) /
  `trackman_history.csv`(2019~2024 투구 로그, 179만 행, 1:1 결합 아님) / `sample_submission.csv`
- **제출 방식**: 결과 CSV가 아니라 **코드(zip)**를 제출 — `model/`, `script.py`, `requirements.txt`
- **핵심 제약**: 평가 데이터는 행 단위 독립 예측만 허용 (test 내부 통계·target encoding·rolling 금지),
  `asof_*`는 이미 시점 안전하게 계산된 공식 피처라 사용 가능

---

## 2. Sample.md가 잘 다루고 있는 부분 (요약)

- 카운트(`balls_before`/`strikes_before`) → `count_state`, `pitcher_ahead`/`batter_ahead` 등 조합 피처
- 아웃카운트 × 주자 상황(`outs_before × base_state`) 조합
- 점수차 × 이닝, `close_game`/`blowout_game` 등 상황 중요도 피처
- 주자 상황 컬럼 간 중복성 및 비교 검증 전략
- `home_win_expectancy`/`away_win_expectancy` × 이닝·점수차 interaction
- `Li`(Leverage Index)를 압박 상황 대리 변수로 활용 + 투수별 압박 상황 대응력 피처화
- 투수/타자 좌우 매치업(`hand_matchup`)
- `asof_*` 피처군(표본 수 `n`의 중요성, success/reverse/middle rate 간 파생 변수,
  최근 폼(prev1/3/5), 타자 기준 누적, pitch mix)에 대한 상세한 해석

전반적으로 **"투구 직전 상황 → 심리적 압박/전략 → 제구 성공"이라는 야구적 인과 구조**를
잘 짚고 있습니다. 다만 아래 항목들은 data_description.md 기준으로 볼 때 보강이 필요합니다.

---

## 3. Sample.md 대비 변경/추가가 필요한 점

### 3.1 `trackman_history.csv` — 개인 단위 결합 불가 확인, 리그/구종군 사전지식으로 활용 방향 전환 ⚠️

`data_description.md`는 이 파일을 "참가자가 자유롭게 활용할 수 있는 179만 행짜리 로그"로
명시적으로 강조하고, `Baseline_Train` 노트북조차 "이 베이스라인에서는 사용하지 않으니 직접
활용해 보라"고 안내합니다. Sample.md에는 이 파일을 이용한 아이디어가 하나도 없어 처음에는
"투수별 구종 특성 요약값을 `asof_*`처럼 만들어 붙이자"는 방향을 검토했으나, **직접 데이터를
검증한 결과 개인 단위 결합이 불가능하다는 것을 확인했습니다.**

**검증으로 확인한 차단 요인 두 가지**:

1. **outcome 라벨 부재** — `trackman_history.csv`의 30개 컬럼에는 스트라이크/볼 판정, 코스
   좌표, `success`/`middle` 같은 결과 라벨이 전혀 없음. 있는 건 순수 물리 측정값
   (`rel_speed`, `spin_rate`, `induced_vert_break`, `horz_break`, `extension`, `rel_height`,
   `rel_side`, `zone_speed`)과 구종 태그(`tagged_pitch_type`, `pitch_type_group`)뿐이라,
   "이 투구가 제구에 성공했는가"를 이 파일만으로는 판단할 수 없음
2. **ID 매칭 불가** — `train.csv`의 `pitcher_id`(792명, 20700~24633 범위)와
   `trackman_history.csv`의 `pitcher_trackman_id`(906명, 50008~71775157 범위)를 직접 비교한
   결과 **교집합 0건**. 두 ID는 서로 다른 익명화 체계라 어떤 투수가 trackman에서 어떤
   구종을 던졌는지 연결할 공식 키가 없음

| 하고 싶은 것 | 가능 여부 |
|---|---|
| 특정 투수의 구종별 제구 성공률 | ❌ 불가능 (ID 매칭 불가 + success 라벨 없음) |
| 특정 투수의 구종별 구위 특성(회전수, 무브먼트) | ❌ 불가능 (ID 매칭 불가) |
| 구종군(`pitch_type_group`)별 리그 전체의 일반적인 경향(prior) | ✅ 가능 |

**설계 방향 (확정)**: 개인 단위 연결이 막혔으므로, `trackman_history.csv`는 **리그/구종군
단위의 일반 사전지식(prior)** 을 뽑는 용도로만 사용한다. 즉 `pitch_type_group`
(fastball/breaking/offspeed)별로 무브먼트 크기·구속·회전수 등이 제구 난이도와 통계적으로
어떤 관계를 갖는지 리그 전체 기준으로 먼저 파악해 **prior**로 깔아두고, 이 prior를
`asof_pitcher_*` 같은 개인별 통계를 계산할 때 **표본 수(`n`)에 따라 가중치를 다르게 섞는
shrinkage 방식**으로 결합한다.

```
개인 통계 최종값 = (n * 개인_rate + k * prior) / (n + k)
```

- `n`(표본 수)이 클수록 개인 통계 비중이 커지고, `n`이 작을수록(cold-start) prior(리그/구종군
  일반값) 비중이 커짐
- `k`는 prior를 얼마나 신뢰할지 정하는 감쇠 상수 — 검증 세트로 튜닝
- 이 방식은 3.2절의 cold-start 문제와 3.3절의 고카디널리티(신규 시즌 조합) 문제를
  **동시에 완화**하는 공통 해법이 됨 — `trackman_history.csv` 기반 prior + `asof_*` 개인
  통계를 shrinkage로 합치는 것이 이번 프로젝트의 핵심 피처 엔지니어링 전략

**주의**: `trackman_history.csv`는 `train.csv`/`test.csv`와 1:1 결합 테이블이 아니고,
"평가 시점 이후 정보를 포함하는 방식으로 사용 불가"라는 제약이 있으므로, prior 계산도
**시즌/시점 기준을 지켜 train 쪽 시점에 안전하게 앞서는 범위만** 사용해야 함.

### 3.2 Cold-start / 결측치 처리 전략이 구체화되어 있지 않음

`data_description.md`는 "표본 수가 0인 경우 일부 rate 컬럼은 결측값일 수 있으며, 이런
cold-start 상황의 결측 처리·smoothing·fallback 전략은 참가자가 자유롭게 설계할 수 있다"고
명시합니다. Sample.md는 "`n`과 함께 봐야 한다"는 정성적 언급까지만 하고, 구체적인
처리 방법은 다루지 않습니다.

**추가로 고려할 점**:
- 베이스라인(`SimpleImputer(strategy="median")`)은 단순 중앙값 대치 — `n`이 작을수록
  신뢰도가 낮다는 정보 자체가 사라짐
- 대안: 베이지안 shrinkage(`(n*rate + k*prior) / (n+k)`), `n` 구간별 fallback(예: `n<5`는
  팀/리그 평균으로 대체), 또는 `n` 자체를 가중치/피처로 명시적으로 남기는 방식
- 신인 투수·타자(과거 이력이 거의 없는 경우)에 대한 처리 전략을 별도로 검증할 필요

### 3.3 카디널리티(범주 수) 문제와 test 시점 미출현 범주 대응

Sample.md는 `pitcher_season`, `pitcher_id × batter_hand` 등 고카디널리티 조합 피처를
다수 제안하지만, 다음을 고려하지 않고 있습니다.
- `train`은 2019~2024, `test`(2025)는 학습에 없던 새 시즌 — `pitcher_season` 같은 조합은
  2025 시즌 값 자체가 학습 데이터에 전혀 없어 **사실상 전부 cold-start**가 됨
- 베이스라인의 `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`는
  처음 보는 범주를 전부 `-1`로 뭉개버리므로, 트리 모델이 이를 유의미하게 분리하기 어려움
- **고려할 점**: 고카디널리티 조합 피처는 "범주 자체"보다 "그 조합의 as-of 집계 통계치"로
  변환해서 넣는 편이 test 시점 일반화에 유리함 (Sample.md의 `asof_*` 아이디어와 결합 필요)

### 3.4 평가지표(Brier Skill Score) 관점의 고려사항이 없음

Sample.md는 피처의 "야구적 타당성"에 집중하고 있고, **확률 보정(calibration)**에 대한
언급이 없습니다. Brier score는 예측 확률의 절대적인 정확도(순위가 아니라 값 자체)에 민감한
지표입니다.

**추가로 고려할 점**:
- RandomForest의 `predict_proba`는 극단값(0, 1)에 잘 도달하지 못하거나 반대로 과신하는
  경향이 있어, `CalibratedClassifierCV`(Platt scaling / isotonic) 적용을 검토할 가치가 있음
- 피처 엔지니어링으로 AUC/로그로스가 좋아져도 Brier score가 항상 같이 좋아지진 않으므로,
  검증 시 Brier Skill Score를 직접 모니터링하며 피처를 추가/제거해야 함
- 클래스 불균형(제구 성공률 자체가 한쪽으로 치우쳐 있는 경우) 시 baseline `r(1-r)`가
  작아져 스코어 변동폭이 커질 수 있음 — 검증 세트 크기·분할 방식에 따라 스코어가 불안정할 수 있음

### 3.5 평가 데이터 누출(leakage) 규칙과 Sample.md 아이디어의 충돌 가능성 점검

`data_description.md` 5)~6)절은 "test.csv 내부 행 간 통계·target encoding·rolling 금지"를
명확히 규정합니다. Sample.md의 피처 아이디어 자체는 대부분 `asof_*`류(훈련 시점에 이미
계산된 안전한 피처)라 문제 없지만, **직접 구현 시 주의할 지점**을 명시적으로 남겨야 합니다.

- `pitcher_season`, `hand_matchup`, `count_state` 등 조합 피처의 통계치(성공률 등)를
  만들 때는 반드시 **train.csv 기준으로만 집계**하고, 그 매핑 테이블을 파이프라인/모델
  파일에 함께 저장해 test 시점에는 "조회"만 하도록 구현해야 함 (test 자체 통계 계산 금지)
- `Li × pitcher_id`처럼 "투수가 압박 상황에서 얼마나 흔들리는가"를 나타내는 피처도
  동일하게 train 기준 as-of 집계로 만들어야 하며, `trackman_history.csv`를 결합할 때도
  같은 원칙 적용 필요

### 3.6 다뤄지지 않은 컬럼

Sample.md에서 언급이 없거나 얕게만 다뤄진 원본 컬럼들:
- `pitcher_team_id` / `batter_team_id`: 팀 단위 효과(구장 특성, 팀 수비/포수 리드 등)는
  전혀 다뤄지지 않음 — 포수가 dataset에 없다는 점을 감안하면 `pitcher_team_id`가 그
  간접 프록시가 될 수 있음
- `game_type`: "categorical로 활용 가능" 언급만 있고 실제 가설(정규시즌 vs 포스트시즌
  등 긴장도 차이)은 없음
- `asof_pitcher_pitchmix_n`: pitch mix 표본 수 자체가 `asof_pitcher_n`처럼 신뢰도
  지표로 쓰일 수 있는데 별도로 다뤄지지 않음

---

## 4. 검증(Validation) 전략 관련 추가 고려사항

`Baseline_Train.ipynb`는 2024 시즌을 검증용으로 떼어내는 단일 시간 분할을 사용합니다.
Sample.md의 피처 아이디어들(특히 `pitcher_season`, as-of 파생 변수)을 검증할 때는:

- 단일 fold보다 **여러 시즌 경계로 나눈 시간 기반 K-fold**(예: 2022 검증/2023 검증/2024 검증)로
  일반화 성능을 다각도로 확인하는 것이 안전함 — 특정 시즌에만 과적합된 피처를 걸러낼 수 있음
- 최종 평가가 **2025 시즌(완전히 새로운 시즌)**이라는 점에서, 검증 시에도 "새 시즌"
  상황을 흉내내는 것이 중요 — `pitcher_season`류 피처의 cold-start 취약성을 검증
  단계에서부터 확인해야 함

---

## 5. 다음 단계 제안 (Action Items)

1. ✅ `trackman_history.csv`에서 구종군(`pitch_type_group`)별 리그 전체 prior 통계 산출
   (개인 단위 결합 불가 확인됨 — 3.1절 참고), `asof_pitcher_*`와 shrinkage로 결합하는
   파이프라인 설계 및 구현
2. ✅ `asof_*` 계열 cold-start(`n=0` 또는 작은 `n`) 처리 전략 실험
   (median 대치 vs 위 shrinkage 방식 vs 구간별 fallback) — shrinkage를 기본 방향으로 확정,
   실험 결과 유효성 확인됨 (6절 참고)
3. ✅ Sample.md의 조합 피처(`count_state`, `hand_matchup`, `outs_before × base_state` 등)
   18개 추가 — 전부 행 단위 연산이라 누출 없이 그대로 사용 가능함을 확인 (6.4절)
4. ✅ RandomForest 외 모델(LightGBM) 및 확률 보정(`CalibratedClassifierCV`) 비교 실험
   — LightGBM 채택(685.61), 확률 보정은 이번 데이터에서 역효과 확인 (6.8절)
5. ✅ 시간 기반 다중 fold(walk-forward) 검증으로 Brier Skill Score의 안정성 확인,
   시즌 추세 보정 피처(`season_trend_success_prior`) 추가 (6.5절) — 510.85 달성
6. ✅ `pitcher_team_id`/`game_type` 등 미탐색 컬럼에 대한 가설 수립 및 중요도 검증
   — `batter_team_id`를 범주형으로 바꿔 죽어있던 신호를 살림, 685.86 달성 (6.9절)
7. ✅ Inference 스크립트(`script.py`) 작성 — `features.py`의 `add_shrinkage_features`를
   그대로 재사용해 test.csv에도 동일한 피처를 재현 (6.7절)
8. 2025 시즌에 대한 구조적 단절(ABS 관련 추가 규정 변화 등) 리스크 대응 방안 검토
   — 선형 추세만으로는 못 잡는 한계 인지 (6.5절)
9. ✅ cold-start 추가 대응(prev1/3/5 최근 폼 shrink) + `max_features` 병목 발견 및
   해결 — 606.98 달성, 이전 피처 중요도 결론 일부 정정 (6.6절)
10. 대회 제출 환경의 시간 제한 확인 — `max_features=None`은 학습이 느림(~5분)

---

## 6. 진행 로그 — 실험 결과

베이스라인을 그대로 재현한 뒤, 설계한 shrinkage 전략을 실제로 적용해 효과를 검증했다.
관련 코드: `train_baseline.py`, `build_prior.py`, `features.py`, `train_v2.py`
(전부 프로젝트 루트, `.venv/`에 pandas/scikit-learn/joblib 설치해 실행).

### 6.1 1단계 — 베이스라인 재현 (`train_baseline.py`)

`Baseline_Train.ipynb`와 동일한 47피처 + RandomForest(depth=10) 그대로 재현.
2024 시즌을 검증 세트로 분리.

- 전체 제구 성공률: 52.38%
- **Validation Score (Brier Skill Score): 416.18** ← 이후 모든 개선의 비교 기준점

### 6.2 2단계 — `trackman_history.csv` 기반 구종군 prior 산출 (`build_prior.py`)

- **1차 시도(실패)**: 구종군별 무브먼트 "평균 크기"를 난이도 지수로 사용했더니
  `train.csv` 기반 실제 성공률과 방향이 맞지 않음 (breaking이 "가장 쉬움"으로
  잘못 나옴). 원인: 무브먼트 평균은 구종의 전형적 궤적 특성(패스트볼의 백스핀 상승 등)을
  반영할 뿐 "일관성"과는 다른 개념이었음.
- **2차 시도(부분 성공)**: 무브먼트 "표준편차"(변동성)로 다시 계산 → fastball
  (일관됨, 성공률 0.524)과 breaking(불규칙함, 성공률 0.493)은 방향이 맞았으나,
  offspeed는 여전히 안 맞음(변동성 큰데 성공률도 가장 높음, 0.577).
- **최종 결정**: trackman 기반 난이도 지수는 참고용으로만 `prior_table.json`에
  남기고, 실제 shrinkage 계산에는 **`train.csv` 기반 group prior만 사용**
  (통계적으로 실제 정답 라벨에 근거해 더 견고함).
- `train.csv`에서 WLS로 추정한 구종군별 prior 성공률:
  fastball 0.5244 / breaking 0.4930 / offspeed 0.5772
  (전체 평균 0.5315를 global prior로 별도 보관)

### 6.3 3단계 — 개인별 shrinkage 결합 (`features.py`, `train_v2.py`)

투수의 구종 사용 비율(`asof_pitcher_fastball_rate` 등)로 가중평균한 개인화 prior를
만들고, `(n * 개인_rate + k * prior) / (n + k)` (k=50) 공식으로 원본 `asof_pitcher_*`,
`asof_batter_*`의 success/middle rate에 shrinkage를 적용한 4개 신규 컬럼 추가.

| | Validation Score |
|---|---|
| 베이스라인 (원본 47피처) | 416.18 |
| **shrinkage 피처 4개 추가 (51피처)** | **440.02 (+23.84)** |

- `asof_pitcher_success_rate_shrunk`가 피처 중요도 **2위**(0.099)로, 원본
  `asof_pitcher_success_rate`(3위, 0.094)보다 중요하게 쓰임 — shrinkage가 유의미한
  신호를 추가했음을 확인.
- 결과 모델은 `./model/rf_v2.pkl`에 `{model, features, prior_table}` 형태로 저장 —
  추론 시 동일 구조를 그대로 불러와 재현.

### 6.4 4단계 — Sample.md 조합 피처 (`features.add_combo_features`)

`count_state`, `outs_base_state`, `hand_matchup`, 점수차 이진 플래그(`close_game` 등),
이닝×점수차/기대승률/Li interaction 등 18개 조합 피처 추가. 전부 행 단위 연산이라
누출 위험 없음.

| | Validation Score |
|---|---|
| 3단계까지 (shrinkage) | 440.02 |
| **combo 18개 추가 (69피처)** | **463.65 (+23.63)** |

- `hand_matchup`이 combo 중 기여도 1위(전체 9위, 0.043) — Sample.md 가설(투수·타자
  좌우 매치업이 중요)이 실제로 검증됨.
- **한 번 시행착오**: 개별 기여도 0.002 미만인 11개(점수차 이진 플래그류)를 제거해봤더니
  오히려 점수가 446.96으로 떨어짐(-16.69). RandomForest는 분기마다 피처 일부만
  무작위로 후보로 뽑기 때문에, 전체 피처 수가 줄면 개별 중요도가 낮던 피처들의
  미세한 상호작용 효과까지 함께 사라진 것으로 추정 — "개별 중요도 낮음 = 제거해도
  안전"이라는 가정이 이번 데이터에서는 성립하지 않음을 확인. 18개 전부 유지로 되돌림.

### 6.5 5단계 — 시간 기반 다중 fold 검증 (`train_v3_cv.py`) + 시즌 추세 보정 피처

**1차 시도 (실패, 폐기)**: `season == val_season`만으로 나머지를 전부 학습에 사용하는
방식으로 2022/2023/2024를 검증했더니 2022=1810.72, 2023=0.00, 2024=463.65로 편차가
극심했음. 원인은 예를 들어 val=2022일 때 학습 데이터에 2023~2024(미래 시즌)까지
섞여 들어가는 설계 결함 — 실제 제출 시나리오(과거로만 학습 → 미래 예측)와 인과
구조가 달라 신뢰할 수 없는 결과였음.

**수정**: 검증 시즌보다 앞선 시즌만 학습에 쓰는 **walk-forward(확장 윈도우)** 방식으로
재설계 (`train_v3_cv.py`). 그런데도 val=2023이 여전히 0점으로 나옴 → 원인 조사.

**발견 — 시즌별 성공률이 완만한 하락이 아니라 계단식으로 변함**:

| 시즌 | 성공률 | 전년 대비 |
|---|---|---|
| 2019 | 56.47% | — |
| 2020 | 53.27% | -3.2%p |
| 2021 | 53.28% | +0.0%p |
| 2022 | 52.89% | -0.4%p |
| 2023 | 49.996% | -2.9%p |
| 2024 | 48.61% | -1.4%p |

2020~2022는 거의 평평하다가 2023에 급락. `val=2023` fold는 이 급락 이전 데이터
(2019~2022)만으로 추세를 fit했기 때문에 전혀 예측하지 못했음. **사용자 확인 사실**:
KBO는 2024시즌에 로봇 심판(ABS, Automatic Ball-Strike System)을 최초 도입함 —
심판의 주관적 판정 대신 고정된 기계 판정으로 스트라이크존이 바뀌면서 `control_success`
정의 자체에 영향을 줬을 가능성이 높음. 즉 이건 노이즈가 아니라 **실제 룰 변화로
인한 구조적 단절(structural break)**이며, 단순 선형 추세로는 이런 단절을
사전에 예측할 수 없다는 근본적 한계가 있음.

**RandomForest의 한계**: 트리 모델은 학습 때 본 적 없는 `season` 값으로 선형
추세를 외삽하지 못한다. 그래서 추세를 트리 밖에서 선형회귀로 미리 계산해
"이 시즌의 예상 성공률"을 피처(`season_trend_success_prior`)로 만들어 넣었음
(`features.fit_season_trend` + `add_season_trend_feature`).

| | Validation Score |
|---|---|
| 4단계까지 (shrinkage+combo) | 463.65 |
| **시즌 추세 피처 추가 (70피처)** | **510.85 (+47.20)** |

- `season_trend_success_prior` 피처 중요도 전체 10위(0.035)로 유의미하게 반영됨.
- walk-forward CV 결과(추세 피처 포함): val=2021 1219.51 / val=2022 2104.74 /
  val=2023 0.00 / val=2024 510.85. **val=2023이 여전히 0인 것은 모델 결함이
  아니라 설계상 당연한 결과** — 그 시점엔 존재하지 않았던 정보(2023년의 ABS발
  급락)를 요구하는 상황이었기 때문. 반면 실제 제출 모델(`train_v3.py`)은
  2019~2024 전체로 추세를 fit하므로 2023·2024의 급락을 이미 관측한 뒤
  2025(47.47% 예상)를 외삽함 — 이 fold의 실패를 그대로 안고 있지 않음.
- **남은 리스크**: 2025에 또 다른 구조적 단절(예: ABS 판정 기준 추가 조정,
  새로운 규정 변화)이 있다면 선형 추세로는 잡아낼 수 없음. 이는 모델 설계로
  해소할 수 있는 문제가 아니라 근본적인 한계로 남겨둠.

### 6.6 cold-start 추가 대응 + `max_features` 병목 발견 (최종 최고 기록)

`asof_pitcher_n < 30`은 1.58%에 불과해 표본 자체는 작았지만, **`prev1/3/5_game` 최근 폼
지표의 결측이 1.98%(약 2.9만 행)로 더 컸다.** 이 값이 결측이면 지금까지는 단순 median
대치만 되고 있었음. 리그 prior가 아니라 "그 투수의 (이미 shrink된) 커리어 통산
성공률/middle률"로 fallback하도록 `prev1/3/5_game_*_shrunk` 6개, 그리고
`ball/strike/reverse_rate_shrunk`·cold-start 플래그 3개를 추가로 설계
(`features.add_shrinkage_features` 확장).

**1차 결과 (실패로 보임)**: 82피처로 늘려 재학습했더니 510.85 → 498.99로 **하락**
(-11.86). `hand_matchup`(0.043→0.0003), `count_state`(0.015→0.00004) 등 4단계에서
잘 작동하던 조합 피처의 중요도가 붕괴함. prev1/3/5 shrunk 6개만 남기고 나머지를
제거해도 502.24로 여전히 하락.

**원인 발견 — `RandomForestClassifier(max_features='sqrt')` 기본값이 병목이었음**:
분기마다 후보로 뽑는 피처 수가 `sqrt(전체 피처 수)`(76피처 기준 약 9개)뿐이라,
피처를 추가할수록 기존에 잘 작동하던 피처들이 서로 후보 자리를 빼앗는 현상이었다.
`max_features`를 바꿔가며 실험(76피처 기준):

| `max_features` | 분기당 후보 수 | Validation Score | 학습 시간 |
|---|---|---|---|
| `sqrt`(기본값) | ~9개 | 502.24 | 34.9s |
| 0.3 | ~23개 | 591.83 | 77.8s |
| 0.5 | ~38개 | 602.34 | 127.4s |
| **`None`(전체)** | 76개 전부 | **606.58** | 193.7s |

**cold-start 확장 16개 + `max_features=None`을 함께 적용한 최종 결과**:

| | Validation Score |
|---|---|
| 5단계까지 (70피처, `max_features='sqrt'`) | 510.85 |
| **cold-start 16개, `max_features=None` 적용 (82피처)** | **606.98 (+96.13)** |
| 베이스라인(1단계) 대비 | **+190.80** |

**정정 사항**: `max_features=None`에서는 `hand_matchup`(0.000005), `count_state`
(0.000000)의 중요도가 완전히 사라졌다. 이는 4단계에서 내렸던 "hand_matchup이
중요하다"는 결론이 **`max_features='sqrt'`라는 하이퍼파라미터 설정의 착시**였음을
의미한다 — 후보가 적을 때는 약한 피처도 상대적으로 자주 뽑히지만, 전체 피처를
검토하면 더 강한 신호(`asof_pitcher_prev1/3_game_middle_rate_shrunk`,
`season_trend_success_prior`, `balls_before` 등)에 항상 밀린다. Sample.md
조합 피처들이 "쓸모없다"는 뜻은 아니고, 지금 있는 다른 피처들과 정보가 겹쳐서
한계 기여가 거의 없다는 뜻 — 제거해도 성능에 해는 없지만 굳이 뺄 이유도 없어
그대로 유지.

**교훈**: 피처 엔지니어링 실험(4~6단계)의 결론 상당수가 사실 `max_features`
하이퍼파라미터에 의해 왜곡되어 있었다. 이후 피처 추가/제거 실험은 반드시
`max_features=None`(또는 충분히 큰 값) 기준으로 재검증해야 함.

⚠️ 트레이드오프: `max_features=None`은 학습 시간이 길다(1회 fit 약 233초, 전체
재학습까지 포함하면 약 5분). 대회 제출 환경의 시간 제한을 반드시 확인해야 함.

### 6.7 실제 제출용 `script.py` 작성 (`submission/`)

원본 `open/baseline_submit/`(대회 배포 참고자료)은 그대로 보존하고, 실제 제출용
폴더 `submission/`을 새로 만들어 아래 구조로 구성했다.

```
submission/
├── model/rf_v3.pkl        # {model, features, prior_table} 딕셔너리 통째로 저장됨
├── features.py             # shrinkage/combo/season-trend 피처 엔지니어링 (학습·추론 공용)
├── script.py                # 추론 스크립트
├── requirements.txt          # numpy 추가 (features.py가 사용)
└── data/                     # test.csv, sample_submission.csv (형식 확인용 샘플)
```

핵심 변경점:
- `rf_v3.pkl`이 `{"model", "features", "prior_table"}` 딕셔너리로 저장되어 있어서
  (`train_v3.py` 참고), **별도 `prior_table.json` 파일 없이 모델 파일 하나로 추론**이
  가능하도록 설계 — prior 계산 로직과 모델이 항상 짝을 맞춰 배포되어 버전 불일치
  위험이 없음
- `build_features()`가 `features.py`의 `add_shrinkage_features` →
  `add_combo_features` → `add_season_trend_feature`를 학습 때와 동일한 순서로
  적용한 뒤, 저장해 둔 `feature_names`(82개) 순서 그대로 선택
- 원본 베이스라인 `script.py`는 47개 원본 컬럼만 쓰는 걸 전제하고 있어서, 우리
  모델(82피처)을 그대로 태우면 컬럼 누락으로 실패했을 것 — 이번 작업으로 해결

**검증**: `test.csv` 5행 샘플로 `python3 script.py` 실행 → `output/submission.csv`
5행 정상 생성 확인 (확률값 0.40~0.51 범위, 0~1 사이 정상 분포).

```
row_id,control_success
TEST_000001,0.4434698469960285
TEST_000017,0.40512243994673525
TEST_000213,0.48118253675633677
TEST_005332,0.5085350063655147
TEST_035185,0.5106576937637819
```

**남은 작업**: 실제 제출 시 `submission/` 폴더를 zip으로 묶어야 함. `data/`는
평가 서버가 실제 데이터로 교체하므로 제출 zip에는 빼도 되는지 대회 규칙 재확인 필요.

### 6.8 모델 비교 (RandomForest → LightGBM) + 확률 보정 실험 (최종 최고 기록)

동일한 82피처, 동일한 전처리(`OrdinalEncoder`+`SimpleImputer`)로 분류기만
RandomForest → LightGBM으로 교체해 순수하게 모델 자체의 차이를 비교했다
(`train_v4_lgbm.py`).

**1차 시도 (실패)**: LightGBM 기본값에 가까운 설정
(`n_estimators=500, lr=0.05, min_child_samples=200, subsample=0.8`)으로는
546.38 — RandomForest(606.98)보다 낮았음.

**2차 시도 (성공)**: `n_estimators=3000, learning_rate=0.02, num_leaves=63,
min_child_samples=50, subsample=1.0`으로 늘리고 2024 검증셋으로 early stopping
적용(`best_iteration=211`) → **685.61**, RandomForest 대비 **+78.63**. 학습
시간도 18초로 RandomForest(233초)보다 훨씬 빠름.

<table header-row="true">
	<tr>
		<td></td>
		<td>Validation Score</td>
		<td>학습 시간</td>
	</tr>
	<tr>
		<td>RandomForest (max_features=None, 82피처)</td>
		<td>606.98</td>
		<td>233s</td>
	</tr>
	<tr>
		<td>LightGBM (튜닝 전)</td>
		<td>546.38</td>
		<td>24s</td>
	</tr>
	<tr>
		<td>**LightGBM (튜닝, early stopping)**</td>
		<td>**685.61 (+78.63)**</td>
		<td>**18s**</td>
	</tr>
</table>

**확률 보정(`CalibratedClassifierCV`) 실험 — 두 갈래 결과, 원인 규명**:

- 1차: 2024 검증셋을 calib/eval 절반씩 나눠 테스트 → sigmoid 보정 +67.99,
  isotonic +64.25 개선. 얼핏 보정이 효과적으로 보임.
- 2차(다른 단계와 동일한 기준으로 재검증): 다른 모든 실험과 똑같이
  "2019~2023만으로 학습"한 뒤 `CalibratedClassifierCV(cv=5)`로 보정, 2024로
  평가 → **sigmoid/isotonic 둘 다 Score 0.00으로 급락**.
- **원인**: 1차 실험은 보정을 2024 데이터 일부로 학습했기 때문에 "2024의 실제
  분포(성공률 48.6%)"를 직접 반영할 수 있었던 것. 2차 실험처럼 보정도
  2019~2023(과거의 더 높은 성공률 분포)만으로 학습하면, 이미 존재하는
  `season_trend_success_prior` 피처가 하던 일을 보정기가 구식 데이터로 다시
  하려다 실패하는 셈 — **확률 보정의 일반적 효과가 아니라 시즌 분포 이동을
  얼마나 최신 데이터로 반영했는지의 문제**였음이 드러남.
- **결론**: 이번 프로젝트에서는 확률 보정을 추가하지 않는다 — 이미
  `season_trend_success_prior` 피처가 같은 문제를 해결하고 있고, 보정을
  잘못된 시점 데이터로 학습하면 오히려 해가 됨.

**최종 채택 모델**: LightGBM(튜닝, 보정 없음) = **685.61**
(베이스라인 대비 **+269.43**). 전체 데이터(2019~2024)로 재학습해
`./model/lgbm_v1.pkl`에 저장, `submission/`의 모델도 이걸로 교체하고
`test.csv` 5행 샘플로 재검증 완료.

### 6.9 Action Item 6 — `pitcher_team_id`/`game_type` 점검 (최종 최고 기록)

**`game_type` 검증**: 실제 데이터를 보니 `R`(정규시즌, 131만 행)과 `F`(포스트시즌
추정, 16만 행)의 성공률 차이가 **8.9%p**(51.4% vs 60.3%)로 뚜렷했다. 이미
`CAT_COLS`에 포함돼 정상적으로 인코딩되고 있었고, 중요도도 순위 14~45위권으로
꾸준히 반영되고 있었음 — Sample.md가 언급만 하고 검증 안 했던 가설이 실제로
맞았음을 확인.

**`pitcher_team_id`/`batter_team_id` 검증**: 두 컬럼 다 원래 `CAT_COLS`에
없어서 **수치형으로 취급**되고 있었다는 걸 발견. 팀 ID는 크기 비교가 무의미한
명목형 값인데 숫자로 다뤄지면 트리가 "team_id ≤ 15.5" 같은 무의미한 기준으로만
분기할 수 있음. 실제로 확인해보니 `batter_team_id` 중요도가 **완전히 0**(전체
82개 중 77위)이었음.

범주형으로 바꿔서 재학습한 결과:

<table header-row="true">
	<tr>
		<td></td>
		<td>순위/중요도</td>
	</tr>
	<tr>
		<td>batter_team_id (수치형 취급 시)</td>
		<td>77위 / 0</td>
	</tr>
	<tr>
		<td>**batter_team_id (범주형 전환 후)**</td>
		<td>**7위 / 396**</td>
	</tr>
	<tr>
		<td>pitcher_team_id (범주형 전환 후)</td>
		<td>41위 / 137</td>
	</tr>
</table>

<table header-row="true">
	<tr>
		<td></td>
		<td>Validation Score</td>
	</tr>
	<tr>
		<td>6.8절까지 (팀ID 수치형)</td>
		<td>685.61</td>
	</tr>
	<tr>
		<td>**팀ID 범주형 전환 추가**</td>
		<td>**685.86 (+0.25)**</td>
	</tr>
</table>

점수 개선폭 자체는 작지만(+0.25), **완전히 죽어있던 신호(`batter_team_id`)를
되살렸다는 점**이 핵심 — 아마 팀별 구장 특성이나 수비/포수 리드 차이 같은 것을
반영하는 것으로 추정된다(3.6절 가설과 일치). `train_v4_lgbm.py`의 `CAT_COLS`에
반영해 최종 모델(`./model/lgbm_v1.pkl`, `submission/`)에 적용 완료.

**최종 최고 기록(당시): 685.86 (베이스라인 대비 +269.68)**

### 6.10 리더보드 제출 결과 + 2차 개선 라운드

**첫 제출 결과**: 베이스라인 제출 549.51 vs 우리 모델(685.86 버전) **758.37**
(1등 1156.70, 격차 398점/약 52%). 추론 시간은 3~4초로 `max_features=None`의
느린 학습 시간(item 10 우려)은 제출 채점과 무관함을 확인 — 채점 서버는
`script.py`(추론만) 실행, 학습은 로컬에서만 하면 됨.

**시도했으나 실패한 것들** (정직하게 기록):

- **CatBoost**(네이티브 범주형 처리): 639.22 — LightGBM(685.61)보다 오히려
  나쁨, 학습도 220초로 훨씬 느림. 미채택.
- **RandomForest + LightGBM 앙상블**: RF 단독 610.69로 LightGBM(708.12)보다
  훨씬 약해서, 어떤 가중치로 섞어도(0.3~0.7) LightGBM 단독보다 낮음. 미채택.
- **투수×타자 매치업 피처**(pitcher_id, batter_id 쌍별 성공률): 처음엔
  전체 train으로 만든 테이블을 학습 데이터에 그대로 적용해 self-leakage
  발생(Score 0.00, n=1인 쌍은 정답이 그대로 노출). Leave-one-out으로 고쳐도
  -213.28(477.24), `k`를 1000까지 올려도(사실상 신호 제거) 여전히 -1.50.
  **원인**: 투구 순서/시각 데이터가 없어 완벽한 walk-forward 계산이 불가능 —
  LOO도 학습 구간 내 "미래" 매치업이 과거 행에 살짝 새는 걸 못 막음. 미채택.
- **투수 단위 Li 가중 "클러치" 성공률**(매치업과 같은 LOO 방식, 표본은
  중앙값 855구로 매치업보다 훨씬 큼): 그런데도 동일하게 파국적 실패
  (Score 0.00, best_iteration 81로 조기 붕괴) — 표본 크기 문제가 아니라
  **LOO(학습)와 lookup(검증)의 인코딩 방식 차이 자체가 GBM에 악용되는
  구조적 문제**로 추정. 미채택.
- **`inning × home/away_win_expectancy`**(diff 버전과 별개로 개별 추가):
  700.96, -7.16 — 노이즈 수준이라 확실한 이득 없음. 미채택.
- **`asof_pitcher_*_shrunk × asof_batter_*_shrunk` 곱/차이**: 660.01,
  -48.11 — 이미 있는 피처들의 결정론적 조합이라 트리가 스플릿으로 재현
  가능한 정보였고, 중복이 오히려 분기 선택에 방해가 된 것으로 추정. 미채택.

**성공한 것들**:

<table header-row="true">
	<tr>
		<td>변경</td>
		<td>Validation Score</td>
	</tr>
	<tr>
		<td>6.9절까지 (팀ID 범주형, k=50)</td>
		<td>685.86</td>
	</tr>
	<tr>
		<td>+ pitcher_id/batter_id 범주형 전환</td>
		<td>690.52 (+4.66)</td>
	</tr>
	<tr>
		<td>+ 회귀(MSE) 목적함수 (참고용, 최종 미채택)</td>
		<td>691.98 (+1.46)</td>
	</tr>
	<tr>
		<td>+ shrinkage k 튜닝 (50 → 85)</td>
		<td>708.12 (+17.60)</td>
	</tr>
	<tr>
		<td>**pitcher_season, pitcher_batter_hand, Li interaction 5개 추가**</td>
		<td>**710.00 (+1.88)**</td>
	</tr>
</table>

(회귀 목적함수는 소폭 개선이지만 파이프라인 복잡도 대비 이득이 작아 최종
모델에는 분류(classification) 유지 — 필요시 재검토 가능)

**교훈**: 데이터에 정확한 시퀀스/타임스탬프가 없는 상태에서 "쌍(pair) 단위
과거 이력"을 직접 재구성하려는 시도는 표본 크기와 무관하게 거의 항상
실패한다 — 공식 `asof_*` 컬럼처럼 이미 안전하게 계산된 것을 활용하는 게
안전하고, 우리가 직접 walk-forward를 흉내 내려는 시도는 신중해야 함.

**현재 최종 최고 기록: 710.00 (로컬 검증) / 리더보드 재제출 대기 중**
(리더보드 1등 1156.70과의 격차는 이번 라운드로 크게 좁혀지지 않음 — 격차를
메우려면 더 근본적으로 다른 접근이 필요해 보임: 정교한 하이퍼파라미터
탐색(Optuna), 더 강력한 feature selection, 또는 전혀 다른 모델링 전략 검토)

### 6.11 두 번째 제출 — 로컬 개선이 실제로는 악화됨 (중요한 반성)

710.00 버전(k=85 + pitcher_season 등)을 제출한 결과: **실제 리더보드
753.41**, 첫 제출(758.37)보다 오히려 **-4.96 하락.** 로컬 검증은
685.86→710.00(+24.14)로 올랐는데 실제론 반대로 갔다 — **단일 2024 검증
분할에 과적합된 하이퍼파라미터/피처 선택**이었다는 뜻.

가장 유력한 범인: `k` 그리드 서치 자체가 인접 값끼리 ±7~15로 들쭉날쭉했는데
그중 우연히 가장 높게 나온 `k=85` 하나를 그대로 채택한 것 — 전형적인
"노이즈 중 최댓값을 진짜 신호로 착각" 패턴. `pitcher_season`도 2025엔
전부 미확인 범주(-1)로 뭉개지는데 로컬에서 +가 나온 것 자체가 수상했음.

**대응**: `k`를 50으로 되돌리고, `pitcher_season`/`pitcher_batter_hand`/Li
interaction 5개를 전부 제거. `pitcher_id`/`batter_id` 범주형 전환만 유지
(팀ID 때와 같은 원리로 구조적으로 타당하고, 첫 제출에서 이미 검증된
팀ID 방식의 연장선이라 상대적으로 신뢰도 높음) → 로컬 690.52로 복귀.
`submission/`도 이 버전으로 교체, 클린룸 재검증 완료.

**핵심 교훈**: 단일 시즌 검증 분할(2024)만으로 하이퍼파라미터를 고르는 건
위험하다 — 특히 그리드 서치처럼 여러 후보 중 "최댓값"을 고르는 방식은
노이즈를 주울 위험이 크다. 이후로는 다중 fold(walk-forward) 평균으로
검증해야 하며, 로컬 검증 개선이 실제 리더보드 개선을 보장하지 않는다는
점을 계속 염두에 둬야 함.

### 6.12 세 번째 제출 — 로컬/실제 순위가 정확히 역전되는 패턴 확인, 1차로 완전 롤백

`pitcher_id`/`batter_id` 범주형 전환만 유지한 "절충안"(로컬 690.52)을
제출한 결과: **실제 리더보드 755.91.** 세 번의 제출을 정리하면:

<table header-row="true">
	<tr>
		<td>제출</td>
		<td>로컬 검증</td>
		<td>실제 리더보드</td>
	</tr>
	<tr>
		<td>1차 (팀ID만 범주형, k=50)</td>
		<td>685.86</td>
		<td>**758.37**</td>
	</tr>
	<tr>
		<td>2차 (+k=85, pitcher_season 등)</td>
		<td>710.00</td>
		<td>753.41</td>
	</tr>
	<tr>
		<td>3차 (+선수ID 범주형, k=50)</td>
		<td>690.52</td>
		<td>755.91</td>
	</tr>
</table>

**로컬 순위(685.86 < 690.52 < 710.00)와 실제 순위(758.37 > 755.91 >
753.41)가 정확히 뒤집혀 있다.** 우리가 "개선"이라고 판단해 추가한 모든
변경이 실제로는 전부 역효과였다 — `pitcher_id`/`batter_id` 범주형 전환처럼
`pitcher_team_id`/`batter_team_id`와 같은 원리라 믿었던 것도 예외가 아니었음.

**대응**: 1차 제출 구성(팀ID만 범주형, `k=50`, 추가 interaction 없음)으로
완전히 롤백. `train_v4_lgbm.py`의 `BASE_CAT_COLS`에서 `pitcher_id`/
`batter_id` 제거, 재학습 결과 685.86 정확히 재현 확인. `submission/` 교체,
클린룸 검증 결과 첫 제출 때와 출력값 100% 동일함을 확인.

**중요한 결론**: 지금까지 실제 리더보드에서 검증된 가장 좋은 구성은
**1차 제출(685.86 로컬 / 758.37 실제)**이며, 이후의 모든 로컬 기반 튜닝은
근거 없이 실제 성능을 깎아왔다. 앞으로 추가 개선을 시도하려면:
1. 다중 fold(walk-forward) 평균 검증으로 노이즈에 강한 결정을 내리거나
2. 매 변경마다 실제 제출로 검증하되, 한 번에 하나씩만 바꿔서 인과관계를
   명확히 하거나
3. 확신이 서지 않는 변경은 아예 시도하지 않는 것이 안전함이 이번 세 번의
   제출로 뚜렷하게 확인됨.

### 6.13 미탐색 컬럼 추가 라운드 — 전부 기각 (685.86 유지)

3차 제출 롤백 이후, 3.6절에서 짚었던 미탐색 컬럼들을 하나씩 점검했다.
전부 채택 기준(명확한 개선 + 0이 아닌 피처 중요도)을 통과하지 못해
**전부 기각**했고, 현재 최고 기록은 여전히 685.86(로컬)/758.37(실제)이다.

- **`game_dayofweek`**: 화~일요일 성공률은 52.2~52.5%로 거의 동일해 "선발
  로테이션 간접 반영" 가설을 뒷받침하지 못함. 월요일만 49.97%로 낮은데
  경기 수도 6% 수준(KBO는 보통 월요일 휴식일이라 우천 보충경기 등 예외적
  경기로 추정). `is_monday` 플래그를 추가해도 점수 변화 0.00, 중요도 0 —
  원본 숫자 컬럼(현재 55위)이 이미 완전히 포착 중.
- **`score_diff_home`**: `top_bottom` × `score_diff_pitcher_team`으로
  100% 재구성되는 수학적으로 중복인 값임을 확인(T행은 완전히 동일, B행은
  부호만 반대). `home_team_leading`/`trailing`/`×inning` 추가 시 682.07로
  오히려 하락(-3.79), 신규 피처 중요도도 0에 가까움.
- **`runner_on_1b/2b/3b` 개별 조합 (병살타 유도 가설)**: DP 가능 상황
  (`runner_on_1b`ㆍ`outs_before<=1`)과 아닌 상황의 성공률 차이는 0.22%p로
  미미. 피처로 추가해도(`dp_situation`, `jam_situation`,
  `dp_situation×hand_matchup`) 중요도가 사실상 0인데 점수만 +2.75로 흔들려
  이전 사례들과 같은 "early stopping 지점 차이로 인한 노이즈"로 판단, 기각.
- **투구수/피로 가설**: `train.csv`/`test.csv`엔 경기 내 투구수 컬럼이
  없음(`asof_pitcher_n`은 커리어 누적). 대리 지표 `inning`으로 보면 1회
  53.2% → 9회 51.5% → 연장 50~51%로 뚜렷한 하락 확인, `trackman_history.csv`
  (개인 매칭 불가, 리그 전체 통계로만)에서도 투구 1~10구(137.5km/h) →
  81구+(135.0~135.2km/h)로 구속이 실제로 떨어지는 물리적 증거 확인 —
  가설 자체는 **맞음**. 다만 `inning`이 이미 원본 피처로 13위에 올라 있어
  잘 포착되고 있었음. `inning × asof_pitcher_success/middle_rate_shrunk`
  interaction을 추가해봤으나 680.71로 하락(-5.15) — 이번엔 신규 피처
  중요도가 199~232로 낮지 않았는데도 전체 성능은 떨어져, "중요도가 있어도
  일반화가 안 될 수 있다"는 걸 다시 확인.

**패턴**: 이미 강한 두 피처(`inning`, `asof_pitcher_*_shrunk` 등)를 곱해서
새 interaction을 만드는 시도가 이번 프로젝트에서 계속 실패하고 있다
(매치업, 클러치, `asof` 곱/차이, 이번 `inning×pitcher` 모두 마이너스).
이후 유사한 시도는 우선순위를 낮추는 게 합리적으로 보임.

### 6.14 다중 fold(walk-forward) 재검증 — 로컬 신호는 확인되지만 실제 전이는 보장 안 됨

3차 제출 롤백의 근거였던 두 변경(`k=85`, `pitcher_id`/`batter_id` 범주형
전환)이 단일 2024 분할만의 노이즈였는지 확인하기 위해, `val_season ∈
{2022, 2023, 2024}`로 walk-forward 다중 fold 재검증(`train_v6_multifold_check.py`)을
돌렸다. 2023은 이미 알려진 대로(6.5절, ABS 단절) 전 구성에서 0점에 가깝게
나오는 병적인 fold라 판단 기준(2022/2024 평균)에서 제외.

| 구성 | 2022 | 2024 | 평균 |
|---|---|---|---|
| A. 팀ID만 범주형, k=50 (현재 채택) | 2261.64 | 685.86 | 1473.75 |
| B. +선수ID 범주형, k=50 (3차 제출) | 2266.23 | 690.52 | 1478.38 |
| C. 팀ID만 범주형, k=85 | 2267.86 | 697.26 | 1482.56 |
| D. +선수ID 범주형, k=85 | 2267.01 | 708.12 | 1487.57 |

B/C/D 모두 2022·2024 두 fold에서 일관되게 A보다 좋게 나와, **단일 분할의
순수 노이즈는 아니고 2019~2024 안에서는 재현되는 신호**임을 확인했다.

**그런데 B(선수ID 범주형+k=50)는 이미 3차 제출로 실제 검증까지 끝난 구성으로,
실제 리더보드에서는 755.91로 A(758.37)보다 낮았다.** 즉 다중 fold로
일관되게 확인된 개선조차 실제 2025 리더보드로는 전이되지 않았다 —
**다중 fold 검증도 이 문제에서는 "노이즈 여부"만 걸러줄 뿐, "2025로의
전이 가능성"까지는 보장하지 못한다는 결론**. 이후 실험은 이 한계를
전제로 진행.

### 6.15 XGBoost 비교 — 채택 안 함

`train_v7_xgboost.py`: LightGBM과 동일한 82피처(팀ID만 범주형, k=50)로
전처리를 그대로 두고 분류기만 XGBoost로 교체, 2024 홀드아웃에서 비교.

| 설정 | Score |
|---|---|
| LightGBM (현재 채택) | 685.86 |
| XGBoost, leaf-wise(leaves=63), depth-wise(depth=6/8), lr 0.02/0.05 등 5가지 설정 | 590.70 ~ 657.00 (전부 하락) |

CatBoost(6.11절)에 이어 XGBoost도 LightGBM을 이기지 못함 — 이 데이터셋
(shrinkage/비율 기반 수치형 피처 위주, 122만 행)에서는 LightGBM의 leaf-wise
성장 방식이 유독 잘 맞는 것으로 보임. 모델 라이브러리 교체 방향은 종료.

### 6.16 Optuna 정규화 탐색 + 4차 제출 — 다중 fold 개선이 다시 한번 실제로는 전이 안 됨

`train_v8_optuna.py`: `reg_alpha`/`reg_lambda`/`min_split_gain`을 30 trial
TPE 탐색(목적함수 = 6.14의 다중 fold 2022/2024 평균, 단일 분할 노이즈 회피
목적). 최적(trial 28, `reg_alpha=1.047, reg_lambda=0.029,
min_split_gain=0.219`)이 기준선 1473.75 → 1481.65(+7.89), 단일 2024
분할에서도 685.86 → 691.43(+5.57)로 방향이 일관돼 `train_v9_regularized.py`로
전체 데이터 재학습 후 **4차 제출**했다.

**결과: 756점 — A(758.37)보다 낮음.** 6.14절의 B와 완전히 같은 패턴이다:
다중 fold로 확인한 개선(+7.89)이 실제로는 전이되지 않았다. 이걸로
"피처 추가 / 범주형 인코딩 / k 튜닝 / 정규화" 네 가지 서로 다른 종류의
변경이 전부 같은 방향으로 어긋난 셈이라, 우연이 아니라 **이 문제 자체가
로컬 검증(단일 분할이든 다중 fold든)으로는 2025 실제 성능을 예측하기
어려운 구조적 특성을 갖는다**는 결론에 무게가 실린다. 제출용 모델은
정규화 없는 원본(A) 구성으로 즉시 복구.

### 6.17 Recency weighting 시도 — 명확히 기각 (실제 제출 없이 로컬에서 기각)

6.16까지의 네 차례 실제 제출이 전부 "2019~2024 안에서 더 잘 맞추는" 방향의
변경이었다는 공통점에 착안해, 반대로 "무엇을 배울지" 자체를 바꿔보는
시도로 recency weighting을 테스트했다(`train_v10_recency.py`).
`sample_weight = decay ** (val_season - season)`로 오래된 시즌(특히
2019~2020, ABS 도입 전)의 영향력을 줄이는 방식.

| decay | 2022 | 2024 | 평균 |
|---|---|---|---|
| 1.0 (가중치 없음, 현재 A) | 2261.64 | 685.86 | 1473.75 |
| 0.9 | 2260.40 | 672.79 | 1466.59 |
| 0.7 | 2251.18 | 669.46 | 1460.32 |
| 0.5 | 2227.70 | 607.08 | 1417.39 |
| 0.3 | 2206.18 | 623.58 | 1414.88 |

decay를 낮출수록(최근 시즌에 집중할수록) 2022·2024 두 fold 모두 **단조롭게
악화** — 이전 시도들과 달리 로컬 검증 단계에서부터 명확히 나빠서 실제
제출 없이 기각. `asof_*`/shrinkage 피처가 이미 "그 시점까지의 누적 이력"을
통계적으로 처리하고 있어서, 오래된 시즌도 낡은 데이터가 아니라 유효한
학습 표본이었던 것으로 보인다 — sample_weight로 죽이면 실질 표본 수만
줄어드는 효과가 더 컸다. 같은 논리로 오래된 시즌을 아예 드롭하는 방향도
추가 시도 없이 낮은 우선순위로 판단.

### 6.18 🎉 CatBoost 전환 — 실제 리더보드 902.04점 (5차 제출, 현재 최고)

`train_v14_blend.py`에서 LightGBM+XGBoost+CatBoost 블렌딩을 검증하다가,
**블렌딩보다 CatBoost 단독이 모든 fold에서 가장 좋다**는 걸 발견했다.

| 모델 | 2022 | 2024 |
|---|---|---|
| LightGBM 단독(A, 4차까지의 프로덕션) | 2261.64 | 685.86 |
| XGBoost 단독 | 2256.41 | 657.00 |
| **CatBoost 단독** | **2347.67** | **783.01** |
| LGBM+XGB+CAT 균등 블렌딩 | 2310.81 | 737.77 |
| LGBM+CAT 블렌딩 | 2324.11 | 759.54 |

피처(82개)ㆍshrinkage `k=50`ㆍ`CAT_COLS`는 A와 완전히 동일하고 분류기만
교체했다(`depth=6`, `learning_rate=0.02`, `l2_leaf_reg=3.0`).
6.11절에서 CatBoost가 639.22로 LightGBM보다 나빴던 것과 결과가 뒤집힌
이유는 두 가지로 보인다: (1) `learning_rate`를 0.02로 낮춰 충분히
학습시킨 것, (2) `OrdinalEncoder`로 미리 수치화하지 않고 **CatBoost의
네이티브 범주형 처리(ordered target statistics)** 를 그대로 쓴 것 —
팀ID 범주형 전환만으로도 개선이 있었던 6.9절의 연장선.

**결과: 로컬 685.86 → 783.01(+97.15), 실제 리더보드 758.37 → 902.04(+143.7).**

#### ⭐ 가장 중요한 방법론 교훈 — "개선의 크기"가 전이 여부를 가른다

6.10~6.17절에서 실패한 것들(피처 추가, 범주형 인코딩, k 튜닝, 정규화,
recency weighting)은 **전부 로컬 +5~9 수준의 미세 개선**이었고 실제로는
전부 하락했다. 반면 CatBoost는 로컬 +97로 자릿수가 다른 개선이었고
실제로도 +143.7로 크게 전이됐다.

즉 지금까지 내렸던 "로컬 검증은 실제를 예측 못 한다"는 결론은 정확하지
않았다. 올바른 결론은 **"노이즈 밴드(±10) 안의 개선은 못 믿고, 자릿수가
다른 개선은 믿을 수 있다"** 이다. 앞으로 채택 기준은 이걸로 한다 —
±10 수준 등락은 아예 제출하지 않는다.

#### ⚠️ 제출 환경 이슈 — pickle에 pandas 객체를 넣으면 안 된다

이 모델의 첫 두 번의 제출은 서버에서 다음 에러로 실패했다.

```
NotImplementedError: (<StringDtype(storage='python', na_value=nan)>,
                      array(['season', 'game_month', 'game_dayofweek', ...]))
```

원인은 CatBoost가 아니라 **`joblib.load()` 단계**였다.
`CatBoostWrapper.medians_`를 pandas Series로 저장했는데, 학습 환경(로컬
pandas 3.0.5)에서 만든 그 Series의 Index dtype이 `StringDtype`이라 평가
서버 pandas(`requirements.txt`의 2.3.3)가 복원하지 못한 것 — **모델을
쓰기도 전에 죽으므로 추론 코드를 아무리 고쳐도 소용이 없었다**(실제로
추론 경로만 고친 수정을 두 번 하며 제출 기회를 낭비했다).

- **수정**: `medians_`를 `num_cols` 순서의 순수 numpy float64 배열로 저장.
  pickle에 pandas 객체가 하나도 없음을 원시 바이트 스캔으로 확인.
- **교훈**: 로컬 클린룸 테스트(같은 인터프리터로 zip 풀고 실행)로는 이
  부류의 버그를 절대 못 잡는다 — 학습과 같은 pandas를 쓰니 항상 통과한다.
  반드시 **서버 환경을 재현**해서 검증해야 한다:
  `uv venv --python 3.11` + `uv pip install -r requirements.txt`.
  이렇게 재현하니 수정 전 파일에서 동일 에러가 그대로 재현됐고(원인 특정
  확인), 수정 후 실제 규모 25만 행에서 1.34초ㆍ1.6GB로 정상 완료됐다.

### 6.19 팀원 코드 비교 검증 — 우리 환경에서 직접 실행

팀원 두 명의 코드를 받아 우리 데이터ㆍ환경에서 직접 돌려 비교했다.

| 코드 | 결과 |
|---|---|
| **ayeon** (R/F 완전 분리 + LGBㆍXGBㆍCATㆍDCNv2 4-way 앙상블) | walk-forward ENSEMBLE AVG BSS=1002 (2022=2271, 2023=0, 2024=736) — 자체 보고치(731)와 거의 일치, 재현성 있는 좋은 코드. 실측 LB 772.38 |
| **kang** (RandomForest + `CalibratedClassifierCV(isotonic)`) | 2022=**0.00**, 2024=**43.85** — 사실상 망가짐 |

kang 코드가 무너진 원인은 명확하다 — `season`을 원본 수치형 피처로 그대로
넣었는데 RandomForest는 학습 때 못 본 미래 시즌 값을 외삽하지 못한다
(6.5절에서 우리가 겪고 `season_trend_success_prior`를 만든 바로 그 문제).
여기에 확률 보정까지 얹으면서 더 무너졌다.

#### ayeon 요소를 우리 CatBoost에 결합 시도 — 전부 실패

| 결합안 | 2024 |
|---|---|
| R/F 완전 분리 + CatBoost (`merged/`) | 680.20 |
| R/F 분리 + 직전시즌 리그별 shrinkage prior (`variant_league_split/`) | 702.91 |
| 공유모델 + 직전시즌 리그별 shrinkage prior (`variant_shared_model/`) | 771.97 |
| **순수 CatBoost 공유모델 (현재 902점 구성)** | **783.01** |

셋 다 아무것도 안 얹은 원본보다 낮았다. 추정 원인: CatBoost의 강점이
`game_type`을 다른 범주형과 교차 조합하는 것 + R/F를 합친 큰 학습
데이터인데, R/F 분리가 정확히 그 둘을 없앤다. **즉 902점 코드에 ayeon
요소는 하나도 들어가 있지 않다** — 서로 다른 두 파이프라인에서 각자
좋았던 요소를 합친다고 더 좋아지지 않는다는 사례.

다만 서로 독립적으로 같은 결론에 도달한 것들은 신뢰도가 매우 높다:
`trackman_history.csv` 직접 결합 기각, `pitcher_id`/`batter_id` 범주형
기각(ayeon은 R 3구간 전부 -29/-115/-122, 우리는 실제 LB 755.91<758.37),
Li 조건부 클러치 피처 기각(서로 다른 방법론인 OOF와 LOO로 각각 실패),
명시적 추세ㆍ레벨 보정 기각(ayeon은 시즌 센터링/레짐 가중치/Platt,
우리는 recency weighting/CalibratedClassifierCV).

### 6.20 kang 아이디어 3개 결합 시도 — 기각 (902점 구성 유지)

kang 코드의 파생 피처 8개 중, ayeon이 이미 재검증해 기각한 3개
(`is_scoring_pos`, `pitcher_batter_diff`, `recent_trend`)와 우리가 이미 갖고
있는 2개(`is_same_hand`≈`hand_matchup`, `is_full_count`)를 빼면 **아무도
검증하지 않은 3개**가 남는다 — `crisis_pressure`(`li × (주자수+1)`),
`stubbornness_index`(`최대 구종비율 × li`), `control_struggle_index`
(`누적 볼비율 × li`). 이 3개를 현재 902점 CatBoost 구성에 추가해 검증했다
(`train_v16_kang_features.py`).

| val_season | baseline (902점 구성) | +kang 3개 | 차이 |
|---|---|---|---|
| 2022 | 2347.67 | 2349.93 | +2.25 |
| **2024** | **783.01** | **770.09** | **-12.92** |
| 평균 | 1565.34 | 1560.01 | **-5.33** |

baseline이 이전 측정값과 소수점까지 재현돼 실험 자체는 신뢰할 수 있다.
신규 피처 중요도가 0.1~0.33으로 82개 중 최하위권 — 모델이 거의 쓰지 않는데
피처 수만 늘어난 셈이고, 가장 중요한 2024 fold에서 명확히 하락했다. **기각.**

**패턴 재확인 (6번째)**: "이미 있는 피처를 곱해서 만든 interaction"은 이
프로젝트에서 계속 실패한다 — 매치업, 클러치, `asof` 곱/차이,
`inning×asof_pitcher`, `inning×win_expectancy`, 그리고 이번 `li×3종`.
모델을 LightGBM에서 CatBoost로 바꿔도 이 패턴은 유지됐다. ayeon이 내린
"`li`가 target과 거의 무상관이라 조건부로 쪼개면 신호가 아니라 노이즈만
남는다"는 진단과도 일치한다. 앞으로 이 부류 아이디어는 시도하지 않는다.
