"""LightGBM 기반 control_success 예측 모델 - 공용 피처/파라미터 정의.

시즌 단위 walk-forward holdout(train.py 참고: 2021→22, 22→23, 23→24)으로
검증해 다음을 확인했다.

- asof_* 이력 피처(19개)는 시즌이 바뀌어도 신호가 유지된다 (AVG BSS 646).
- pitcher_hand x batter_hand 좌우 매치업(원-핫 4종 + 동손 여부)을 추가하면
  3개 시즌 경계 전부에서 일관되게 BSS가 개선된다 (AVG BSS 680).
- trackman_history.csv 기반 시즌 리그 평균 구위 지표(구속/회전수 등, 2025년치는
  2024 값을 그대로 carry-forward)는 season 자체와 마찬가지로 "시즌을 가리키는
  값"이라 오히려 일반화를 해친다는 것을 직접 실험으로 확인했다 (AVG BSS 605) -
  채택하지 않는다.
- balls_before/strikes_before/inning/home_win_expectancy/num_runners_on/li는
  asof_*+매치업 위에 하나씩 추가했을 때 3개 시즌 경계 모두 꾸준히 개선됐다
  (AVG BSS 680 -> 805). 이 피처들은 야구 규칙/경기 역학에 기반한 값이라
  연도별로 성격이 변하지 않는 것으로 보인다.
- game_type(1군/퓨처스)은 리포트상 성공률 차이가 가장 커서(0.514 vs 0.603)
  유망해 보였지만, 실제로 추가해보면 특정 시즌 경계(train<=2022 -> val 2023)에서
  BSS가 0으로 폭락하는 불안정한 패턴을 보였다 (season과 비슷한 방식으로 신뢰할
  수 없음) - 채택하지 않는다.
- outs_before는 추가해도 개선이 없어(리포트의 상관계수 0에 가깝다는 관찰과
  일치) 제외했다.
- season/balls_before/strikes_before/inning을 전부 한꺼번에 넣었던 이전
  로지스틱회귀 버전이 실패한 건 이 피처들 자체가 나빠서가 아니라 season의
  비선형 드리프트가 나머지 신호를 덮어썼기 때문으로 보인다.
- feature_fraction/bagging_fraction 서브샘플링을 추가하면 과적합이 줄어
  3개 시즌 경계 모두 추가로 개선된다 (AVG BSS 805 -> 825). 반면 최근 시즌에
  샘플 가중치를 더 주는 방식은 구간별로 들쭉날쭉해 채택하지 않았다.
- 볼카운트 파생(count_pressure, 3볼/2스트라이크 플래그, 12-way count 범주)은
  거의 개선이 없었다 (+-6 이내, 노이즈 수준) - LightGBM이 balls_before/
  strikes_before 원본만으로 이미 필요한 분기를 스스로 찾기 때문으로 보인다.
  채택하지 않는다.
- asof_pitcher_success_rate/asof_batter_success_rate에 표본수 기반 Bayesian
  shrinkage(원본 대신 교체, 원본 옆에 추가 둘 다)를 적용하면 오히려 나빠졌다
  (825 -> 805~822). 트리 모델은 asof_pitcher_n/asof_batter_n을 별도 피처로
  이미 갖고 있어 "n이 작으면 이 rate를 덜 믿는" 분기를 스스로 학습할 수 있는데,
  미리 shrink하면 오히려 원본 정보(정확한 n, 극단값)를 지워버리는 셈이라
  채택하지 않는다.
- "위기 상황(li 상위 20%)에서의 투수별 성공률"을 OOF(K-fold, fold 제외
  누적)로 만들어 추가하면 뚜렷하게 나빠졌다 (825 -> 753, 3개 구간 전부 악화).
  li 자체가 애초에 target과 상관관계가 거의 없어서(리포트 확인), 조건부로
  쪼개면 신호가 아니라 표본이 작아진 노이즈만 남는 것으로 보인다. 채택하지
  않는다.
- LightGBM 단일 모델보다 LightGBM+XGBoost+CatBoost 균등 가중 평균 앙상블이
  3개 시즌 경계 모두(또는 대부분) 일관되게 낫다 (AVG BSS 825 -> 847). 개별
  모델 성능은 서로 비슷비슷하지만(LGB 825, XGB 814, CAT 812) 평균을 내면
  노이즈가 줄어드는 것으로 보인다. 가중치를 불균등하게 튜닝해봐도(예: 2:1:2)
  840~849 범위로 큰 차이가 없어 균등 가중치를 그대로 쓴다.
- game_type(R/F)별 레짐 가중치(R: 최근 시즌 지수 감쇠, F: 2023 이전 행 제외)를
  시도했으나 기각했다. bss_R/bss_F 서브그룹 지표만 보면 개선처럼 보이지만
  (F가 0에서 크게 개선), 실제 채택 기준인 pooled walk-forward AVG BSS로 보면
  3모델 앙상블 기준 847 -> 724로 오히려 악화된다 - 특히 train<=2021 구간이
  1795 -> 919로 거의 반토막나는 급락이 있었다 (Phase5 "walk-forward 급락 =
  과적합" 경고 그대로). 서브그룹 BSS가 좋아 보여도 R/F 성공률 수준 차이 때문에
  pooled BSS와 반대 방향으로 움직일 수 있다는 걸 보여준 사례 - 앞으로 이런
  가중치/서브셋 실험은 반드시 pooled walk-forward AVG BSS로 최종 판정할 것.
- 위 발견 이후 game_type을 is_F 원-핫 피처로 직접 추가하는 것도 재시도했지만
  (17번째 항목과 동일 아이디어를 레짐 구조를 안 상태에서 재검증), train<=2022
  -> val=2023에서 BSS=0으로 붕괴하는 게 정확히 재현됐다 (AVG 825 -> 797).
  원인이 명확해졌다: F의 성공률 자체가 2022(0.71)->2023(0.47)에 뒤집히는
  비정상(non-stationary) 값이라 season을 직접 피처로 넣는 것과 동일한 실패
  구조다 - 2022 이전 데이터로 "is_F=1 -> 고성공률"을 학습한 트리가 2023부터는
  반대로 작동한다. 재시도할 가치 없음, 완전히 기각.
- asof_pitcher_success_rate shrinkage(33번째 항목)도 축소 기준을 "레짐이 섞인
  전역 평균"이 아니라 "직전 시즌의 (game_type) 평균"(leak-safe, 미래 정보
  없음)으로 바꿔 재시도했으나 여전히 개선이 없었다 (원본 옆에 추가: k=10/30/100
  각각 AVG 819/825/815, 원본 대체: k=30 AVG 830) - 전부 baseline(825) 대비
  노이즈 수준 내 등락. 축소 기준 자체의 문제가 아니라 애초에 트리가
  asof_pitcher_n으로 이미 신뢰도를 스스로 반영한다는 원래 결론이 맞았다.
  재시도 완료, 기각 확정.
- LightGBM+XGBoost+CatBoost 3종 앙상블(트리 계열)에 DCNv2(Deep & Cross
  Network v2, PyTorch) 하나를 4번째 멤버로 균등 가중 추가하면 R/F 분리 구조
  위에서 추가로 개선된다. DCNv2 단독 성능은 트리보다 낮지만(bss_R 639 vs
  685, bss_F 168 vs 498) 오차 패턴이 트리와 달라 블렌딩하면 이득이 남 -
  walk-forward 3구간 모두(R은 3구간 전부, F는 신체제 데이터가 있는 마지막
  구간에서만 테스트 가능) 4-way 균등 블렌딩이 3-tree 대비 일관되게 좋았다
  (train<=2021: bss_R 513->523, train<=2022: 644->657, train<=2023:
  bss_R 685->699 / bss_F 498->514 / bss_all 701->715). 가중치를 0.1~0.3
  범위로 바꿔봐도 결과가 크게 안 흔들려(710~715) 노이즈가 아니라 진짜
  다양성 효과로 판단, 균등 가중(1/4)을 채택했다. NaN은 표준화 후 0으로
  채우고 결측 여부를 별도 지시 피처로 추가해 처리한다(트리처럼 자체 분기
  학습이 안 되므로).
- 위 세 실험(가중치/피처/shrinkage)은 전부 R과 F를 같은 트리 앙상블에 섞어
  넣은 채로 레짐 신호만 주입하려 한 시도였다. R/F를 아예 별도 모델로
  완전히 분리해서 학습하고 game_type으로 라우팅하는 방식을 시도하니 채택.
  F는 구레짐(2023 이전) 행을 섞으면 분리해도 여전히 붕괴한다는 걸 먼저 확인했다
  (F 전체(2019-2023) 학습 -> bss_F=0, pred_mean 0.569 vs 실제 0.459로 과대예측;
  F를 신체제(2023)만으로 학습 -> bss_F=456로 회복, pred_mean 0.465 vs 실제
  0.459로 정상 캘리브레이션). 즉 F의 문제는 "R과 섞여서"가 아니라 F 자체
  내부에 2022->2023 레벨 반전이 있다는 것 - R을 완전히 떼어내도 구F를 학습에
  포함하면 여전히 깨진다.
  최종 채택안: R은 전체 데이터 + 기존 표준 파라미터로, F는 season >= 2023
  행만 + 표본이 작아(신체제 기준 R의 1/4 수준, 55,696행) 더 가벼운 파라미터
  (LGB num_leaves 31->15/min_child_samples 200->100, XGB max_depth 5->3/
  min_child_weight 200->100, CAT depth 5->3/l2_leaf_reg 200->100)로 각각
  독립 학습한다. walk-forward train<=2023->val=2024(유일하게 신체제 F 데이터가
  존재하는 구간, 2021/2022 컷오프는 2023 이전이라 이 설계를 테스트할 F 데이터가
  구조적으로 없음) 기준 공유 3모델 앙상블 bss_all=510(bss_R=655,bss_F=0) ->
  분리 3모델 앙상블 bss_all=701(bss_R=685,bss_F=498) - R/F 둘 다 트레이드오프
  없이 동시에 개선. (split_regime/LGB_F_PARAMS 등 참고)

- (LB 772.38 제출 이후) R/F 완전 분리 구조 위에서 신규 피처 5개를 하나씩
  재검증했다.
  * is_scoring_pos(주자 2/3루 여부), pitcher_batter_diff(투수-타자 성공률
    차), R 시즌 드리프트 센터링(asof_pitcher/batter_rate - 직전시즌 R평균)은
    R/F 모두 노이즈 밴드(대략 ±10) 안이거나 가장 중요한 train<=2023->val=2024
    구간에서 마이너스라 채택하지 않았다.
  * recent_trend(asof_pitcher_prev1_game_success_rate - asof_pitcher_
    success_rate)는 예상대로(prev1이 단일 경기라 노이즈 큼) F에서 뚜렷하게
    악화됐다(-18~-27, 3모델 앙상블 기준) - 기각.
  * asof_pitcher_success_rate shrinkage를 다시 시도했는데, 이번엔 33번째
    항목(공유 모델에서 기각)과 달리 채택했다. 차이는 축소 기준: 이제 R/F가
    완전히 분리돼 있으므로 "직전 시즌의 (해당 game_type) 평균"을 각 리그
    자체 이력만으로 leak-safe하게 계산할 수 있다(compute_season_means/
    add_shrunk_pitcher_feature 참고, k=100으로 스윕해서 확정). walk-forward
    3구간 전부에서 R은 동일하거나 개선(+3~+17), F는 크게 개선(+27~+34,
    baseline bss_F 498 -> 525) - 33번째 항목이 실패했던 건 shrink 아이디어
    자체가 나빠서가 아니라 그때는 축소 기준(전역/레짐 섞인 평균)이 오염돼
    있었기 때문이었다는 게 확인됐다. pitcher_batter_diff와 같이 넣으면
    오히려 나빠져서(525 -> 489, 정보가 겹쳐 트리가 혼란) 같이 쓰지 않는다 -
    shrinkage 단독으로 R/F 공통 FEATURES에 추가한다.

- shrinkage를 추가하고 나서 F용 DCNv2가 6개 시드 전부 bss_F=0으로 완전히
  붕괴하는 걸 발견했다(트리는 멀쩡). 원인: shrunk_pitcher_rate가
  asof_pitcher_success_rate와 사실상 중복 정보라, CrossLayer의
  x0 * linear(x) 곱셈 구조에서 다중공선성이 학습 자체를 붕괴시킨다
  (bss_train은 0->2500까지 계속 오르는데 bss_val은 최적 epoch에서도 0 -
  전형적 오버피팅). 트리는 중복 피처가 있어도 그냥 더 나은 쪽으로 분기하면
  그만이라 문제없지만 신경망은 다르다. DCN_FEATURES(FEATURES에서
  shrunk_pitcher_rate만 제외)를 따로 둬서 DCNv2 입력에서만 뺐다(R도 대칭성을
  위해 제외, R은 원래도 영향 거의 없었음: 630~639 동일). 겸사겸사 표준화값을
  +-DCN_CLIP(5.0)으로 클리핑하는 것도 추가했다 - F 같은 소표본 subset은
  일부 pitcher/batter의 middle_rate/ball_rate/li 같은 비율 피처가
  z-score +25까지 튀는 경우가 있어 마찬가지로 CrossLayer를 불안정하게
  만들 수 있다.
- 이 수정 후에도 F용 DCNv2는 시드에 따라 단일 실행 bss_F가 33~159로
  들쭉날쭉했다(std가 평균의 40%대) - 표본이 작아(신체제 F 5.6만행) 신경망
  하나의 학습이 원래 노이즈에 민감하다. 여러 시드로 각각 학습해 예측을
  평균 내는(BatchEnsemble이 아니라 그냥 독립 재학습 N번) 방식으로
  안정화했다: F는 1시드 33 -> 5시드 평균 366으로 극적으로 개선/안정화됐고,
  R도 도움이 됐다(1시드 630 -> 3시드 674) - 다만 R은 학습셋이 훨씬 커서
  시드 늘리는 비용이 크고 개선폭도 F만큼 크지 않아 3개로 제한했다
  (DCN_R_N_SEEDS=3, DCN_F_N_SEEDS=5).
- pitcher_id/batter_id를 CatBoost 범주형(cat_features)으로 R/F 각각 독립
  시도했으나 뚜렷하게 악화돼 기각했다 - R walk-forward 3구간 전부
  마이너스(-29, -115, -122), F도 -174. asof_pitcher_success_rate 등 이미
  강력한 투수/타자별 as-of 집계 피처가 있는 상태에서 원시 ID를 추가로 주면,
  CatBoost의 ordered target statistics가 특정 시즌/상대에 묶인 학습 데이터의
  우연한 패턴을 그 선수 고유의 성질인 것처럼 암기해버리는 것으로 보인다 -
  season을 직접 피처로 넣었을 때와 같은 "학습 시점에만 유효한 정보를
  외운다" 실패 패턴의 변형이다. 재시도 가치 낮음.
- F 전용 하이퍼파라미터를 그리드서치 + 시드 3개 평균으로 재탐색했다("F는
  표본이 작으니 params를 덜 보수적으로(min_child_samples 낮추고 등)"라는
  가설 검증 목적). 결과는 가설과 정반대였다 - min_child_samples를 낮추면
  (더 복잡한 트리 허용) 일관되게 나빠졌고, 오히려 더 얕은 트리가 계속
  좋았다. LGB_F_PARAMS(num_leaves 15->5, min_child_samples 100->200,
  learning_rate 0.03->0.01)와 XGB_F_PARAMS(max_depth 3->2)를 채택했다
  (시드평균 bss_F: LGB 457->494, XGB 467->497, 둘 다 std<=9로 재현성
  확인). CAT_F_PARAMS는 그리드서치 결과 원래 설정(depth=3, l2_leaf_reg=100)
  이 이미 거의 최적이라(bss_F=511, depth=4가 근소하게 더 좋지만 차이가
  노이즈 수준) 그대로 뒀다. F가 표본이 작다는 사실 자체는 맞지만, 그래서
  필요한 건 "덜 보수적인 트리"가 아니라 "더 보수적인(얕은) 트리"였다 -
  표본이 작을수록 과적합 여지가 커지는 게 당연한데 처음 LGB_F_PARAMS를
  정할 때(R 기준값을 대충 절반으로 스케일) 이 방향을 반대로 짚었던 것으로
  보인다.
- 위 F 재탐색을 채택한 직후 walk-forward 최종 숫자가 이전 마일스톤(732)보다
  낮게(724) 나와서 재현성 문제가 있는지 점검했다(사용자 지적). 확인 결과
  LGB/XGB/CAT_BASE_PARAMS 어디에도 random_state가 고정돼 있지 않았다 -
  TREE_SEED=0으로 전부 고정했더니 트리 부분은 완전히 결정론적이 됐다(별도
  프로세스로 3회 재실행해도 소수점까지 동일, std=0). 다만 DCN을 포함한 전체
  파이프라인은 고정 후에도 여전히 미세한(~0.3%, 2~6/700대) 프로세스 간 편차가
  남는다 - PyTorch의 멀티스레드 CPU 연산이 manual_seed로도 완전히 결정론적이
  되지 않는 잘 알려진 특성으로 보이며(부동소수점 합산 순서가 스레드 스케줄링에
  따라 달라짐), 크기가 작아 추가로 손대지 않는다(단일 스레드 강제는 학습
  속도를 크게 깎는데 비해 얻는 게 이 정도 노이즈 제거뿐).
  "732 vs 724" 자체는 회귀가 아니라 서로 다른 걸 잰 것이었다 - 732는 F용
  DCNv2가 아직 다중시드 평균화되기 전(고정 안 된 단일 시드) 스냅샷이고, 724는
  5시드 평균화 이후 값이다. 같은 DCN(5시드 평균)을 고정하고 트리 파라미터만
  구/신으로 바꿔 비교하는 통제 실험을 다시 돌려보니, F 재탐색의 진짜 순수
  효과는 bss_F 531 -> 536으로 +5뿐이었다 - 처음 보고했던 "LGB +37/XGB +30"은
  여러 랜덤시드 평균값이었는데, 프로덕션은 고정 시드 하나만 쓰다 보니(특히
  XGB는 우연히 seed=0에서 구 설정도 이미 점수가 높아서) 실제 이전분이 훨씬
  작았다. 그래도 방향은 맞고 재현 가능한(std=0) 개선이라 채택은 유지한다.
  R에도 같은 "더 얕게" 논리가 적용되는지 확인했으나 정반대로 나왔다(LGB
  num_leaves 낮추면 672->662~668로 전부 악화, XGB max_depth 낮추면
  672->645~658로 전부 악화) - R은 데이터가 많아(130만행) 이미 깊은 트리가
  최적이라 F와 다른 처방이 맞다는 뜻. R은 변경하지 않는다.
- R/F 분리 + shrinkage + F 하이퍼파라미터 재탐색을 다 반영한 새 구조에서
  캘리브레이션을 재점검했다. ECE는 예전 공유 모델(0.0143)보다 이미 나아졌지만
  (R 0.00866, F 0.00933) 완전히 사라지진 않았다 - 특히 R은 마지막 구간만
  빼고 전 구간에서 일관되게 과대예측(gap +0.01~0.014)하는 뚜렷한 패턴이
  남아있었다. Platt scaling(logit 1차 로지스틱 회귀)을 적용해 검증했는데,
  이번엔 val(2024)을 다시 반으로 쪼개서(3~6월에 적합, 7~10월에 평가 - 적합에
  안 쓴 진짜 held-out) 리키지 없이 확인했다: R bss 534->558(+24),
  F 561->575(+14) - 둘 다 실제 개선으로 보여서 일단 채택하고 LB 제출까지
  했었다(train<=2023->val=2024 전체로 다시 적합한 최종 파라미터 사용).
- **위 Platt scaling을 이후 완전히 철회했다.** LB가 772.38 -> 763으로
  떨어져서 원인을 다시 조사하다가(shrink_season_means 매핑 버그, game-level
  누수는 코드/구조상 확인 결과 둘 다 기각 - 매핑은 실제 값으로 재확인하니
  정상, game_id 컬럼 자체가 없고 시즌 단위 분할이라 게임이 train/val
  경계를 넘을 수 없음) Platt이 유력한 원인으로 드러났다. R의 walk-forward
  3구간 각각에서 Platt을 개별 적합해보니 계수의 **부호 자체가 해마다
  뒤집혔다**(2022 intercept=+0.005 거의 중립, 2023 intercept=+0.017 과소
  예측 보정, 2024 intercept=-0.028 과대예측 보정). 결정적으로, 2023에
  적합한 Platt을 2024 예측에 적용하는 연도 간 전이 테스트를 해보니 bss_R이
  703->676으로 오히려 악화됐다(반대 방향인 2022->2023 적용은 666->670로
  미미한 개선에 그침) - 이건 실제 배포 상황(2024에 적합한 Platt을 2025
  테스트에 적용)과 정확히 같은 구조로, 똑같이 실패할 가능성이 높다는 뜻이다.
  "같은 해 안에서" 통과했던 월별 분할 검증(+24/+14)은 방법론이 틀려서가
  아니라, R이 매년 드리프트량 자체가 달라(연속 드리프트, 매년 동일한 속도가
  아님) "그 해 고유의 잔차"를 그 해 안에서는 일관되게 잡아낸 것뿐이었다 -
  안정적인 모델 편향이 아니라 매년 다시 계산해야 하는 그 해만의 상수였던
  것이다. F는 애초에 신체제 데이터가 1년(2024)뿐이라 이런 연도 간 교차
  검증 자체가 불가능했다는 것도 뒤늦게 확인됐다 - R처럼 사전에 이 실패
  패턴을 잡을 방법이 구조적으로 없었다. fit_platt/apply_platt 함수 자체는
  재사용 가능한 도구로 common.py에 남겨두되, train.py/script.py 파이프라인
  에서는 완전히 뺐다(train.py는 진단용 출력만 남김). 교훈: 같은 해 내부의
  held-out 분할로 검증하는 것만으로는 부족하고, 반드시 다른 해로의 전이
  여부까지 확인해야 한다 - 특히 R처럼 알려진 연도별 드리프트가 있는
  타겟에서는 "그 해 잔차"와 "안정적 편향"을 구분하지 못하면 이런 식으로
  역효과가 난다.
- **패턴 종합**: R 시즌 드리프트 센터링(기각), 레짐 가중치(기각, 롤백함),
  Platt 재보정(채택했다가 위에서 철회) - 이 세 가지는 서로 다른 시점에
  독립적으로 나온 아이디어였지만 전부 "R의 과거 추세/레벨을 근거로 미래
  레벨을 명시적으로 조정·보정하려는 시도"라는 공통점이 있고, 셋 다
  실패했다. 우연이 아니라 R의 연도별 드리프트 속도 자체가 일정하지 않아서
  (매년 -0.01~-0.02 사이를 오가며 정확한 크기가 요동침) "과거 N년의
  추세로 내년 레벨을 추정"하는 접근 자체가 이 데이터에서는 구조적으로
  안 통하는 것으로 보인다. 앞으로 이런 종류의 아이디어(R의 레벨을 시즌/
  드리프트 정보로 명시적으로 조정하려는 시도)가 나오면 이 세 번의 실패를
  근거로 우선순위를 낮게 잡는다. (반대로 shrinkage처럼 "표본이 작을 때
  더 큰 표본의 값 쪽으로 끌어당기는" 접근은 레벨을 외삽하는 게 아니라서
  실패 패턴에 해당하지 않는다 - 실제로 shrinkage는 채택됐다.)
- "위기상황(li 상위 20%) 조건부 투수 성공률"을 shrinkage와 함께 재시도했다
  (예전엔 shrinkage 없이 OOF만으로 시도했다가 825->753으로 실패한 항목,
  "shrinkage가 실패한 진짜 이유는 축소 기준 오염"이라는 이후 발견을 근거로
  재검증할 가치가 있다고 판단). K-fold(5) OOF로 pitcher별 위기상황 성공률을
  구하고 asof_pitcher_success_rate를 prior로 shrink(k=10/30/100/300)했다.
  R은 노이즈 수준(±1~4)이었고, F는 k를 올릴수록(원본 신호 비중을 줄일수록)
  계속 좋아지다가 baseline에 근접만 했다(k=300에서 492 vs baseline 490,
  +3 - 노이즈). shrink 강도를 극단으로 올리면 shrunk_value가 prior 자체로
  수렴하므로(중복 피처가 되어 순효과 0) 이 패턴은 "위기상황 조건부 신호
  자체가 원래 거의 없다"는 뜻이다 - shrinkage는 존재하는 신호를 노이즈에서
  덜 손실되게 꺼내는 도구이지, 없는 신호를 만들어내지는 못한다. li가
  target과 거의 무상관이라는 원래 결론(rule.md/data_analysis_report.md
  근거)이 재확인됐다. 기각.
- F 앙상블에서 DCNv2를 TabM(BatchEnsemble - 공유 가중치 + 멤버별 입력/출력
  스케일 벡터 r,s로 K개 가상 서브모델을 한 번에 학습, K=16)으로 교체하는
  프로토타입을 시도했다. 단독 성능은 TabM이 압도적으로 좋았다(DCN 5시드
  평균 bss_F=366 vs TabM 3시드 평균=495, TabM은 단일 시드조차 366을 대부분
  이겼다) - 작은 표본(F 5.6만행)에서 암묵적 앙상블 구조가 단일 신경망보다
  안정적이라는 가설이 맞았다. 그런데 트리 3종 + 각각을 블렌딩해보니 정반대
  결과가 나왔다: 트리+DCN=536, 트리+TabM=527 - DCN 비중을 0~100%로
  스윕해도 단조적으로 DCN 100%가 최고였다(536->534->531->527). 즉 DCN이
  개별 성능은 더 나쁘지만 트리와 오차 패턴이 더 달라서(다양성 기여도가 더
  커서) 앙상블에는 더 유용하다 - "개별 멤버 성능"과 "앙상블 기여도"가
  반대로 갈 수 있다는 걸 보여준 사례(is_scoring_pos 등에서 이미 본 패턴의
  변형). 프로덕션은 DCNv2를 유지, TabM은 채택하지 않는다.
- **F 전용 shrinkage 채택 + F 하이퍼파라미터 재탐색을 둘 다 철회했다.**
  Platt을 철회한 뒤에도 LB가 772.4 -> 763.6으로 거의 그대로였다(+0.3에
  불과, 로컬 R 전이 테스트가 시사한 -27 스케일과 전혀 안 맞음) - 즉 763
  회귀의 진짜 원인은 Platt이 아니라 R/F 통합 제출에 같이 묶여 있던 다른
  변경(shrinkage, F 하이퍼파라미터)일 가능성이 커졌다. F의 2024 검증셋을
  4개 구간(3-4월/5-6월/7-8월/9-10월)으로 쪼개 두 변경을 각각 다시
  확인했다:
  * F 하이퍼파라미터(신 vs 구): 구간별 delta가 +1/+11/-16/+26으로
    들쭉날쭉 - 전체 순효과 +4는 여러 구간이 서로 상쇄된 결과지 일관된
    개선이 아니다.
  * F shrinkage(있음 vs 없음): -4/+5/+41/-19로 더 심하다 - 전체 순효과
    +10은 7-8월 한 구간(+41)이 통째로 만든 숫자이고, 9-10월은 오히려
    -19로 뚜렷하게 나쁘다.
  둘 다 "여러 구간을 평균 내면 작은 양수로 보이지만 실제로는 구간마다
  방향이 뒤집히는" 패턴이라, F 검증셋(신체제 1년치, 3만행 남짓)이 단일
  구간으로 보면 신호와 노이즈를 구분하기엔 너무 작다는 뜻이다 - R의
  shrinkage가 3개 연도(2022/2023/2024) 전부에서 일관되게 +3~+17이었던
  것과 뚜렷이 대비된다(R은 신뢰, F는 이번 두 건 다 기각). F의 LGB/XGB
  파라미터는 원래 값(LGB num_leaves=15/min_child_samples=100/
  learning_rate=0.03, XGB max_depth=3)으로, FEATURES는 shrunk_pitcher_rate
  없이 되돌렸다 - CAT_F_PARAMS는 재탐색 때도 원래 값(depth=3,
  l2_leaf_reg=100)에서 안 바뀌었으므로 그대로 둔다. DCNv2는 애초에
  shrunk_pitcher_rate를 입력으로 받은 적이 없었으므로(다중공선성 버그를
  같은 시점에 같이 고쳤음, 위 항목 참고) 이번 롤백으로 F의 트리 3종과
  DCN이 자동으로 동일한 피처셋(ASOF+MATCHUP+SITUATIONAL, shrinkage 없음)을
  쓰게 된다 - 4개 멤버 전부 같은 시절의 F를 예측하도록 다시 맞춘 것이다.
  DCN의 다중시드 평균화(안정화 목적, 33->366처럼 자릿수가 다른 효과)와
  TREE_SEED 재현성 고정(모델 자체의 결과를 바꾸지 않는 순수 인프라 수정)은
  이번 롤백 대상이 아니다 - 둘 다 단일 구간 우연으로 설명되지 않는 명확한
  근거가 있었다.
  **앞으로 F 관련 실험 규칙**: R은 데이터가 많아(130만행) 단일 val=2024
  구간이나 3개 연도 walk-forward로 충분히 신뢰할 수 있지만, F는 신체제
  데이터가 1년(3만행 남짓)뿐이라 그 안을 여러 구간(최소 2~3개)으로 쪼개
  구간별로 방향이 일관되는지 반드시 확인한 뒤 채택 여부를 정한다 - 전체
  평균 하나만 보고 판단하지 않는다.
- 위 F 롤백 후 script.py(zip 내부 단독 실행용, common.py를 참조하지 않고
  FEATURES 등을 직접 복제해서 갖고 있음)의 복제본 동기화를 깜빡해서 실제로
  피처 개수 불일치(31 vs 30)로 크래시나는 submit.zip을 만들 뻔했다 -
  스모크 테스트(격리 환경에서 실제로 predict까지 끝까지 실행)가 이걸
  잡았다. 이런 수동 이중 정의 구조 자체가 반복적인 위험 요인이라는 게
  확인됐다 - 앞으로 script.py를 고칠 때마다 FEATURES/DCN_R_N_SEEDS/
  DCN_F_N_SEEDS 등 common.py와 겹치는 상수를 전부 diff 확인하고, 격리
  스모크 테스트(단순 실행 성공 여부뿐 아니라 실제 모델의 feature_name()과
  기대 리스트를 이름+순서까지 비교)를 반드시 거친다.
- R shrinkage의 순수 효과를 확인하려고 R/F 둘 다 772.38 시절(shrinkage 없음,
  DCN 단일시드)로 되돌려 제출했는데, 763대보다도 낮은 점수가 나와 혼란이
  있었다. 원인은 R shrinkage가 아니라 **F의 DCN도 같이 단일시드로 되돌린
  것**이었다 - 실제 배포된 dcn_model_F_seed0.pt를 직접 평가해보니
  bss_F=33(F 단일시드 분산 범위 33~514 중 최악에 가까움)이 나왔다. F의
  다중시드 평균화는 이미 검증된 안정화 장치인데, "772.38 완전 재현"을
  하겠다고 이것까지 걷어내면서 F 불안정성 버그를 스스로 재도입한 것 -
  이 결과는 R shrinkage와 무관하게 무효 처리했다. 교훈: 통제 실험은
  "그때 코드 그대로"보다 "지금 검증된 개선분은 유지하고 딱 하나의 변수만
  바꾸는" 방식이 안전하다 - 여러 변수를 한꺼번에 원상복구하면 그 중
  이미 검증됐던 개선분(F 다중시드)까지 같이 꺼져서 새로운 혼란을 만든다.
  최종적으로 F는 다중시드(R=3, F=5) 유지, R의 shrinkage만 단독으로 뺀
  버전으로 재구성해 제출 대기 중이다.
- 위 사건과 별개로, 763.28/763.63을 만들었던 커밋(e629d0e/7b195d5)을
  다시 확인해서 그때는 script.py/common.py 동기화 버그가 없었음을
  정적으로 확인했다(FEATURES/DCN_N_SEEDS/모델 파일명 패턴 전부 일치) -
  "763.28이 두 번 똑같이 나온" 최초 미스터리는 이 desync 버그로는 설명이
  안 된다(원인 미상으로 남음, 이후 실제로 점수가 움직이기 시작해서 더
  캐지는 않음).
- pitcher_win_expectancy(top_bottom으로 투수가 홈/원정인지 판별해
  home_win_expectancy/away_win_expectancy 중 투수 관점 값을 선택)와
  score_diff_pitcher_team(원본 제공 컬럼을 그대로 피처로) 둘 다 기각했다.
  pitcher_win_expectancy는 R 3구간 전부 일관되게 마이너스였고(-6/-5/-5,
  노이즈치고 방향이 너무 일정 - home_win_expectancy와 재조합한 값이
  기존 신호를 흐리는 것으로 보임), F도 전형적인 상쇄 패턴이었다
  (-38/+16/+123/-6, 순효과 +31은 7-8월 한 구간의 착시). score_diff_
  pitcher_team은 R이 노이즈 밴드(-10/+1/-7, 방향 불일치)였고 F도 다시
  크게 상쇄됐다(-25/+20/+75/-61).
- base_state(원본 8종 범주형, LGB 네이티브 categorical_feature로), game_month,
  game_dayofweek 셋 다 기각했다. base_state는 R 노이즈(+2/-8/-5), F는
  순효과까지 마이너스(-76/+13/+53/-64, 전체 -17). game_month는 R 3구간
  전부 마이너스(-2/-8/-3), F도 불안정+순음수(-77/-10/+91/-64, 전체 -4).
  game_dayofweek도 R 3구간 전부 마이너스(-7/-5/-7), F 역시 불안정+순음수
  (-29/-10/+69/-64, 전체 -9).
  **메타 패턴**: pitcher_win_expectancy/game_month/game_dayofweek 세
  피처 모두 R에서 3/3 구간 일관되게 마이너스였다 - 순수 노이즈라면 부호가
  섞여야 정상인데 이렇게 일관된 건 우연이 아닐 가능성이 있다. LGB_BASE_
  PARAMS의 feature_fraction=0.6(트리마다 피처 60%만 무작위 샘플링) 구조상,
  새 피처를 하나 추가하면 전체 피처 수가 늘어나면서 기존에 유용했던
  피처들이 각 트리에서 뽑힐 확률이 미세하게 희석되는 구조적 효과일
  가능성이 있다 - 즉 새 피처 자체의 정보량과 무관하게, "쓸모없는 피처를
  추가하는 것 자체"가 이 구성에서는 약한 비용을 문다는 뜻일 수 있다.
  확실하진 않지만, 앞으로 R에 새 피처를 추가할 때는 이 구조적 희석 비용
  (대략 -5~-8 수준)을 감안해 그보다 명확히 큰 개선이 아니면 채택하지
  않는 게 안전하다.
- asof_pitcher_late_inning_rate(이 투수가 8회 이후에 등판하는 빈도 - 성공률과
  엮지 않은 순수 사용 패턴/보직 신호, shrunk_pitcher_rate와 동일한 설계로
  train 내부는 진짜 causal 누적, val/test에는 train 끝 시점 스냅샷을 고정해서
  적용해 test.csv rolling 금지 규칙을 지킴)을 R 3구간 walk-forward + 트리
  3종 앙상블로 먼저 스크리닝했다. 결과: -5/-8/-2 - 바로 위 항목에서 확인한
  "새 피처 추가 자체의 구조적 희석 비용(-5~-8)"과 사실상 구분이 안 되는
  범위라 실제 신호가 없는 것으로 판단, 4모델 블렌드 단계까지 갈 필요 없이
  1단계에서 기각. "위기/상황/보직" 계열 피처(li 조건부 성공률 등, 위 항목들
  참고)가 반복적으로 실패하는 패턴이 이번에도 재현됐다 - li 자체가 target과
  거의 무상관이라는 EDA 결론이 "성공률과 엮은" 버전뿐 아니라 "순수 빈도"
  버전에도 마찬가지로 적용되는 것으로 보인다.

- trackman_mix_break_volatility(투수의 asof 구종 믹스로 가중평균한 수평+수직
  브레이크 변동성 - trackman_history.csv 기반, "존에서 크게 벗어난 공"이라는
  타겟의 실패 조건과 직결시키려는 물리 기반 설계, 2026-08-23)를 시도했다.
  이전에 기각됐던 "시즌 리그 평균 구위 지표"(605 vs 680) 실패와 구조적으로
  다르게 설계했다 - "레짐"을 시즌별로 값이 바뀌는 lookup key가 아니라
  trackman_history를 얼마나 신뢰할지 정하는 데이터 필터로만 써서, 최종
  (game_type, pitch_type_group) -> 변동성 테이블이 시즌에 의존하지 않게(테이블
  단 하나, 모든 행에 동일 적용) 만들었다. 1차 시도(R: 그 시점 최신 시즌
  단일년치만 필터)는 -10/+2/+2로 방향이 안 섞여 기각 후보였으나, 2021 구간의
  -10이 표본 부족(단일 시즌) 때문일 수 있다는 가설이 있어 재검증했다. 특정
  구간만 고치는 사후 땜질을 피하기 위해 "최소 2시즌 트레일링 윈도우"라는
  일반 규칙으로 재설계해 3구간 전부에 동일하게 적용했더니(2020-21/2021-22/
  2022-23), 결과가 -13/-6/-1로 오히려 전부 악화되고 이전엔 +2였던 두 구간마저
  마이너스로 뒤집혔다 - 표본 부족 가설이 틀렸다는 뜻이고(창을 넓혀도 안
  좋아짐), 방향도 이번엔 3구간 전부 일관되게 마이너스라 신호 자체가 없다고
  결론. 재설계로도 실패해서 완전히 기각, 재시도 가치 없음.
- R 내부 재분리(선발/불펜 등판 패턴 기준, R/F 분리가 만든 이득을 R 안에서도
  노리려는 시도)를 1단계 저비용 진단으로 먼저 확인했다(2026-08-23,
  try_r_inning_split.py) - 투수별 지속 역할 라벨(asof 기반, 임베딩 vocab과
  같은 패턴)을 만드는 정교한 버전 전에, 그 행 자체의 inning 값만으로 즉시
  early(<=5이닝)/late(6이닝+)로 라우팅해서 R 내부에 이질성 신호가 조금이라도
  있는지만 봤다. 가장 작은 서브그룹(train<=2021의 late, 284,869행)도 F
  신체제 표본(55,696행)보다 5배 이상 커서 표본 부족은 원인이 될 수 없다.
  결과: -48/+2/+6 - 3구간 전부 일관되지 않고, 가장 작은 구간에서 나온 -48은
  노이즈로 설명하기엔 너무 크다. 트리가 이미 inning을 피처로 갖고 있어
  early/late 상호작용을 스스로 학습할 수 있는데, 억지로 완전히 분리하면
  오히려 각 서브모델의 학습 데이터/피처 다양성만 줄어드는 것으로 보인다.
  진단 단계에서 신호가 없어 정교한 버전(1안)에 시간을 쓰지 않고 기각한다 -
  R은 F와 달리 game_type 수준의 구조적 레짐 반전이 없으므로, R/F 분리가
  통했던 메커니즘(서로 다른 타겟 분포/드리프트 방향을 가진 두 레짐을 억지로
  한 모델에 우겨넣지 않기)이 R 내부에는 적용되지 않는 것으로 보인다.
- 주자상황/불리한카운트/이닝후반 압박 계열을 두 가지 최종 변형으로 재시도했다
  (2026-08-25). (1) 투수별 조건부 as-of 성공률(asof_pitcher_scoring_pos_cond_rate/
  behind_count_cond_rate/late_inning_cond_rate, shrinkage 적용, 3개 블록) - R
  3구간 +10/-68/+4로 실패, -68은 위기상황(li) 조건부 성공률의 825->753 실패와
  동일한 메커니즘(투수별 표본이 조건으로 더 쪼개져서 노이즈만 남음). (2) 투수별이
  아니라 리그 전체(population) 수준 고정 상수(시간 무관, 두 값짜리 target
  encoding) - 블록 테스트 -10/-4/+14로 3구간 중 2개가 마이너스지만 가장
  프로덕션에 가까운 구간(train<=2023->val2024)만 +14로 튀어서 재검증 필요
  판단. 세 조건을 개별 ablation으로 재확인(scoring_pos/behind_count/
  late_inning 각각 단독): -3/-1/+5, -8/-1/+4, -9/-1/+6 - 셋 다 똑같은 모양
  (split1 소폭 마이너스, split2 거의 0, split3만 소폭 플러스)으로 나와서 블록의
  +14는 세 개의 작고 개별적으로는 일관되지 않은 기여가 거의 더해진 것뿐임을
  확인했다(+5+4+6=15 ≈ 블록의 +14) - 숨겨진 진짜 신호가 다른 두 개에 희석된
  게 아니라, 애초에 셋 다 신호가 없고 split3이라는 특정 구간에 공통으로 우연히
  걸린 것으로 결론. 주자상황/카운트/이닝 압박 계열(오늘 10개 이상 변형 시도)을
  여기서 완전히 종결한다 - 재시도 가치 없음.
- F 전용 SSL(2026-08-25, try_f_ssl_pretrain.py): R+F 전체 train_df(<=cutoff, 라벨/
  game_type 제외)로 denoising autoencoder(15~20% 피처 마스킹, encoder 128->64,
  decoder 64->128->원복)를 pretrain한 뒤, encoder를 F 라벨(tr_F)로만 fine-tune해서
  6번째 앙상블 멤버로 추가했다. walk-forward 무결성을 위해 pretrain도 train_df
  <=cutoff로 제한(미래 시즌 피처 분포까지 라벨 없이 미리 보는 것도 일종의 누출이라고
  판단). SSL 단독 성능은 501로 준수했지만(개별 트리 모델과 비슷한 수준), 5-way(기존
  프로덕션) -> 6-way(+SSL) whole-split bss_F는 543->543(+1)로 사실상 무변화 - 월별
  4구간도 +9/+8/-6/-16로 방향 불일치, 순효과 없음. 판정 기준(whole-split)에서
  기각 - SSL이 뭔가는 배웠지만 기존 5-way 앙상블이 이미 포착하고 있던 것과 겹쳐서
  새로운 다양성 기여가 없는 것으로 보인다.
LightGBM/XGBoost/CatBoost 모두 결측치를 학습 시 결정한 분기 방향으로 그대로
처리하므로 asof_*의 cold-start NaN을 별도로 대치(impute)하지 않는다.

script.py(제출용 추론 코드)는 zip 내부에서 로컬 모듈 import 없이 단독 실행돼야
하므로 이 모듈을 참조하지 않고 동일한 정의를 직접 포함한다.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

TARGET = "control_success"

ASOF_FEATURES = [
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
MATCHUP_FEATURES = [
    "matchup_1_1",
    "matchup_1_2",
    "matchup_2_1",
    "matchup_2_2",
    "same_hand",
]
SITUATIONAL_FEATURES = [
    "balls_before",
    "strikes_before",
    "inning",
    "home_win_expectancy",
    "num_runners_on",
    "li",
]
# add_shrunk_pitcher_feature()가 계산해서 덧붙일 수 있는 파생 피처 - 원본 CSV
# 컬럼이 아니므로 ASOF_FEATURES(usecols 목록)에는 넣지 않는다.
SHRUNK_FEATURES = ["shrunk_pitcher_rate"]

# **772.38 재현 통제 실험 중 - R shrinkage도 임시로 뺐다.** F(shrinkage+
# 하이퍼파라미터)를 되돌려도 LB가 763대에서 안 움직여서(763.28->763.63->
# 763.90, 772.38과 여전히 8.5점 차이) 남은 차이는 R shrinkage뿐이다. R
# shrinkage는 walk-forward 3개 연도 전부에서 robust하게(+3~+17) 나왔지만,
# 축소 기준이 "직전 시즌 평균"이라 구조적으로는 시즌 센터링/레짐 가중치/
# Platt과 같은 "과거 레벨로 미래를 보정" 계열이다 - walk-forward 3구간이
# 전부 드리프트가 계속되는 익숙한 패턴 안이었을 뿐, 2025이 그 패턴을
# 벗어나면 같은 실패가 반복될 수 있다는 우려가 있다. 772.38과 최대한
# 동일한 조건(피처셋+DCN 단일시드)에서 제출해 이 가설과 "그래도 763대에
# 머무르면 모델링이 아니라 인프라 문제"라는 대안 가설을 가른다. 결과가
# 나오면(common.py 상단 이력에 기록 예정) 다시 원복하거나 확정한다.
FEATURES = ASOF_FEATURES + MATCHUP_FEATURES + SITUATIONAL_FEATURES
FEATURES_F = FEATURES
DCN_FEATURES = FEATURES

# asof_pitcher_success_rate를 표본수(asof_pitcher_n) 기반으로 "직전 시즌의
# 해당 game_type 평균"쪽으로 shrink한 값을 원본 옆에 추가한다(대체 아님).
# k는 walk-forward로 스윕해서 확정한 값 - R은 영향이 거의 없고(동일~소폭
# 개선) F는 크게 개선된다(위 rationale 참고).
SHRINK_K = 100


def compute_season_means(df):
    """(game_type, season) -> TARGET 평균. 반드시 학습 데이터에서만 계산해야
    누출이 없다(compute_season_means(train_df), 절대 val/test 포함 df 아님)."""
    return df.groupby(["game_type", "season"])[TARGET].mean()


def add_shrunk_pitcher_feature(df, season_means, k=SHRINK_K):
    """각 행을 그 행 자신의 (season - 1) 시즌 평균(prior)쪽으로 shrink한다 -
    "그 시즌이 시작되기 전에 이미 확정된 값"만 쓰므로 leak-safe하다. season_means는
    항상 학습 데이터로만 계산된 것을 넘겨야 한다(val/test에도 동일 테이블 재사용).
    """
    df = df.copy()
    prior_idx = pd.MultiIndex.from_arrays([df["game_type"], df["season"] - 1])
    prior = season_means.reindex(prior_idx).to_numpy()
    n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    raw = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    shrunk = (n * raw + k * prior) / (n + k)
    shrunk[np.isnan(raw) | np.isnan(prior)] = np.nan
    df["shrunk_pitcher_rate"] = shrunk
    return df

# num_leaves/min_child_samples/learning_rate/feature_fraction/bagging_fraction은
# walk-forward holdout(AVG BSS 기준)으로 튜닝한 값.
# n_estimators는 train<=2023 -> val 2024 early stopping 기준 best_iteration(~200)에
# 맞춘 고정값 - 최종 모델은 전체 데이터(2019~2024)로 학습하므로 early stopping용
# 홀드아웃이 없어 이 값을 그대로 사용한다.
# bagging_fraction/subsample/rsm이 1.0 미만이라 세 모델 다 내부적으로 행/열
# 서브샘플링 난수를 쓴다 - random_state를 고정하지 않으면 매 실행마다 다른
# 트리가 나와서 walk-forward 숫자가 재현 안 되는 걸 뒤늦게 발견했다(사용자가
# 지적). 전부 고정한다 - 사전학습을 마친 최종 제출 모델도 이 값으로 학습돼야
# submit.zip을 다시 빌드해도 항상 같은 가중치가 나온다.
TREE_SEED = 0

# 2026-08-23 Optuna(TPE, optuna_r_search.py) 30 trials로 R 트리 파라미터를
# 재탐색해서 채택했다 - 지금까지는 대부분 수동으로 몇 개 값만 시도했었다.
# trial 0을 현재 프로덕션 값으로 enqueue해서 직접 비교(3-tree 블렌드 avg
# BSS 616.8), 최적 trial은 627.6(+10.8) - 3구간 델타 +19/+12/+1(전부 양수지만
# 가장 중요한 train<=2023->val2024 구간에서는 거의 0에 가까움, 다른 여러
# trial도 625~627대에 몰려있어 뚜렷한 단일 최적점이라기보다 완만한 고원).
# **4모델(DCN 포함) 블렌드로 재검증**(verify_optuna_best_blend.py, 오늘 확립한
# "부분 앙상블 개선이 전체 블렌드 개선을 보장 안 함" 규칙 적용) - 델타는
# +12/+9/+1로 tree-3-only보다 작아지지만 3구간 전부 양수 유지, F 아키텍처/R
# 시드 사례와 달리 음전환 없음. AVG BSS 645.3->652.4(+7.1). 학습률이 바뀌어서
# (특히 CAT: 0.03->0.059) LGB_FINAL_N_ESTIMATORS 등도 같은 방식(train<=2023->
# val2024 early stopping 기준 best_iteration)으로 재도출했다
# (get_optuna_final_n_estimators.py): LGB 200->133, XGB 180->110, CAT 370->180.
LGB_BASE_PARAMS = dict(
    objective="binary",
    learning_rate=0.03436822690526636,
    num_leaves=34,
    min_child_samples=168,
    feature_fraction=0.44300453584712784,
    bagging_fraction=0.9557377752299039,
    bagging_freq=1,
    random_state=TREE_SEED,
)
LGB_FINAL_N_ESTIMATORS = 133

XGB_BASE_PARAMS = dict(
    objective="binary:logistic",
    learning_rate=0.029800285477956538,
    max_depth=8,
    min_child_weight=340,
    subsample=0.7409175238876318,
    colsample_bytree=0.4588317189261018,
    eval_metric="logloss",
    random_state=TREE_SEED,
)
XGB_FINAL_N_ESTIMATORS = 110

CAT_BASE_PARAMS = dict(
    loss_function="Logloss",
    learning_rate=0.058582019413996565,
    depth=7,
    l2_leaf_reg=117.4895068424343,
    subsample=0.885369039661351,
    rsm=0.7541995020365571,
    verbose=False,
    allow_writing_files=False,
    random_seed=TREE_SEED,
)
CAT_FINAL_N_ESTIMATORS = 180

# 이전 버전(LightGBM 단일 모델)과의 호환을 위해 유지.
FINAL_N_ESTIMATORS = LGB_FINAL_N_ESTIMATORS

# F(퓨처스)는 신체제(season >= F_NEW_REGIME_START)만 학습에 쓴다 - 구F를 섞으면
# 2022->2023 레벨 반전 때문에 R과 분리해도 여전히 무너진다. 표본이 R의 1/4
# 수준으로 작아 더 가벼운(과적합 억제) 파라미터를 쓴다.
#
# LGB/XGB를 한때 더 얕게(num_leaves=15->5 등) 재탐색해서 채택했었으나 이후
# 철회했다 - F 검증셋을 여러 구간으로 쪼개보니 구간마다 방향이 뒤집히는
# 우연이었다(공용 rationale 참고). 원래의 "R 기준값을 절반으로 스케일"한
# 값으로 되돌린다.
F_NEW_REGIME_START = 2023

# **2026-08-23 주의**: R의 LGB_BASE_PARAMS/XGB_BASE_PARAMS/CAT_BASE_PARAMS가 Optuna
# 재탐색으로 바뀌면서(위 rationale), 예전처럼 `dict(LGB_BASE_PARAMS, ...)`로 F 파라미터를
# 만들면 R용으로만 검증된 새 학습률/서브샘플링 값이 F에 조용히 새어 들어간다 -
# validation-lessons rule 9("R/F는 절대 하이퍼파라미터를 공유하지 않는다")를 그대로
# 어기는 것이다. 그래서 F 파라미터는 이제 R_BASE_PARAMS를 전혀 참조하지 않고, F에서
# 실제로 검증됐던 값(원래의 "R 기준값을 절반으로 스케일"한 값, 위 rationale 참고)을
# 완전히 독립적으로 하드코딩한다 - R 쪽 재탐색이 몇 번을 더 일어나도 F는 영향받지 않는다.
# **2026-08-24 Optuna(TPE, optuna_f_search.py) 30 trials로 재탐색해서 채택했다가,
# 같은 날 LB 792.24->790.78(-1.46)로 되돌려서 완전히 철회했다.** 목적함수는
# 처음부터 whole-split 4-way(DCN 포함) 블렌드 bss_F였다(단일 브리어 계산, 월별
# 평균/합산 아님 - 코드로 직접 확인됨: y_F/p_blend를 마스킹 없이 통째로
# brier_skill_score에 넣은 값이 그대로 trial의 반환값이었다) - 그래서 F DCN
# 아키텍처 건(월별 vs whole-split 지표 자체를 잘못 봐서 실패)과는 다른 종류의
# 실패다: 이번엔 방법론 자체는 맞았는데(local whole-split +22, 월별 4구간 중
# 3/4 개선) 그 whole-split 검증된 이득 자체가 LB로 전이가 안 됐다. LB 델타
# (-1.46)는 이 프로젝트가 기록해 온 DCN 파이프라인 노이즈 하한(2~6점)보다도
# 작아서, "F local whole-split 검증도 LB 전이를 보장하지 않을 수 있다"는
# 새로운 불확실성으로 봐야 한다(validation-lessons rule 12 참고 - 거긴
# capacity/structure 계열이 로컬보다 LB에서 더 후하게 전이된 사례만 있었는데,
# 이번은 반대 방향 전이 실패 사례). 원래 값(R 기준값을 절반으로 스케일한 최초
# 설정)으로 완전히 되돌린다 - 재시도하려면 반드시 whole-split 재검증을 다시
# 거친 뒤에도 LB 전이 자체가 불확실하다는 걸 감안해서 판단할 것.
LGB_F_PARAMS = dict(
    objective="binary", learning_rate=0.03, num_leaves=15, min_child_samples=100,
    feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, random_state=TREE_SEED,
)
LGB_F_FINAL_N_ESTIMATORS = 85

XGB_F_PARAMS = dict(
    objective="binary:logistic", learning_rate=0.03, max_depth=3, min_child_weight=100,
    subsample=0.8, colsample_bytree=0.6, eval_metric="logloss", random_state=TREE_SEED,
)
XGB_F_FINAL_N_ESTIMATORS = 100

CAT_F_PARAMS = dict(
    loss_function="Logloss", learning_rate=0.03, depth=3, l2_leaf_reg=100,
    subsample=0.8, rsm=0.6, verbose=False, allow_writing_files=False, random_seed=TREE_SEED,
)
CAT_F_FINAL_N_ESTIMATORS = 280


def split_regime(df, f_new_regime_start=F_NEW_REGIME_START):
    """df를 (R 전체, F) 두 학습셋으로 나눈다.

    F는 신체제(season >= f_new_regime_start) 행만 쓴다 - 구F를 섞으면 레벨
    반전 때문에 무너지는 것을 확인했다. 학습 윈도우에 신체제 F가 아직 없으면
    (walk-forward의 train<=2021/2022처럼 f_new_regime_start 이전 컷오프)
    있는 F 전체로 대체한다 - 이 경우 그 구간은 이 설계의 실제 검증으로 보지
    않는다(F_NEW_REGIME_START 이후 데이터가 없어 구조적으로 테스트 불가).
    """
    r_df = df[df["game_type"] == "R"]
    f_new = df[(df["game_type"] == "F") & (df["season"] >= f_new_regime_start)]
    f_df = f_new if len(f_new) > 0 else df[df["game_type"] == "F"]
    return r_df, f_df


# DCNv2 - 트리 3종 앙상블에 4번째 멤버로 균등 가중 추가한다(위 rationale 참고).
# R/F 각각 다른 크기(F는 표본이 작아 더 작은 네트워크)를 쓴다.
# epoch 수는 walk-forward에서 val loss 기준 early stopping(patience=8)으로
# 확인한 값(R~11, F~15)에 맞춘 고정값 - 최종 모델은 held-out이 없어 이 값을
# 그대로 쓴다(트리의 *_FINAL_N_ESTIMATORS와 동일한 패턴).
#
# R DCN 아키텍처를 그리드서치(cross_layers x deep_dims)로 재탐색했다 -
# 최초 프로토타입 때 정해진 뒤로 한 번도 재검증한 적이 없었다. 원래 설정
# (cross_layers=3, deep_dims=(128,64))보다 cross_layers=2, deep_dims=
# (256,128)이 walk-forward 3구간 전부에서 견고하게 좋았다(시드 3개 평균
# 기준 +26~+28, std 4~38로 델타가 노이즈보다 뚜렷이 큼: 2021->22 414->442,
# 2022->23 589->615, 2023->24 631->659). 더 키우거나(512,256 / 3-layer
# deep) cross_layers를 1로 줄여봐도 전부 646보다 나빴다(628~641) - 정점
# 확인됨. R은 데이터가 많아(130만행) 더 큰 네트워크를 감당할 여력이 있었던
# 것으로 보인다(F와 반대 방향 - F는 계속 더 작게/얕게가 이겼음).
DCN_R_PARAMS = dict(cross_layers=2, deep_dims=(256, 128), dropout=0.1)
DCN_R_BATCH_SIZE = 4096
DCN_R_FINAL_EPOCHS = 11

# F DCN 아키텍처 재탐색(2026-08-22, search_f_dcn_arch.py) - 기존 값(cross_layers=2,
# deep_dims=(32,16))은 R의 원래 설정을 대충 축소한 추정치였을 뿐, R처럼 실제
# 그리드서치를 거친 적이 없었다. R의 재탐색 방향(cross_layers 줄이기)을 F에도
# 적용해보다가, F에서는 cross_layers를 아예 0으로(CrossLayer 완전히 제거,
# 순수 MLP+원본 피처 concat) 줄이는 쪽이 훨씬 좋다는 걸 발견했다 - cross=1
# 근방을 먼저 탐색했을 때도 이미 개선이 보였지만(예: cross=1,dims=(64,32)
# 시드3개평균 386 vs 베이스라인 293), cross=0으로 완전히 제거하니 더 크게
# 개선됐다(455). deep_dims도 F 표본 크기에서는 더 작을수록(16,8) 나았다
# (485, 128/256 같은 큰 폭은 오히려 나빠짐). CrossLayer의 x0*linear(x)
# 곱셈 구조가 다중공선성에 취약하다는 건 이미 shrunk_pitcher_rate 배제
# 사례(DCN_FEATURES 참고)에서 확인된 패턴인데, F처럼 표본이 작은 경우에는
# CrossLayer 자체가 (특정 피처와의 다중공선성 여부와 무관하게) 불필요한
# 불안정성의 원천이었던 것으로 보인다.
# 최종 확인은 프로덕션 시드 수(DCN_F_N_SEEDS=5)로 재검증했다 - train<=2023->
# val=2024, 월별 4구간(3-4/5-6/7-8/9-10월) 전부에서 방향 일관되게 개선:
# 베이스라인 overall=366(구간: 0/441/479/89, 3-4월은 완전 붕괴) -> 신규
# overall=462(구간: 326/525/403/128, 4구간 전부 베이스라인보다 우수, 특히
# 완전 붕괴하던 3-4월을 0->326으로 회복). 4구간 전부 개선인 깨끗한 채택 사례.
#
# **2026-08-23 전면 철회, cross_layers=2/deep_dims=(32,16)로 복귀.** 위
# 개선은 F-DCN **단독** 성능 기준이었다(트리와 블렌딩 안 함) - 실제 4모델
# 블렌드로 재확인하니(isolate_r_vs_f_change.py) bss_F가 531->526로 오히려
# 악화됐다. verify_f_whole_split.py로 366/462가 계산 방식 문제(월별 분할 vs
# whole-split)가 아님을 확인했고, reweight_new_f_dcn.py로 볼록결합 가중치
# 재탐색도 시도했지만 균등가중치(0.25x4)로 다시 수렴해 531을 회복하지
# 못했다 - 즉 CrossLayer 제거로 DCN 자체는 더 똑똑해졌지만 그만큼 트리
# 모델들과 오차 패턴이 수렴해서(다양성 손실) 블렌드에는 순손해였다. TabM
# 사례(위 섹션 참고)와 정확히 같은 "개별 성능 vs 앙상블 기여도 역전"
# 패턴. 원래 값으로 완전히 되돌린다 - 이 항목의 366->462는 "DCN 단독
# 벤치마크로는 유효하지만 이 구조에서는 배포하지 않는다"는 기록으로 남긴다.
DCN_F_PARAMS = dict(cross_layers=2, deep_dims=(32, 16), dropout=0.2)
DCN_F_BATCH_SIZE = 1024
DCN_F_FINAL_EPOCHS = 15

DCN_LR = 1e-3
DCN_WEIGHT_DECAY = 1e-5

# 시드마다 성능 편차가 크다는 걸 확인해서(F: 33~382 range, 단일시드 std가
# 평균의 40%대) 여러 시드로 각각 학습해 예측을 평균 낸다(F 1시드 33 -> 5시드
# 평균 366, R 1시드 630 -> 3시드 674).
#
# 한때 772.38 재현 통제 실험을 위해 둘 다 1(단일시드)로 되돌렸다가 LB가
# 763.90보다도 낮은 752.5로 나왔다 - "R shrinkage가 사실 도움이 됐다"는
# 뜻으로 오해할 뻔했지만, 실제 배포된 F seed0 체크포인트를 직접 평가해보니
# bss_F=33(F 단일시드 분산 범위 33~514 중 최악에 가까움)이 나왔다. 즉 F의
# DCN까지 같이 1시드로 되돌린 게 R shrinkage 실험에 다른 변수(F DCN
# 불안정성 재발)를 섞어버린 것이었다 - 그 결과는 무효. 다중시드 평균은
# 원래 검증된 대로 유지한다(R=3, F=5). R shrinkage 여부만 단독으로 다시
# 테스트해야 한다.
#
# R도 F처럼 3->5시드로 확장했다(2026-08-22, search_r_dcn_seeds.py) - 순수 분산
# 감소 시도라 용량/구조 변경이 아니고, 지금까지의 레벨 보정 계열 실패 패턴과
# 무관해 리스크가 낮다고 판단했다. R walk-forward 3구간 전부에서 동일 방향
# (양수) 확인: 2021->22 501->512(+11), 2022->23 661->676(+14), 2023->24
# 703->706(+3) - 폭은 F만큼 극적이지 않지만(R은 데이터가 많아 이미 비교적
# 안정적) 3구간 전부 일관되게 개선이라 채택.
#
# **2026-08-23 철회, 3으로 복귀.** DCN-only 비교(위)에서는 3/3 구간 양수였지만,
# isolate_r_vs_f_change.py로 4모델 블렌드 레벨에서 재확인하니 유효한 두 구간
# (F 데이터가 없는 2구간은 판단 불가) 모두 일관되게 -2였다(split1 ENSEMBLE
# 2251->2249, split3 731->729, bss_F는 이 구간에서 불변이라 전적으로 R
# DCN 탓). DCN 단독 정확도는 늘었지만 트리와의 다양성 기여가 그만큼 줄어든
# 것으로 보인다 - F 아키텍처 철회(위 DCN_F_PARAMS)와 동일한 실패 구조.
# 두 유효 구간에서 동일 크기(-2)로 일관됐다는 게 노이즈가 아니라 실제 효과라는
# 근거였다.
DCN_R_N_SEEDS = 3
DCN_F_N_SEEDS = 5

# pitcher/batter 저차원 임베딩을 R 5번째 앙상블 멤버로 추가한다(2026-08-23,
# try_pitcher_batter_embedding.py). CatBoost 범주형 ID 실패(위 rationale,
# R -29/-115/-122 - ordered target statistics가 시즌에 묶인 우연한 패턴을
# 암기)와 달리, (1) 저차원(8)으로 용량을 강하게 제한하고 (2) target statistic이
# 아니라 gradient descent로 학습되는 분산 표현을 쓰면 실패 패턴이 재발하지
# 않는다는 걸 확인했다. R 3구간 walk-forward, 5-way 블렌드(기존 4모델+embed)
# vs 기존 4-way 델타: +15/+7/+4 - 3구간 전부 일관되게 양수. 데이터가 커질수록
# 델타가 줄어드는 패턴(+15->+7->+4)이 뚜렷해서, 전체 데이터(2019~2024)로 학습되는
# 프로덕션에서는 이 추세를 한 단계 더 외삽해야 한다 - +4보다 작을 가능성까지
# 염두에 두되, 방향이 3구간 내내 일관되게 양수인 게 핵심 근거다. embed_dim/hidden
# 크기는 스윕 없이 첫 시도값을 그대로 채택했다(embed_dim을 키우면 다시 암기 위험이
# 있다는 게 설계 의도이므로 작게 유지) - 이후 시드 수 스윕(1/3/5/7)에서도 현재
# 3시드가 이미 최적권임을 재확인했다(2026-08-24, sweep_embed_r_seeds.py, 델타가
# n=1일 때 +8.4로 오히려 살짝 더 높았지만 스프레드가 작아 노이즈 수준으로 판단,
# 변경 안 함).
# **명명 규칙 주의(2026-08-24)**: 이 상수들은 R 전용인데도 원래 R 접두사가 없었다 -
# LGB_BASE_PARAMS가 F로 새어들어갔던 것과 같은 사고가 재발할 뻔해서, F 임베딩을
# 추가하면서 EMBED_R_* 로 전부 개명했다(기존 EMBED_DIM/EMBED_HIDDEN/EMBED_DROPOUT/
# EMBED_BATCH_SIZE -> EMBED_R_DIM/EMBED_R_HIDDEN/EMBED_R_DROPOUT/EMBED_R_BATCH_SIZE).
EMBED_R_DIM = 8
EMBED_R_HIDDEN = (64, 32)
EMBED_R_DROPOUT = 0.1
EMBED_R_BATCH_SIZE = 4096
EMBED_R_N_SEEDS = 3  # R DCN과 동일 관례 - 시드 편차가 클 수 있어 다중 시드 평균
# DCN_R_FINAL_EPOCHS와 동일한 방식(train<=2023->val=2024 early stopping,
# patience=8 기준 확인한 값)으로 도출 - 최종 모델은 held-out이 없어 이 고정값을 쓴다.
EMBED_R_FINAL_EPOCHS = 6

# F 임베딩(2026-08-24 채택, try_f_embedding.py/try_f_embed_asym_dim.py). R보다
# unseen ID 비율이 훨씬 높다(47.1% pitcher/43.2% batter, 유일한 유효 구간 기준 -
# R의 7~20%와 대비). hidden=(16,8)은 R의 (64,32)보다 훨씬 작게 잡았다 - DCN이
# F에서 훨씬 작은 네트워크를 원했던 것과 같은 논리(F 표본 자체가 작음).
# n_seeds=5는 F DCN과 동일 관례(표본 작을수록 시드 편차 커서 더 많은 평균 필요).
# whole-split bss_F 531->543(+12), 독립된 두 실행(대칭 embed_dim 첫 시도, 이후
# pitcher/batter 비대칭 dim 스윕의 대조군)에서 정확히 동일하게 재현됐다 - 우연이
# 아니라는 근거. 비대칭 dim(6,10)/(10,6) 둘 다 노이즈 수준(+1~+2)에 그쳐 대칭이
# 확실히 더 나았다 - F의 pitcher/batter 클래스 수 비율(240 vs 269, 1.12배)이
# 크지 않아 비대칭을 정당화하지 못했다. **주의**: F DCN 아키텍처(+96 로컬 ->
# LB 손해)와 F Optuna(+22 whole-split -> LB -1.46) 둘 다 "whole-split 기준으로도
# 좋아 보였는데 LB에서 배신당한" 전례라, +12라는 크기 자체도 크지 않으므로 기대치를
# 보수적으로 잡는다 - 반드시 단독 격리 제출로 실제 전이 여부를 확인한다.
EMBED_F_DIM = 8
EMBED_F_HIDDEN = (16, 8)
EMBED_F_DROPOUT = 0.2
EMBED_F_BATCH_SIZE = 1024
EMBED_F_N_SEEDS = 5
# train<=2023->val=2024 early stopping(patience=8) 기준 확인한 값.
EMBED_F_FINAL_EPOCHS = 10


class EmbedMLP(nn.Module):
    """pitcher_id/batter_id를 저차원 임베딩(index 0=unknown, 미확인 ID 전용)으로
    학습해 DCN과 동일한 전처리(DCN_FEATURES, fit_dcn_preprocessing/
    apply_dcn_preprocessing)를 거친 raw 피처와 concat 후 MLP로 예측한다."""

    def __init__(self, n_raw, n_pitchers, n_batters, embed_dim=EMBED_R_DIM, hidden=EMBED_R_HIDDEN, dropout=EMBED_R_DROPOUT):
        super().__init__()
        self.pitcher_emb = nn.Embedding(n_pitchers + 1, embed_dim)  # 0 = unknown
        self.batter_emb = nn.Embedding(n_batters + 1, embed_dim)
        layers, prev = [], n_raw + 2 * embed_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_raw, pid, bid):
        x = torch.cat([x_raw, self.pitcher_emb(pid), self.batter_emb(bid)], dim=1)
        return self.mlp(x).squeeze(-1)


def build_id_vocab(train_df, col):
    """train_df에서만 어휘를 만든다(val/test는 절대 포함하지 않음 - 누출 방지).
    0은 unknown(미확인 ID, 예: 2025년 신인)으로 예약한다."""
    return {v: i + 1 for i, v in enumerate(sorted(train_df[col].unique()))}


def encode_ids(df, col, vocab):
    return df[col].map(vocab).fillna(0).to_numpy(dtype=np.int64)


class CrossLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x0, x):
        return x0 * self.linear(x) + x


class DCNv2(nn.Module):
    def __init__(self, in_dim, cross_layers=3, deep_dims=(128, 64), dropout=0.1):
        super().__init__()
        self.cross_layers = nn.ModuleList([CrossLayer(in_dim) for _ in range(cross_layers)])
        deep, prev = [], in_dim
        for d in deep_dims:
            deep += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)]
            prev = d
        self.deep = nn.Sequential(*deep)
        self.out = nn.Linear(in_dim + deep_dims[-1], 1)

    def forward(self, x):
        xc = x
        for layer in self.cross_layers:
            xc = layer(x, xc)
        xd = self.deep(x)
        return self.out(torch.cat([xc, xd], dim=1)).squeeze(-1)


def fit_dcn_preprocessing(df):
    """DCNv2 입력 정규화 통계를 학습셋에서만 계산한다(누출 방지). DCN_FEATURES를
    쓴다(FEATURES가 아님) - shrunk_pitcher_rate는 제외(아래 apply_dcn_preprocessing
    참고).

    NaN이 있던 피처는 표준화 후 0으로 채우고, 결측 여부를 나타내는 지시
    피처를 별도로 덧붙인다 - 트리와 달리 신경망은 NaN을 자체적으로 분기
    처리할 수 없기 때문이다.
    """
    X = df[DCN_FEATURES].to_numpy(dtype=np.float64)
    miss_cols = [i for i in range(X.shape[1]) if np.isnan(X[:, i]).any()]
    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std == 0] = 1.0
    return {"mean": mean, "std": std, "miss_cols": miss_cols}


DCN_CLIP = 5.0


def apply_dcn_preprocessing(df, stats):
    """DCN_FEATURES만 쓰고(shrunk_pitcher_rate 제외 - asof_pitcher_success_rate와
    거의 중복이라 CrossLayer의 곱셈 구조에서 다중공선성으로 학습이 붕괴하는 것을
    확인했다, F에서 6개 시드 전부 bss_F=0), 표준화 후 +-DCN_CLIP으로 클리핑한다
    (F처럼 표본이 작은 subset은 극소수 pitcher/batter의 비율 피처가 z-score
    +25까지 튀는 경우가 있어 마찬가지로 CrossLayer를 불안정하게 만든다)."""
    X = df[DCN_FEATURES].to_numpy(dtype=np.float64)
    nan_mask = np.isnan(X)
    Xn = (X - stats["mean"]) / stats["std"]
    Xn = np.clip(Xn, -DCN_CLIP, DCN_CLIP)
    Xn[nan_mask] = 0.0
    if stats["miss_cols"]:
        miss_ind = nan_mask[:, stats["miss_cols"]].astype(np.float64)
        Xn = np.concatenate([Xn, miss_ind], axis=1)
    return Xn.astype(np.float32)


def add_matchup_features(df):
    df = df.copy()
    for a in (1, 2):
        for b in (1, 2):
            df[f"matchup_{a}_{b}"] = (
                (df["pitcher_hand"] == a) & (df["batter_hand"] == b)
            ).astype(np.float64)
    df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float64)
    return df


def brier_skill_score(y_true, y_pred):
    brier = brier_score_loss(y_true, y_pred)
    r = np.mean(y_true)
    baseline = r * (1 - r)
    bss = max(0.0, 100000 * (1 - brier / baseline))
    return brier, bss


def fit_platt(p, y):
    """logit(p) -> y 1차원 로지스틱 회귀(Platt scaling)를 적합한다. (coef, intercept)
    반환 - json으로 저장하기 쉽게 스칼라만 넘긴다."""
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    lr = LogisticRegression().fit(logit.reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p, coef, intercept):
    logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    z = coef * logit + intercept
    return 1.0 / (1.0 + np.exp(-z))
