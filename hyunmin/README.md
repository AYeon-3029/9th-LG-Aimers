pandas
numpy
scikit-learn
joblib
lightgbm
xgboost



import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

def main():
    print("1. 데이터 불러오기...")
    DATA_DIR = "./data"
    OUT_DIR = "./output"
    
    # 🚀 시간 초과 걱정 끝! 147만 개 전체 데이터를 다시 100% 투입합니다.
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), encoding="utf-8-sig")
    sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"), encoding="utf-8-sig")
    
    ID_COL = "row_id"
    TARGET_COL = "control_success"

    print("2. [유지] 적중률이 증명된 현민님 핵심 가설 + 기싸움/최근 폼 피처...")
    for df in [train, test]:
        df['crisis_pressure'] = df['li'] * (df['num_runners_on'] + 1)
        df['is_full_count'] = ((df['balls_before'] == 3) & (df['strikes_before'] == 2)).astype(int)
        
        max_rate = df[['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate']].max(axis=1)
        df['max_pitch_rate'] = max_rate.fillna(0)
        df['stubbornness_index'] = df['max_pitch_rate'] * df['li']
        
        df['control_struggle_index'] = df['asof_pitcher_ball_rate'].fillna(0) * df['li']
        df['is_scoring_pos'] = ((df['runner_on_2b'] == 1) | (df['runner_on_3b'] == 1)).astype(int)

        df['pitcher_batter_diff'] = df['asof_pitcher_success_rate'].fillna(0.5) - df['asof_batter_success_rate'].fillna(0.5)
        df['recent_trend'] = (df['asof_pitcher_prev1_game_success_rate'].fillna(df['asof_pitcher_success_rate']) - df['asof_pitcher_success_rate'].fillna(0.5)).fillna(0)
        df['is_same_hand'] = (df['pitcher_hand'] == df['batter_hand']).astype(int)

    print("3. 안전한 전처리 파이프라인...")
    CAT_COLS = ["top_bottom", "game_type", "base_state"]
    FEATURES = [c for c in test.columns if c != ID_COL]
    NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

    X_train = train[FEATURES]
    y_train = train[TARGET_COL]
    X_test = test[FEATURES]
    test_ids = test[ID_COL].tolist()

    preprocessor = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), NUM_COLS),
    ])

    print("4. [엔진 전면 교체] 초고속/고성능 HistGradientBoosting 가동...")
    model = Pipeline([
        ("pre", preprocessor),
        ("clf", HistGradientBoostingClassifier(
            max_iter=300,            # 300번 연속으로 정밀하게 오차를 깎아냅니다 (속도 매우 빠름)
            learning_rate=0.05,      # 천천히, 정교하게 학습하여 BSS 점수를 방어
            max_depth=12,            # 가설 변수들의 복잡한 상호작용을 깊게 학습
            min_samples_leaf=150,    # 과도한 확신(Overfitting) 방어
            l2_regularization=0.1,   # 🌟 BSS 최적화: 극단적 확률 부여를 통계적으로 억제
            random_state=42,
        ))
    ])
    
    model.fit(X_train, y_train)

    print("5. 예측 및 정답지 병합...")
    preds = model.predict_proba(X_test)[:, 1]

    pred_map = dict(zip(test_ids, preds))
    values = []
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        values.append(pred_map.get(rid, cur))
    
    sub[TARGET_COL] = values

    print("6. 제출 파일 저장 완료!")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
