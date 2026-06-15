import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent

# --- 설정 ---
DATA_PATH = project_root / "dataset" / "verified" / "metadata.csv"
MODEL_SAVE_PATH = project_root / "vsams" / "models" / "surface_classifier.joblib"
FEATURES = ["roughness", "gloss", "directionality"]
LABEL = "label"


def train():
    print("🚀 V-SAMS 표면 판별 모델 학습 시작...")

    # 1. 데이터 로드
    if not DATA_PATH.exists():
        print(f"❌ 에러: 데이터 파일을 찾을 수 없습니다 ({DATA_PATH})")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"📊 총 {len(df)}개의 원본 데이터 로드 완료.")

    # 2. 전처리 (Good 데이터만 선별)
    # Note: CSV에 저장될 때 'Good ✅' 문자열로 저장됨
    df_clean = df[df["mask_quality"].str.contains("Good", na=False)].copy()
    print(f"🧹 'Good' 품질 데이터 {len(df_clean)}개 선별 완료.")

    if len(df_clean) < 10:
        print("❌ 에러: 학습에 필요한 데이터가 너무 부족합니다.")
        return

    X = df_clean[FEATURES].values
    y = df_clean[LABEL].values

    # 3. 5-Fold 교차 검증 준비
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []

    print("\n🔄 5-Fold 교차 검증 수행 중...")
    for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 모델 정의 및 학습
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        fold_accuracies.append(acc)
        print(f"   - Fold {i+1} 정확도: {acc:.4f}")

    avg_acc = np.mean(fold_accuracies)
    print(f"\n✅ 교차 검증 평균 정확도: {avg_acc:.4f}")

    # 4. 전체 데이터로 최종 모델 학습
    print("\n🎯 전체 데이터셋으로 최종 모델 학습 중...")
    final_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    )
    final_model.fit(X, y)

    # 5. 모델 저장
    os.makedirs(MODEL_SAVE_PATH.parent, exist_ok=True)
    joblib.dump(final_model, MODEL_SAVE_PATH)
    print(f"💾 모델 저장 완료: {MODEL_SAVE_PATH}")

    # 6. 최종 성능 리포트 (전체 데이터 대상)
    y_final_pred = final_model.predict(X)
    print("\n--- 최종 학습 결과 리포트 ---")
    print(classification_report(y, y_final_pred))

    print("\n--- 혼동 행렬 (Confusion Matrix) ---")
    labels = sorted(df_clean[LABEL].unique())
    cm = confusion_matrix(y, y_final_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    print("\n✨ 모든 과정이 완료되었습니다.")


if __name__ == "__main__":
    train()
