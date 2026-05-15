import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent))

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB

def run_performance_validation(test_data_dir):
    """
    테스트 데이터 디렉토리를 순회하며 성능을 검증합니다.
    디렉토리 구조: test_data_dir/{product_name}/*.jpg
    각 이미지에 대한 박스 정보는 {product_name}/metadata.csv에 있다고 가정하거나 
    수동 인터페이스를 활용합니다.
    """
    evaluator = SurfaceEvaluator()
    db = SubstrateDB()
    
    if db.df is None:
        print("Error: DB를 로드할 수 없습니다.")
        return

    results = []
    
    print("="*50)
    print(" V-SAMS 알고리즘 성능 검증 (Coin-Reflection)")
    print("="*50)

    # 실제 성능 검증 로직 (여기서는 구조적 가이드만 제공)
    # 팁: 실제 검증 시에는 각 이미지별로 수동 지정된 coin_box, ref_box 좌표가 필요합니다.
    
    # 예시 결과 요약 출력
    print(f"\n[대상 제품군]: {len(db.df)} 종")
    print("-" * 50)
    print(f"{'Product':<15} | {'Ref Ra':<8} | {'Ref Gloss':<8}")
    print("-" * 50)
    for _, row in db.df.sort_values("product_name").iterrows():
        print(f"{row['product_name']:<15} | {row['roughness_avg']:<8.4f} | {row['gloss_avg']:<8.1f}")
    
    print("\n[알고리즘 검증 방법 가이드]")
    print("1. 각 제품별 대표 사진에서 동전과 반사광 영역 좌표를 획득하세요.")
    print("2. evaluator.analyze(img, custom_boxes=[coin_box, ref_box])를 호출하세요.")
    print("3. 반환된 roughness, gloss를 위 기준값과 비교하여 오차(Error)를 기록하세요.")
    print("-" * 50)
    print("성능 평가 스크립트 준비 완료.")

if __name__ == "__main__":
    run_performance_validation("dataset/verified")
