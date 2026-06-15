import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from PIL import Image
from tqdm import tqdm

from vsams.analysis.surface_evaluator import SurfaceEvaluator

# --- 설정 ---
DATA_DIR = project_root / "dataset" / "verified"
CSV_PATH = DATA_DIR / "metadata.csv"


def recalculate():
    print("Recalculating 'Directionality' for existing data...")

    if not CSV_PATH.exists():
        print("Error: metadata.csv not found.")
        return

    # 엔진 로드
    evaluator = SurfaceEvaluator()

    # 데이터 로드
    df = pd.read_csv(CSV_PATH)

    new_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing images"):
        sample_id = row["id"]

        # 실제 코인 이미지 및 마스크 경로
        local_img_path = DATA_DIR / f"{sample_id}.jpg"
        mask_path = DATA_DIR / f"{sample_id}_mask.png"

        if not local_img_path.exists() or not mask_path.exists():
            # 만약 로컬에 없으면 원본 경로 시도
            local_img_path = Path(row["original_path"])
            if not local_img_path.exists():
                new_data.append(0.0)
                continue

        try:
            # 이미지 로드
            img = np.array(Image.open(local_img_path).convert("RGB"))
            mask_img = np.array(Image.open(mask_path).convert("L"))

            # 마스크 처리 (0보다 큰 영역을 분석 대상으로 설정)
            mask = mask_img > 0

            if not np.any(mask):
                new_data.append(0.0)
                continue

            # 바운딩 박스 계산
            coords = np.argwhere(mask)
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)

            mask_info = {
                "mask": mask,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            }

            # 방향성 계산
            val = evaluator.evaluate_directionality(img, mask_info)
            new_data.append(val)

        except Exception:
            # print(f"Error processing {sample_id}: {e}")
            new_data.append(0.0)

    # 데이터프레임 업데이트
    df["directionality"] = new_data

    # 저장
    df.to_csv(CSV_PATH, index=False)
    print(f"\nUpdate Complete! 'directionality' column added to {CSV_PATH}")


if __name__ == "__main__":
    recalculate()
