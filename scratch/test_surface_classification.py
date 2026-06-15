import sys
from pathlib import Path

from PIL import Image

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB


def run_test(sample_per_folder=5):
    db = SubstrateDB()
    evaluator = SurfaceEvaluator()

    test_root = Path("test_260420_surface")
    folders = [
        "260420_BA_reflect",
        "260420_#4_reflect",
        "260420_HL_reflect",
        "260420_SM_reflect",
        "260420_2B_reflect",
    ]

    # 폴더명에서 실제 라벨 추출 (예: BA, #4, HL, SM, 2B)
    label_map = {
        "260420_BA_reflect": "BA",
        "260420_#4_reflect": "#4",
        "260420_HL_reflect": "HL",
        "260420_SM_reflect": "SM",
        "260420_2B_reflect": "2B",
    }

    results = []

    print(
        f"{'Folder':<20} | {'Sample':<5} | {'Pred':<5} | {'Ra':<6} | {'Gloss':<6} | {'Match'}"
    )
    print("-" * 70)

    correct = 0
    total = 0

    for folder_name in folders:
        folder_path = test_root / folder_name
        if not folder_path.exists():
            print(f"Skipping {folder_name} (Not found)")
            continue

        gt_label = label_map[folder_name]
        images = list(folder_path.glob("*.jpg"))[:sample_per_folder]

        for img_path in images:
            try:
                img = Image.open(img_path)
                analysis = evaluator.analyze(img)

                # DB에서 가장 가까운 제품 찾기
                match = db.find_closest(analysis["roughness"], analysis["gloss"])
                pred_label = match["product_name"] if match else "Unknown"

                is_correct = (
                    gt_label in pred_label
                )  # 예측된 제품명에 정답 라벨이 포함되어 있는지 확인
                if is_correct:
                    correct += 1
                total += 1

                print(
                    f"{gt_label:<20} | {img_path.name[-8:]} | {pred_label:<5} | {analysis['roughness']:.3f} | {analysis['gloss']:.1f} | {is_correct}"
                )

                results.append(
                    {
                        "gt": gt_label,
                        "pred": pred_label,
                        "ra": analysis["roughness"],
                        "gloss": analysis["gloss"],
                        "correct": is_correct,
                    }
                )
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print("-" * 70)
    print(f"Final Accuracy: {accuracy:.2f}% ({correct}/{total})")

    return results


if __name__ == "__main__":
    run_test(sample_per_folder=3)  # 빠른 확인을 위해 폴더당 3개씩 테스트
