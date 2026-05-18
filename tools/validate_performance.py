import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from vsams.analysis.surface_evaluator import SurfaceEvaluator
from vsams.utils.substrate_db import SubstrateDB


def run_performance_validation(test_data_dir: str) -> None:
    """
    Performs benchmark performance validation over the verified steel samples.
    """
    evaluator = SurfaceEvaluator()  # noqa: F841
    db = SubstrateDB()

    if db.df is None:
        print("Error: Failed to load DB.")
        return

    results: List[Dict[str, Any]] = []  # noqa: F841

    print("=" * 50)
    print(" V-SAMS Performance Validation (Coin-Reflection)")
    print("=" * 50)

    # Actual validation iteration guide
    print(f"Target data directory: {test_data_dir}")

    # Summary output
    print(f"\n[Reference Products: {len(db.df)} types]")
    print("-" * 50)
    print(f"{'Product':<15} | {'Ref Ra':<8} | {'Ref Gloss':<8}")
    print("-" * 50)
    for _, row in db.df.sort_values("product_name").iterrows():
        print(
            f"{row['product_name']:<15} | {row['roughness_avg']:<8.4f} | {row['gloss_avg']:<8.1f}"
        )

    print("\n[Validation Methodology Guide]")
    print("1. Obtain reference coin and reflection target bounding box coordinates.")
    print("2. Call evaluator.analyze(img, custom_boxes=[coin_box, ref_box]).")
    print("3. Record prediction roughness/gloss errors against baseline properties.")
    print("-" * 50)
    print("Performance validation stub ready.")


if __name__ == "__main__":
    run_performance_validation("dataset/verified")
