from pathlib import Path
import os

import httpx
import numpy as np
import pandas as pd
from loguru import logger

from vsams.paths import DATA_DIR

# 004 DB API URL from env
MODULE_004_URL = os.getenv("MODULE_004_URL", "http://localhost:8004")


class SubstrateDB:
    def __init__(self):
        self.df = None
        self.visual_library = None

        self._load_from_api()
        self.load_visual_library(DATA_DIR / "visual_library.pth")

    def _load_from_api(self):
        """004 DB API에서 피착재 물성 데이터를 가져옵니다."""
        try:
            res = httpx.get(f"{MODULE_004_URL}/adherend-properties", timeout=5.0)
            res.raise_for_status()
            records = res.json()
            if not records:
                logger.error("004 API returned empty adherend-properties list.")
                return

            self.df = pd.DataFrame(records)

            # API 응답 필드명을 내부 표준 필드명으로 매핑
            rename_map = {
                "surface_energy_md": "energy_md",
                "surface_energy_td": "energy_td",
            }
            self.df = self.df.rename(columns=rename_map)

            numeric_cols = ["roughness_md", "roughness_td", "gloss_md", "gloss_td"]
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            if "roughness_md" in self.df.columns and "roughness_td" in self.df.columns:
                self.df["roughness_avg"] = self.df[["roughness_md", "roughness_td"]].mean(axis=1)
            if "gloss_md" in self.df.columns and "gloss_td" in self.df.columns:
                self.df["gloss_avg"] = self.df[["gloss_md", "gloss_td"]].mean(axis=1)

            self.df = self.df.dropna(subset=["roughness_avg", "gloss_avg"]).reset_index(drop=True)
            logger.info(f"Loaded {len(self.df)} products from 004 API ({MODULE_004_URL}).")
        except Exception as e:
            logger.error(f"Failed to load substrate data from 004 API: {e}")

    def load_visual_library(self, library_path):
        library_path = Path(library_path)
        if not library_path.exists():
            self.visual_library = None
            return

        import torch

        try:
            # weights_only=False is required because the library contains numpy arrays.
            self.visual_library = torch.load(library_path, weights_only=False)
            logger.info(f"Loaded visual library with {len(self.visual_library)} products.")
        except Exception as e:
            logger.error(f"Error loading visual library: {e}")
            self.visual_library = None

    def find_closest(self, roughness, glossiness):
        if self.df is None or self.df.empty:
            return None

        r_err = (self.df["roughness_avg"] - roughness) / 0.1
        g_err = (self.df["gloss_avg"] - glossiness) / 10.0
        self.df["distance"] = np.sqrt(r_err**2 + g_err**2)

        closest_idx = self.df["distance"].idxmin()
        closest = self.df.loc[closest_idx]
        return closest.to_dict()

    def find_closest_top_k(self, roughness, glossiness, k=5):
        if self.df is None or self.df.empty:
            return []

        r_err = (self.df["roughness_avg"] - roughness) / 0.1
        g_err = (self.df["gloss_avg"] - glossiness) / 10.0
        self.df["distance"] = np.sqrt(r_err**2 + g_err**2)
        top_k = self.df.nsmallest(k, "distance")

        results = []
        for _, row in top_k.iterrows():
            results.append(
                {
                    "product_name": row["product_name"],
                    "distance": float(row["distance"]),
                    "roughness": float(row["roughness_avg"]),
                    "gloss": float(row["gloss_avg"]),
                }
            )
        return results

    def find_visual_match(self, input_features, k=5):
        if self.visual_library is None:
            return None

        matches = []
        feat_norm = np.linalg.norm(input_features)

        for item in self.visual_library:
            ref_features = item["features"]
            dot = np.dot(ref_features, input_features)
            ref_norms = np.linalg.norm(ref_features, axis=1)
            similarities = dot / (ref_norms * feat_norm + 1e-8)

            score = float(np.mean(similarities))
            matches.append(
                {
                    "product_name": item["product_name"],
                    "similarity": score,
                    "ref_image": item["ref_image"],
                }
            )

        matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        return matches[:k]


if __name__ == "__main__":
    db = SubstrateDB()
    if db.df is not None:
        print(db.df[["product_name", "roughness_avg", "gloss_avg"]].head())
        res = db.find_closest(0.4, 20.0)
        print("\nClosest Match for (0.4, 20.0):")
        if res:
            print(
                f"Name: {res['product_name']}, "
                f"Ra: {res['roughness_avg']:.4f}, "
                f"Gloss: {res['gloss_avg']:.2f}"
            )
