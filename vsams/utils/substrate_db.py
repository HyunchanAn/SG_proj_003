from pathlib import Path

import numpy as np
import pandas as pd

from vsams.paths import DATA_DIR


class SubstrateDB:
    def __init__(self, excel_path=None):
        if excel_path is None:
            excel_path = DATA_DIR / "substrate_properties.xlsx"

        self.excel_path = Path(excel_path).resolve()
        self.df = None
        self.visual_library = None
        self.load_db()
        self.load_visual_library(DATA_DIR / "visual_library.pth")

    def load_db(self):
        if not self.excel_path.exists():
            print(f"Warning: Database file not found at {self.excel_path}")
            return

        try:
            # We use header=None and manually slice to avoid issues with multi-index headers.
            df_raw = pd.read_excel(self.excel_path, header=None)

            data = df_raw.iloc[6:].copy()
            mapping = {
                1: "no",
                2: "classification",
                4: "product_name",
                5: "roughness_md",
                6: "roughness_td",
                7: "gloss_md",
                8: "gloss_td",
                9: "energy_md",
                10: "energy_td",
            }

            self.df = data[list(mapping.keys())].rename(columns=mapping)
            self.df = self.df.dropna(subset=["product_name"]).reset_index(drop=True)

            numeric_cols = ["roughness_md", "roughness_td", "gloss_md", "gloss_td"]
            for col in numeric_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            self.df["roughness_avg"] = self.df[["roughness_md", "roughness_td"]].mean(axis=1)
            self.df["gloss_avg"] = self.df[["gloss_md", "gloss_td"]].mean(axis=1)
            self.df = self.df.dropna(subset=["roughness_avg", "gloss_avg"]).reset_index(drop=True)

            print(f"Successfully loaded {len(self.df)} products from DB.")
        except Exception as e:
            print(f"Error loading Excel DB: {e}")

    def load_visual_library(self, library_path):
        library_path = Path(library_path)
        if not library_path.exists():
            self.visual_library = None
            return

        import torch

        try:
            # weights_only=False is required because the library contains numpy arrays.
            self.visual_library = torch.load(library_path, weights_only=False)
            print(f"Loaded visual library with {len(self.visual_library)} products.")
        except Exception as e:
            print(f"Error loading visual library: {e}")
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
