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
    def __init__(self, excel_path=None, use_api=True):
        if excel_path is None:
            excel_path = DATA_DIR / "substrate_properties.xlsx"

        self.excel_path = Path(excel_path).resolve()
        self.df = None
        self.visual_library = None

        # 기본 경로 설정 (기존 코드 유지)
        if excel_path is None:
            default_excel = DATA_DIR / "substrate_properties.xlsx"
            if default_excel.exists():
                self.excel_path = default_excel
            else:
                # 사용자가 제공한 새 엑셀 파일이 루트에 있음
                self.excel_path = Path("피착재 종류 및 물성(AI화)2.xlsx").resolve()
        else:
            self.excel_path = Path(excel_path).resolve()

        # 004 API 우선 시도, 실패 시 로컬 Excel 폴백
        loaded = False
        if use_api:
            loaded = self.load_from_api()

        if not loaded:
            if "AI화" in str(self.excel_path):
                self.load_new_db()
            else:
                self.load_db()

        self.load_visual_library(DATA_DIR / "visual_library.pth")

    def load_from_api(self) -> bool:
        """004 DB API에서 피착재 물성 데이터를 가져옵니다. 실패 시 False를 반환합니다."""
        try:
            res = httpx.get(f"{MODULE_004_URL}/adherend-properties", timeout=5.0)
            res.raise_for_status()
            records = res.json()
            if not records:
                logger.warning("004 API returned empty adherend-properties list.")
                return False

            self.df = pd.DataFrame(records)

            # API 응답 필드명을 내부 표준 필드명으로 매핑
            rename_map = {
                "product_name": "product_name",
                "roughness_md": "roughness_md",
                "roughness_td": "roughness_td",
                "gloss_md": "gloss_md",
                "gloss_td": "gloss_td",
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
            return True
        except Exception as e:
            logger.warning(f"004 API unavailable, falling back to local Excel: {e}")
            return False


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

            self.df["roughness_avg"] = self.df[["roughness_md", "roughness_td"]].mean(
                axis=1
            )
            self.df["gloss_avg"] = self.df[["gloss_md", "gloss_td"]].mean(axis=1)
            self.df = self.df.dropna(subset=["roughness_avg", "gloss_avg"]).reset_index(
                drop=True
            )

            print(f"Successfully loaded {len(self.df)} products from DB.")
        except Exception as e:
            print(f"Error loading Excel DB: {e}")

    def load_new_db(self):
        """사용자가 제공한 '피착재 종류 및 물성(AI화)2.xlsx' 파일 전용 로더"""
        if not self.excel_path.exists():
            print(f"Warning: New database file not found at {self.excel_path}")
            return

        try:
            # 7행(Index 6)부터 데이터 시작
            df_raw = pd.read_excel(self.excel_path, header=None)
            data = df_raw.iloc[6:].copy()
            mapping = {
                4: "product_name",
                5: "roughness_md",
                6: "roughness_td",
                7: "gloss_md",
                8: "gloss_td",
            }
            self.df = data[list(mapping.keys())].rename(columns=mapping)

            # 5가지 대상 품목만 필터링 (BA, #4, HL, SM, 2B)
            targets = ["BA", "#4", "HL", "SM", "2B"]
            self.df = self.df[
                self.df["product_name"].str.contains("|".join(targets), na=False)
            ].reset_index(drop=True)

            numeric_cols = ["roughness_md", "roughness_td", "gloss_md", "gloss_td"]
            for col in numeric_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

            self.df["roughness_avg"] = self.df[["roughness_md", "roughness_td"]].mean(
                axis=1
            )
            self.df["gloss_avg"] = self.df[["gloss_md", "gloss_td"]].mean(axis=1)
            self.df = self.df.dropna(subset=["roughness_avg", "gloss_avg"]).reset_index(
                drop=True
            )

            print(f"Successfully loaded {len(self.df)} target products from new DB.")
        except Exception as e:
            print(f"Error loading New Excel DB: {e}")

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
