import pandas as pd
import numpy as np
import os

class SubstrateDB:
    def __init__(self, excel_path='dataset/피착재 종류 및 물성(AI화)2.xlsx'):
        self.excel_path = os.path.abspath(excel_path)
        self.df = None
        self.visual_library = None
        self.load_db()
        # Use absolute path for visual library to ensure it's found in Streamlit context
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_path = os.path.join(base_dir, 'data', 'visual_library.pth')
        self.load_visual_library(lib_path)

    def load_db(self):
        if not os.path.exists(self.excel_path):
            print(f"Warning: Database file not found at {self.excel_path}")
            return

        try:
            # We use header=None and manually slice to avoid encoding issues with multi-index headers
            df_raw = pd.read_excel(self.excel_path, header=None)
            
            # Data usually starts from row 6 (0-indexed)
            # Column mapping (0-indexed):
            # 1: No, 2: Classification, 4: Product Name, 
            # 5: Ra MD, 6: Ra TD, 7: Gloss MD, 8: Gloss TD, 9: Energy MD, 10: Energy TD
            
            data = df_raw.iloc[6:].copy()
            mapping = {
                1: 'no',
                2: 'classification',
                4: 'product_name',
                5: 'roughness_md',
                6: 'roughness_td',
                7: 'gloss_md',
                8: 'gloss_td',
                9: 'energy_md',
                10: 'energy_td'
            }
            
            # Keep only mapped columns
            self.df = data[list(mapping.keys())].rename(columns=mapping)
            
            # Clean: Drop rows with no product name
            self.df = self.df.dropna(subset=['product_name']).reset_index(drop=True)
            
            # Convert numeric columns
            numeric_cols = ['roughness_md', 'roughness_td', 'gloss_md', 'gloss_td']
            for col in numeric_cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Recalculate averages
            self.df['roughness_avg'] = self.df[['roughness_md', 'roughness_td']].mean(axis=1)
            self.df['gloss_avg'] = self.df[['gloss_md', 'gloss_td']].mean(axis=1)
            
            # Drop products with NO property data
            self.df = self.df.dropna(subset=['roughness_avg', 'gloss_avg']).reset_index(drop=True)
            
            print(f"Successfully loaded {len(self.df)} products from DB.")
        except Exception as e:
            print(f"Error loading Excel DB: {e}")

    def load_visual_library(self, library_path='vsams/data/visual_library.pth'):
        if os.path.exists(library_path):
            import torch
            try:
                # Use weights_only=False because the library contains numpy arrays
                self.visual_library = torch.load(library_path, weights_only=False)
                print(f"Loaded visual library with {len(self.visual_library)} products.")
            except Exception as e:
                print(f"Error loading visual library: {e}")
                self.visual_library = None
        else:
            self.visual_library = None

    def find_closest(self, roughness, glossiness):
        """
        Find the closest product based on estimated roughness (Ra) and glossiness (%).
        """
        if self.df is None or self.df.empty:
            return None

        # Roughness typical range: 0.35 - 0.8
        # Gloss typical range: 10 - 40
        # Normalize weights
        # Roughness error 0.1 is significant, Gloss error 5.0 is significant.
        
        r_err = (self.df['roughness_avg'] - roughness) / 0.1
        g_err = (self.df['gloss_avg'] - glossiness) / 10.0
        
        self.df['distance'] = np.sqrt(r_err**2 + g_err**2)
        
        closest_idx = self.df['distance'].idxmin()
        closest = self.df.loc[closest_idx]
        return closest.to_dict()

    def find_closest_top_k(self, roughness, glossiness, k=5):
        """
        Find TOP-K closest products based on properties.
        """
        if self.df is None or self.df.empty:
            return []
            
        r_err = (self.df['roughness_avg'] - roughness) / 0.1
        g_err = (self.df['gloss_avg'] - glossiness) / 10.0
        
        self.df['distance'] = np.sqrt(r_err**2 + g_err**2)
        top_k = self.df.nsmallest(k, 'distance')
        
        results = []
        for _, row in top_k.iterrows():
            results.append({
                'product_name': row['product_name'],
                'distance': float(row['distance']),
                'roughness': float(row['roughness_avg']),
                'gloss': float(row['gloss_avg'])
            })
        return results

    def find_visual_match(self, input_features, k=5):
        """
        Find all product matches ranked by visual feature similarity.
        """
        if self.visual_library is None:
            return None
            
        matches = []
        feat_norm = np.linalg.norm(input_features)
        
        for item in self.visual_library:
            ref_features = item['features'] # shape (N, 2048)
            dot = np.dot(ref_features, input_features)
            ref_norms = np.linalg.norm(ref_features, axis=1)
            similarities = dot / (ref_norms * feat_norm + 1e-8)
            
            score = float(np.mean(similarities))
            matches.append({
                'product_name': item['product_name'],
                'similarity': score,
                'ref_image': item['ref_image']
            })
        
        # Sort by similarity descending
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        return matches[:k]

if __name__ == "__main__":
    db = SubstrateDB()
    if db.df is not None:
        print(db.df[['product_name', 'roughness_avg', 'gloss_avg']].head())
        res = db.find_closest(0.4, 20.0)
        print("\nClosest Match for (0.4, 20.0):")
        if res:
            print(f"Name: {res['product_name']}, Ra: {res['roughness_avg']:.4f}, Gloss: {res['gloss_avg']:.2f}")
