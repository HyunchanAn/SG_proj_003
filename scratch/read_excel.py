import sys

import pandas as pd

try:
    file_path = "/Users/hyunchanan/Documents/GitHub/SG_proj_003/dataset/세계화학 제품분류_6.xlsx"
    df = pd.read_excel(file_path)
    print("--- EXCEL CONTENT START ---")
    print(df.to_string())
    print("--- EXCEL CONTENT END ---")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
