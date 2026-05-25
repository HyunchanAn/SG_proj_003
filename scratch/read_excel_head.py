import pandas as pd
import sys

try:
    file_path = "/Users/hyunchanan/Documents/GitHub/SG_proj_003/dataset/세계화학 제품분류_6.xlsx"
    # Read sheet names first
    xl = pd.ExcelFile(file_path)
    print(f"Sheets: {xl.sheet_names}")
    
    # Read first sheet
    df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])
    print("--- FIRST 20 ROWS ---")
    print(df.head(20).to_string())
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
