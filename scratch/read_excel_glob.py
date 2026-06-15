import glob
import os
import sys

import pandas as pd

try:
    files = glob.glob("/Users/hyunchanan/Documents/GitHub/SG_proj_003/dataset/*.xlsx")
    if not files:
        print("No Excel files found.")
        sys.exit(1)

    file_path = files[0]
    print(f"Reading: {os.path.basename(file_path)}")

    xl = pd.ExcelFile(file_path)
    print(f"Sheets: {xl.sheet_names}")

    df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])
    print("--- FIRST 30 ROWS ---")
    # Increase max rows and columns for display
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 1000)
    print(df.head(30).to_string())
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
