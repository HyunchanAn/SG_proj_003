import shutil
import os

source_file = "pre-buff-1.png"
target_dir = "dataset/train/Metal_Hairline"
os.makedirs(target_dir, exist_ok=True)

if not os.path.exists(source_file):
    print(f"Error: {source_file} not found.")
else:
    for i in range(30):
        target_file = os.path.join(target_dir, f"aug_pre_buff_{i+1}.png")
        shutil.copy(source_file, target_file)
    print(f"Successfully created 30 augmented copies in {target_dir}")
