import os
import requests
import tarfile
import shutil
from tqdm import tqdm
from huggingface_hub import snapshot_download

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"Skipping download: {filename} already exists.")
        return
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

def setup_full_dataset():
    # 1. DTD (Describable Textures Dataset)
    dtd_url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    dtd_tar = "dtd.tar.gz"
    download_file(dtd_url, dtd_tar)
    
    if not os.path.exists("dtd"):
        print("Extracting DTD...")
        with tarfile.open(dtd_tar, "r:gz") as tar:
            tar.extractall()
    
    # 2. MINC-2500 (Materials in Context) - Hugging Face 사용
    print("Downloading MINC-2500 from Hugging Face...")
    minc_path = snapshot_download(repo_id="mcimpoi/minc-2500_split_1", repo_type="dataset", local_dir="minc-2500")

    # 3. V-SAMS 구조로 매핑
    dataset_root = "dataset/train"
    if os.path.exists(dataset_root):
        shutil.rmtree(dataset_root)
    os.makedirs(dataset_root, exist_ok=True)

    minc_map = {
        "metal": "Metal",
        "plastic": "Plastic",
        "glass": "Glass",
        "painted": "Painted",
        "wood": "Wood",
        "other": "Other",
        "ceramic": "Other",
        "carpet": "Other",
        "leather": "Other",
        "paper": "Other",
        "stone": "Other"
    }

    dtd_map = {
        "glossy": "Glossy",
        "bumpy": "Rough",
        "pitted": "Rough",
        "grooved": "Hairline",
        "matted": "Matte",
        "dotted": "Pattern",
        "striped": "Pattern",
        "grid": "Pattern",
        "scaly": "Rough",
        "meshed": "Pattern"
    }

    DEFAULT_MAT = "Other"
    DEFAULT_FIN = "Other"

    # 이미지 수집 루프 (MINC)
    print("Mapping MINC images from Parquet files...")
    import pandas as pd
    import io
    from PIL import Image

    # MINC Label ID Mapping (from README.md)
    MINC_LABELS = {
        0: "brick", 1: "carpet", 2: "ceramic", 3: "fabric", 4: "foliage",
        5: "food", 6: "glass", 7: "hair", 8: "leather", 9: "metal",
        10: "mirror", 11: "other", 12: "painted", 13: "paper", 14: "plastic",
        15: "polishedstone", 16: "skin", 17: "sky", 18: "stone", 19: "tile",
        20: "wallpaper", 21: "water", 22: "wood"
    }

    minc_data_dir = os.path.join("minc-2500", "data")
    if os.path.exists(minc_data_dir):
        parquet_files = [f for f in os.listdir(minc_data_dir) if f.endswith(".parquet")]
        
        for p_file in parquet_files:
            print(f"Processing {p_file}...")
            df = pd.read_parquet(os.path.join(minc_data_dir, p_file))
            
            # Iterate rows
            for idx, row in df.iterrows():
                label_id = row['label']
                label_name = MINC_LABELS.get(label_id, "unknown")
                
                # Check if this class is in our target map
                if label_name in minc_map:
                    mat = minc_map[label_name]
                    vsams_folder = f"{mat}_{DEFAULT_FIN}"
                    target_dir = os.path.join(dataset_root, vsams_folder)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Limit samples per class per file to avoid too many small files if not needed
                    # But since we want to retrain properly, let's take a good amount.
                    # Simple check: if folder has > 300 items, skip.
                    if len(os.listdir(target_dir)) >= 300:
                        continue
                        
                    # Save Image
                    img_bytes = row['image']['bytes']
                    img = Image.open(io.BytesIO(img_bytes))
                    
                    # Save as JPG
                    save_path = os.path.join(target_dir, f"minc_{label_name}_{idx}.jpg")
                    img.save(save_path)


    # 이미지 수집 루프 (DTD)
    print("Mapping DTD images...")
    dtd_images_base = os.path.join("dtd", "images")
    if os.path.exists(dtd_images_base):
        for dtd_class in os.listdir(dtd_images_base):
            src_dir = os.path.join(dtd_images_base, dtd_class)
            if not os.path.isdir(src_dir): continue
            
            fin = dtd_map.get(dtd_class, DEFAULT_FIN)
            vsams_folder = f"{DEFAULT_MAT}_{fin}"
            target_dir = os.path.join(dataset_root, vsams_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            for img in os.listdir(src_dir):
                shutil.copy(os.path.join(src_dir, img), os.path.join(target_dir, img))

    print("Full Dataset Bootstrapping Complete.")

if __name__ == "__main__":
    setup_full_dataset()
