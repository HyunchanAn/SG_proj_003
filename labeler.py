import streamlit as st
import os
import shutil
from datetime import datetime
from PIL import Image
import hashlib

# --- Config ---
DATASET_ROOT = "dataset"
MATERIALS = ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"]
FINISHES = ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]

st.set_page_config(page_title="V-SAMS Data Labeler", page_icon="🏷️", layout="centered")

# --- Helper Functions ---
def get_class_name(material, finish):
    return f"{material}_{finish}"

def save_image(uploaded_file, material, finish):
    # 1. Prepare Directory
    class_name = get_class_name(material, finish)
    save_dir = os.path.join(DATASET_ROOT, "train", class_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. Generate Sanitized Filename (Safely handle non-English chars)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    original_name = uploaded_file.name
    name, ext = os.path.splitext(original_name)
    
    # Create a short hash of the original name for uniqueness and safety
    name_hash = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
    new_filename = f"{timestamp}_{name_hash}{ext}"
    save_path = os.path.join(save_dir, new_filename)
    
    # 3. Save
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return save_path

def count_stats():
    stats = {}
    if not os.path.exists(os.path.join(DATASET_ROOT, "train")):
        return stats
        
    for class_name in os.listdir(os.path.join(DATASET_ROOT, "train")):
        class_dir = os.path.join(DATASET_ROOT, "train", class_name)
        if os.path.isdir(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[class_name] = count
    return stats

# --- UI Layout ---
st.title("🏷️ 간편 데이터 라벨러")
st.markdown("이미지를 업로드하고 재질(Material)과 마감(Finish)을 선택하면 자동으로 폴더에 정리해줍니다.")

# 1. Stats Overview
with st.expander("📊 현재 수집 현황 보기", expanded=False):
    stats = count_stats()
    if stats:
        st.bar_chart(stats)
        st.write(stats)
    else:
        st.info("아직 수집된 데이터가 없습니다.")

st.divider()

# 2. Upload Section
st.subheader("1. 사진 선택")
uploaded_files = st.file_uploader("사진을 드래그하거나 선택하세요 (여러 장 가능)", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    st.subheader("2. 라벨 선택 및 저장")
    
    # Global Settings for Batch Processing
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**모재 (Material)**")
        selected_material = st.radio("Material", MATERIALS, label_visibility="collapsed")
    with col2:
        st.markdown("**마감 (Finish)**")
        selected_finish = st.radio("Finish", FINISHES, label_visibility="collapsed")
    
    target_class = get_class_name(selected_material, selected_finish)
    st.info(f"📂 저장될 폴더명: **dataset/train/{target_class}/**")
    
    if st.button("💾 이 설정으로 모든 사진 저장하기", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        saved_count = 0
        
        for i, file in enumerate(uploaded_files):
            save_path = save_image(file, selected_material, selected_finish)
            saved_count += 1
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        st.success(f"✅ {saved_count}장의 사진을 '{target_class}' 폴더에 저장했습니다!")
        st.balloons()
        
    # Preview
    st.divider()
    st.caption(f"미리보기 ({len(uploaded_files)}장 선택됨)")
    
    # Show first 3 images as preview
    cols = st.columns(3)
    for i, file in enumerate(uploaded_files[:3]):
        with cols[i]:
            st.image(file, use_container_width=True)
            
    if len(uploaded_files) > 3:
        st.write(f"...외 {len(uploaded_files)-3}장")

else:
    st.info("👆 먼저 위에서 사진을 업로드해주세요.")
